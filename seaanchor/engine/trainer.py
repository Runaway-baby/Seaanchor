import os
import logging
import time
import weakref
from collections import OrderedDict, Counter
import copy

import numpy as np

import torch
from fvcore.nn import get_bn_modules
from torch.utils.data.sampler import Sampler

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.engine import hooks, HookBase
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.solver.build import maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, get_detection_dataset_dicts
from detectron2.utils.events import get_event_storage

from detectron2.utils import comm
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format

from detectron2.solver import LRMultiplier
from detectron2.modeling import build_model
from detectron2.structures import Boxes

from fvcore.common.param_scheduler import ParamScheduler
from seaanchor.data.nautilus import register_dataset

from detectron2.data import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as detT

import torchvision.transforms as T
import torchvision.transforms.functional as tF

from seaanchor.modeling import add_seaanchor_config

logger = logging.getLogger("detectron2")


class EpochLogger(HookBase):
    def __init__(self, iters_per_epoch, total_epochs=0, log_period=1, eval_period=0):
        self.iters_per_epoch = int(iters_per_epoch)
        self.total_epochs = int(total_epochs)
        self.log_period = max(1, int(log_period))
        self.eval_period = int(eval_period)
        self._last_started_epoch = None

    def _enabled(self):
        return self.iters_per_epoch > 0 and comm.is_main_process()

    def _total_epochs(self):
        if self.total_epochs > 0:
            return self.total_epochs
        return int(np.ceil(float(self.trainer.max_iter) / float(self.iters_per_epoch)))

    def _should_log_epoch(self, epoch):
        return epoch == 1 or epoch == self._total_epochs() or epoch % self.log_period == 0

    def before_train(self):
        if not self._enabled():
            return
        logger.info(
            "Epoch logging enabled: iters_per_epoch=%d, total_epochs=%d, start_iter=%d, max_iter=%d",
            self.iters_per_epoch,
            self._total_epochs(),
            self.trainer.iter,
            self.trainer.max_iter,
        )

    def before_step(self):
        if not self._enabled():
            return
        if self.trainer.iter % self.iters_per_epoch != 0:
            return
        epoch = self.trainer.iter // self.iters_per_epoch + 1
        if epoch > self._total_epochs() or epoch == self._last_started_epoch:
            return
        self._last_started_epoch = epoch
        if self._should_log_epoch(epoch):
            logger.info(
                "[Epoch %d/%d] start, iter=%d/%d",
                epoch,
                self._total_epochs(),
                self.trainer.iter,
                self.trainer.max_iter,
            )

    def after_step(self):
        if not self._enabled():
            return
        next_iter = self.trainer.iter + 1
        is_epoch_end = next_iter % self.iters_per_epoch == 0
        is_final = next_iter >= self.trainer.max_iter
        if not is_epoch_end and not is_final:
            return
        epoch = int(np.ceil(float(next_iter) / float(self.iters_per_epoch)))
        epoch = min(epoch, self._total_epochs())
        if self._should_log_epoch(epoch):
            logger.info(
                "[Epoch %d/%d] end, iter=%d/%d",
                epoch,
                self._total_epochs(),
                next_iter,
                self.trainer.max_iter,
            )
        if self.eval_period > 0 and next_iter % self.eval_period == 0:
            logger.info("[Epoch %d/%d] evaluation/checkpoint scheduled at iter=%d", epoch, self._total_epochs(), next_iter)


def setup(args):
    cfg = get_cfg()
    add_seaanchor_config(cfg)
    # hack to add base yaml
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
        self.with_crops = cfg.INPUT.CLIP_WITH_IMG
        self.with_random_clip_crops = cfg.INPUT.CLIP_RANDOM_CROPS
        self.with_jitter = cfg.INPUT.IMAGE_JITTER
        self.cropfn = T.RandomCrop  # T.RandomCrop([224,224])
        self.aug = T.ColorJitter(brightness=.5, hue=.3)
        self.crop_size = cfg.INPUT.RANDOM_CROP_SIZE

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = detT.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if self.with_jitter:
            dataset_dict["jitter_image"] = self.aug(dataset_dict["image"])

        if self.with_crops:
            bbox = dataset_dict['instances'].gt_boxes.tensor
            csx = (bbox[:, 0] + bbox[:, 2]) * 0.5
            csy = (bbox[:, 1] + bbox[:, 3]) * 0.5
            maxwh = torch.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])
            crops = list()
            gt_boxes = list()
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            for cx, cy, maxdim, label, box in zip(csx, csy, maxwh, dataset_dict['instances'].gt_classes, bbox):

                if int(maxdim) < 10:
                    continue
                x0 = torch.maximum(cx - maxdim * 0.5, torch.tensor(0))
                y0 = torch.maximum(cy - maxdim * 0.5, torch.tensor(0))
                try:
                    imcrop = T.functional.resized_crop(dataset_dict['image'], top=int(y0), left=int(x0),
                                                       height=int(maxdim), width=int(maxdim), size=224)
                    imcrop = imcrop.flip(0) / 255  # bgr --> rgb for clip
                    imcrop = T.functional.normalize(imcrop, mean, std)
                    # print(x0,y0,x0+maxdim,y0+maxdim,dataset_dict['image'].shape)
                    # print(imcrop.min(),imcrop.max() )
                    gt_boxes.append(box.reshape(1, -1))
                except Exception as e:
                    print(e)
                    print('crops:', x0, y0, maxdim)
                    exit()
                # crops.append((imcrop,label))
                crops.append(imcrop.unsqueeze(0))

            if len(crops) == 0:
                dataset_dict['crops'] = []
            else:
                dataset_dict['crops'] = [torch.cat(crops, 0), Boxes(torch.cat(gt_boxes, 0))]

        if self.with_random_clip_crops:
            crops = []
            rbboxs = []

            for i in range(15):
                p = self.cropfn.get_params(dataset_dict['image'], [self.crop_size, self.crop_size])
                c = tF.crop(dataset_dict['image'], *p)
                if self.crop_size != 224:
                    c = tF.resize(img=c, size=224)
                crops.append(c)
                # rbboxs.append(p)

            crops = torch.stack(crops)
            # Boxes(torch.cat(gt_boxes, 0))
            dataset_dict['randomcrops'] = crops
            # dataset_dict['randomcropsbox'] = rbboxs
            # apply same crop bbox to the jittered image
            if self.with_jitter:
                jitter_crops = []
                for p in rbboxs:
                    jc = tF.crop(dataset_dict['jitter_image'], *p)
                    if self.crop_size != 224:
                        jc = tF.resize(img=jc, size=224)
                    jitter_crops.append(jc)

                jcrops = torch.stack(jitter_crops)
                dataset_dict['jitter_randomcrops'] = jcrops

        return dataset_dict




class DomainBalancedTrainingSampler(Sampler):
    """Infinite distributed sampler with controllable source-domain balance.

    An image from domain d receives weight ``count(d) ** -power``. Power 0 is
    the original image-frequency distribution, while power 1 makes domains
    uniform. NAUTILUS uses 0.5 to avoid overfitting its very small 1A2C group.
    """

    def __init__(self, dataset_dicts, power=0.5, seed=None):
        if not dataset_dicts:
            raise ValueError("DomainBalancedTrainingSampler received no records")
        self._size = len(dataset_dicts)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._seed = comm.shared_random_seed() if seed is None else int(seed)

        domain_counts = Counter(
            str(record.get("domain_text", "__unknown__"))
            for record in dataset_dicts
        )
        self.domain_counts = dict(domain_counts)
        weights = [
            float(domain_counts[str(record.get("domain_text", "__unknown__"))])
            ** (-float(power))
            for record in dataset_dicts
        ]
        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._weights /= self._weights.sum()

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        global_position = 0
        while True:
            sampled = torch.multinomial(
                self._weights,
                self._size,
                replacement=True,
                generator=generator,
            ).tolist()
            for index in sampled:
                if (
                    global_position >= self._rank
                    and (global_position - self._rank) % self._world_size == 0
                ):
                    yield index
                global_position += 1


class Trainer(DefaultTrainer):

    def __init__(self, cfg) -> None:
        # DefaultTrainer hard-codes DDP with find_unused_parameters=False.
        # RDM deliberately uses different parameter subsets in its base,
        # offset, and full-training phases, so that default would fail on the
        # iteration after a phase-specific forward. Build the same trainer as
        # Detectron2's DefaultTrainer, but configure DDP for this dynamic graph.
        TrainerBase.__init__(self)
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        use_phase_aware_ddp = comm.get_world_size() > 1
        model = create_ddp_model(
            model,
            broadcast_buffers=False,
            find_unused_parameters=use_phase_aware_ddp,
        )
        self._trainer = (
            AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer
        )(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

        logger.info(
            "Phase-aware DDP: world_size=%d, find_unused_parameters=%s",
            comm.get_world_size(),
            use_phase_aware_ddp,
        )
        self.off_opt_interval = cfg.OFFSET_OPT_INTERVAL[0]
        self.off_opt_iters = cfg.OFFSET_OPT_ITERS
        self._consecutive_nonfinite_grad_steps = 0

    @staticmethod
    def _freeze_clip_batch_norm(model):
        """Keep pretrained CLIP BN statistics fixed for small detection batches."""
        raw_model = model.module if hasattr(model, "module") else model
        backbone = getattr(raw_model, "backbone", None)
        visual_encoder = getattr(backbone, "enc", None)
        if visual_encoder is None:
            return 0
        frozen_count = 0
        for module in visual_encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad = False
                frozen_count += 1
        return frozen_count

    def _clip_and_validate_gradients(self, optimizer):
        """Globally clip the active optimizer and identify non-finite grads."""
        active_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        named_gradients = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if id(parameter) in active_parameter_ids and parameter.grad is not None
        ]
        if not named_gradients:
            return 0.0, []

        with torch.no_grad():
            gradient_norms = torch.stack([
                parameter.grad.detach().float().norm(2)
                for _, parameter in named_gradients
            ])
            finite_mask = torch.isfinite(gradient_norms)
            if not bool(finite_mask.all()):
                bad_indices = (~finite_mask).nonzero(as_tuple=False).flatten().tolist()
                return float("nan"), [
                    named_gradients[index][0] for index in bad_indices[:20]
                ]

            total_norm = gradient_norms.norm(2)
            max_norm = float(self.cfg.SOLVER.GLOBAL_GRAD_CLIP_NORM)
            if max_norm > 0.0:
                clip_coefficient = torch.clamp(
                    max_norm / (total_norm + 1e-6), max=1.0
                )
                for _, parameter in named_gradients:
                    parameter.grad.mul_(clip_coefficient.to(parameter.grad.device))
            return float(total_norm.item()), []

    @staticmethod
    def _batch_debug_identifiers(batched_inputs):
        identifiers = []
        for sample in batched_inputs:
            identifiers.append({
                "image_id": sample.get("image_id"),
                "domain_group": sample.get("domain_group"),
                "file_name": sample.get("file_name"),
            })
        return identifiers

    def _forward_training_phase(self, batched_inputs, phase):
        """Run custom RDM phases without bypassing DDP's forward hooks."""
        if phase not in {"base", "offset", "full"}:
            raise ValueError("Unknown RDM training phase: {}".format(phase))
        if phase == "full":
            return self.model(batched_inputs)
        if hasattr(self.model, "module"):
            # DDP does not expose arbitrary attributes from the wrapped model.
            # Calling module.forward_base() directly would bypass DDP reducer
            # preparation and silently prevent correct gradient synchronization.
            return self.model(batched_inputs, training_phase=phase)
        phase_method = "forward_base" if phase == "base" else "opt_offsets"
        return getattr(self.model, phase_method)(batched_inputs)


    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        allow_tf32 = bool(cfg.SOLVER.ALLOW_TF32) and torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        model = build_model(cfg)
        frozen_bn_count = cls._freeze_clip_batch_norm(model)

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        logger.info("CUDA TF32 enabled: %s", allow_tf32)
        logger.info("Frozen CLIP BatchNorm modules: %d", frozen_bn_count)

        return model

    @classmethod
    def build_train_loader(cls, cfg):
        original = cfg.DATASETS.TRAIN
        print(original)
        cfg.DATASETS.TRAIN = (original[0],)
        balance_power = float(cfg.DATALOADER.DOMAIN_BALANCE_POWER)
        if balance_power > 0.0:
            dataset = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=(
                    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                    if cfg.MODEL.KEYPOINT_ON
                    else 0
                ),
                proposal_files=(
                    cfg.DATASETS.PROPOSAL_FILES_TRAIN
                    if cfg.MODEL.LOAD_PROPOSALS
                    else None
                ),
            )
            sampler = DomainBalancedTrainingSampler(dataset, balance_power)
            logger.info(
                "Source-domain balanced sampling enabled (power=%.3f): %s",
                balance_power,
                sampler.domain_counts,
            )
            data_loader1 = build_detection_train_loader(
                cfg,
                mapper=CustomDatasetMapper(cfg, True),
                dataset=dataset,
                sampler=sampler,
            )
        else:
            data_loader1 = build_detection_train_loader(
                cfg, mapper=CustomDatasetMapper(cfg, True)
            )
        return data_loader1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == 'coco':
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        raise NotImplementedError(
            "No evaluator implemented for dataset '{}' with type '{}'".format(
                dataset_name, evaluator_type
            )
        )

    @classmethod
    def build_optimizer(cls, cfg, model):

        trainable = {'others': [], 'offset': []}
        offset_list = ['offsets', 'weights', 'roi_heads.prompt_learner.ctx', 'roi_heads.prompt_learner.bg_ctx',
                       'roi_heads.prompt_learner.img_ctx']
        rpn_list = ['proposal_generator.rpn_head.conv.weight',
                    'proposal_generator.rpn_head.conv.bias',
                    'proposal_generator.rpn_head.objectness_logits.weight',
                    'proposal_generator.rpn_head.objectness_logits.bias',
                    'proposal_generator.rpn_head.anchor_deltas.weight',
                    'proposal_generator.rpn_head.anchor_deltas.bias']
        for name, val in model.named_parameters():
            head = name.split('.')[0]
            # previously was setting all params to be true
            if val.requires_grad == True:
                print(name)
                if name in offset_list:
                    trainable['offset'].append(val)
                # elif name == 'roi_heads.prompt_learner.ada_ctx':
                    # trainable['others'].append(val)
                    # trainable['offset'].append(val)
                # elif 'projector' in name:
                #     trainable['others'].append(val)
                # elif 'roi_heads.prompt_learner.img_ctx' in name:
                #     trainable['offset'].append(val)
                # elif name in rpn_list:
                #     trainable['offset'].append(val)
                    # trainable['others'].append(val)
                else:
                    trainable['others'].append(val)

        optimizer1 = torch.optim.SGD(
            trainable['others'],
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

        optimizer2 = torch.optim.Adam(
            trainable['offset'],
            0.001,
        )

        optimizer3 = torch.optim.Adam(
            trainable['offset'],
            0.0001,
        )

        return (maybe_add_gradient_clipping(cfg, optimizer1), maybe_add_gradient_clipping(cfg, optimizer2), maybe_add_gradient_clipping(cfg, optimizer3))

    def run_step(self):
        """
        Implement the standard training logic described above.
        """

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        # model.train() is recursive and may switch frozen CLIP BN modules back
        # to training mode after evaluation hooks. Reinforce the invariant on
        # every step without disabling gradients in trainable CLIP layers.
        self._freeze_clip_batch_norm(self.model)
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        data_s = data

        opt_phase = False
        opt_phase_finer = False
        # if len(self.off_opt_interval) and self.iter >= self.off_opt_interval[0] and self.iter < self.off_opt_interval[0]+self.off_opt_iters:
        # Treat OFFSET_OPT_ITERS as an exclusive end so epoch boundaries in
        # train_nautilus.py correspond exactly to [start_epoch, end_epoch).
        if self.iter < self.off_opt_iters:
            # if self.iter == self.off_opt_interval[0]:
            #     self.model.offsets.data = torch.zeros(self.model.offsets.shape).cuda()
            if self.iter >= self.off_opt_interval:  # 1300
                offset_duration = max(1, self.off_opt_iters - self.off_opt_interval)
                fine_window = max(1, int(0.2 * offset_duration))
                fine_start = self.off_opt_iters - fine_window
                if self.iter < fine_start:
                    opt_phase = True
                    loss_dict_s = self._forward_training_phase(data_s, "offset")
                else:
                    opt_phase = True
                    opt_phase_finer = True
                    loss_dict_s = self._forward_training_phase(data_s, "offset")
            else:
                opt_phase = False
                loss_dict_s = self._forward_training_phase(data_s, "base")
            # if self.iter+1 == self.off_opt_interval[0]+self.off_opt_iters:
            #     self.off_opt_interval.pop(0)
        else:
            # for ind, d in enumerate(data_s):
            #     d['image'] = self.aug(d['image'].cuda())
            loss_dict_s = self._forward_training_phase(data_s, "full")
            # print(loss_dict_s)

        # import pdb;pdb.set_trace()
        loss_dict = {}

        loss = 0
        for k, v in loss_dict_s.items():
            loss += v

        if not bool(torch.isfinite(loss.detach()).all()):
            loss_values = {
                key: float(value.detach().float().cpu())
                for key, value in loss_dict_s.items()
            }
            raise FloatingPointError(
                "Non-finite training loss at iter {}: losses={}, batch={}".format(
                    self.iter,
                    loss_values,
                    self._batch_debug_identifiers(data_s),
                )
            )

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()
        self.optimizer[2].zero_grad()

        loss.backward()

        if not opt_phase:
            active_optimizer = self.optimizer[0]
        elif not opt_phase_finer:
            active_optimizer = self.optimizer[1]
        else:
            active_optimizer = self.optimizer[2]

        gradient_norm, bad_gradient_names = self._clip_and_validate_gradients(
            active_optimizer
        )
        if bad_gradient_names:
            self._consecutive_nonfinite_grad_steps += 1
            logger.error(
                "Skipping optimizer step at iter %d due to non-finite gradients "
                "in %s; batch=%s",
                self.iter,
                bad_gradient_names,
                self._batch_debug_identifiers(data_s),
            )
            self.optimizer[0].zero_grad()
            self.optimizer[1].zero_grad()
            self.optimizer[2].zero_grad()
            loss_dict_s["skipped_nonfinite_gradient"] = loss.detach() * 0.0 + 1.0
            self._trainer._write_metrics(loss_dict_s, data_time)
            max_bad_steps = int(
                self.cfg.SOLVER.MAX_CONSECUTIVE_NONFINITE_GRADS
            )
            if self._consecutive_nonfinite_grad_steps >= max_bad_steps:
                raise FloatingPointError(
                    "Non-finite gradients repeated for {} consecutive steps; "
                    "last parameters={}".format(
                        self._consecutive_nonfinite_grad_steps,
                        bad_gradient_names,
                    )
                )
            return

        self._consecutive_nonfinite_grad_steps = 0
        self.storage.put_scalar(
            "global_grad_norm", gradient_norm, smoothing_hint=False
        )

        if not opt_phase:
            self.optimizer[0].step()
        else:
            if not opt_phase_finer:
                self.optimizer[1].step()
            else:
                self.optimizer[2].step()

        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()
        self.optimizer[2].zero_grad()

        for k, v in loss_dict_s.items():
            loss_dict.update({k: v})

        # print(loss_di ct)
        self._trainer._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EpochLogger(
                cfg.TRAIN_ITERS_PER_EPOCH,
                cfg.TRAIN_TOTAL_EPOCHS,
                cfg.TRAIN_EPOCH_LOG_PERIOD,
                cfg.TEST.EVAL_PERIOD,
            ),
            LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def do_test_st(flag):
            if flag == 'st':
                model = self.model
            else:
                print("Error in the flag")

            results = OrderedDict()
            for dataset_name in self.cfg.DATASETS.TEST:
                data_loader = build_detection_test_loader(self.cfg, dataset_name)
                evaluator = self.build_evaluator(self.cfg, dataset_name)
                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)
                    storage = get_event_storage()
                    storage.put_scalar(f'{dataset_name}_AP50', results_i['bbox']['AP50'], smoothing_hint=False)
            if len(results) == 1:
                results = list(results.values())[0]
            return results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        if cfg.TEST.RUN_SECOND_EVAL:
            ret.append(
                hooks.EvalHook(
                    cfg.TEST.EVAL_SAVE_PERIOD,
                    lambda flag='st': do_test_st(flag),
                )
            )

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        return build_lr_scheduler(cfg, optimizer[0])

    def state_dict(self):
        # DefaultTrainer.state_dict() delegates to SimpleTrainer.state_dict(),
        # which assumes a single optimizer and calls self.optimizer.state_dict().
        # This trainer intentionally stores three optimizers in a tuple, so save
        # the TrainerBase state and every optimizer explicitly instead.
        ret = TrainerBase.state_dict(self)
        ret["optimizer1"] = self.optimizer[0].state_dict()
        ret["optimizer2"] = self.optimizer[1].state_dict()
        ret["optimizer3"] = self.optimizer[2].state_dict()
        if isinstance(self._trainer, AMPTrainer):
            ret["grad_scaler"] = self._trainer.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        # Bypass DefaultTrainer.load_state_dict() for the same multi-optimizer
        # reason as above.
        TrainerBase.load_state_dict(self, state_dict)
        for optimizer, key in zip(
            self.optimizer, ("optimizer1", "optimizer2", "optimizer3")
        ):
            if key in state_dict:
                optimizer.load_state_dict(state_dict[key])
        if isinstance(self._trainer, AMPTrainer) and "grad_scaler" in state_dict:
            self._trainer.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.
        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id1 = LRScheduler.get_best_param_group_id(self._optimizer[0])
        self._best_param_group_id2 = LRScheduler.get_best_param_group_id(self._optimizer[1])

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr1 = self._optimizer[0].param_groups[self._best_param_group_id1]["lr"]
        self.trainer.storage.put_scalar("lr", lr1, smoothing_hint=False)
        self.trainer.storage.put_scalar("lr1", lr1, smoothing_hint=False)

        lr2 = self._optimizer[1].param_groups[self._best_param_group_id2]["lr"]
        self.trainer.storage.put_scalar("lr2", lr2, smoothing_hint=False)

        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)




def do_test(cfg, model, model_type=''):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = Trainer.build_evaluator(cfg, dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
    return results


def main(args):
    register_dataset(None)
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    for dataset_name in cfg.DATASETS.TEST:
        if 'daytime_clear_test' in dataset_name:
            trainer.register_hooks([
                hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD, trainer.checkpointer, f'{dataset_name}_AP50',
                                       file_prefix='model_best'),
            ])

    trainer.train()


