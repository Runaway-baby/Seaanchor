# Seaanchor

以 `train.py` 为唯一训练入口的 NAUTILUS 水下目标检测工程。训练链路为 DINOv3 → RDM 域调制 → RPN → OCIA ROI 增强 → CLIP 文本分类。保留原入口的超参数、域划分、三阶段优化、域均衡采样、评估和恢复训练逻辑。

## 文件结构

```text
Seaanchor/
├── train.py                   # 唯一入口：准备数据、启动训练、评估
├── configs/
│   ├── nautilus.yaml           # 检测器及 OCIA / RDM 配置
│   └── domain_text.txt         # 域描述
├── seaanchor/
│   ├── data/nautilus.py        # COCO 合并、40 类校验、数据注册
│   ├── engine/trainer.py       # 三阶段优化、采样、数据映射、日志钩子
│   ├── modeling/
│   │   ├── backbone.py         # DINOv3 主干与特征适配
│   │   ├── ocia.py             # OCIA 提示、傅里叶分支、ROI 适配
│   │   ├── rdm.py              # RDM 域调制与检测架构
│   │   ├── roi_head.py         # ROI 检测与一致性损失
│   │   ├── rpn.py              # 候选框生成
│   │   ├── box_predictor.py    # 框回归和分类输出
│   │   ├── classifier.py       # CLIP 类别分类器
│   │   ├── prompt_learning.py  # 可学习文本提示
│   │   ├── text_prompts.py     # 类名映射和文本模板
│   │   └── config.py           # 自定义配置项
│   └── clip/                  # 本地 CLIP 实现与 BPE 词表
├── tests/test_project.py      # 结构和数据流程验证
├── requirements.txt           # 直接依赖（非锁定环境）
└── outputs/                   # 运行时创建
```

所有项目自定义训练模块均包含在本目录，运行时无需引用旧项目。没有复制数据集、训练日志、缓存、模型权重、另一套检测框架或无关实验入口。保留模块内已有版权声明。

## 环境

训练需要 CUDA、PyTorch、匹配的 torchvision、Detectron2、官方 DINOv3 本地源码及相应主干权重。当前代码使用 `torch.amp.autocast("cuda", ...)`；旧版 PyTorch 1.10 环境不能直接沿用。应在能够运行所选 DINOv3 模型的环境内安装、编译匹配的 Detectron2，再执行：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 只列直接依赖，不声称是验证过的版本锁定。Detectron2 自带的 `COCO-Detection/faster_rcnn_R_101_C4_3x.yaml` 仍作为基础配置。CLIP 实现和词表已内置；RN101 预训练权重仍需位于 CLIP 缓存中，缺失时原加载器会下载。无需额外安装同名 CLIP 包。

## 数据和训练

`--data-root` 目录下应包含：

```text
NAUTILUS-dataset_classified_annotations/<域名>_annotations.json
NAUTILUS-dataset_classified2/<域名>/<图片文件名>
```

训练域为 `1A2A / 1A2C / 1B2A / 1B2C`；测试域为 `1C2A / 1C2C`。40 类 ID 与名称按数据模块中的定义严格校验。合并标注写入数据根目录的 `annotations_ocia/`，因此数据目录需要写权限。只使用 JSON 中声明的图像，修正实际图像尺寸，剔除缺失图像和无效框；`--strict-images` 对缺失或无法读取的图像直接报错。

先检查数据（仅需 Pillow，无需 PyTorch / Detectron2）：

```bash
python train.py --data-root /path/to/newdata --prepare-only --strict-images
```

启动训练（将示例路径替换为真实位置）：

```bash
python train.py --data-root /path/to/newdata --dinov3-repo /path/to/dinov3 --dinov3-weights /path/to/dinov3_vitb16.pth --epochs 100 --ims-per-batch 4 --num-gpus 1
```

默认模型为 `dinov3_vitb16`，输出定位到本项目 `outputs/nautilus`。可用 `--output-dir` 指定输出目录；`DINOV3_REPO`、`DINOV3_WEIGHTS` 环境变量也可提供主干路径。

恢复训练时使用相同参数与输出目录并增加 `--resume`。仅评估时增加 `--eval-only --weights /path/to/model_final.pth`。`--weights` 是完整检测器权重，区别于 `--dinov3-weights`。命令末尾支持配置覆盖，例如 `-- MODEL.OCIA.ENABLED False MODEL.RDM.MULTI_SOURCE False DATALOADER.NUM_WORKERS 4`。

参数名、配置名、模型属性和注释统一采用 OCIA / RDM。属性重命名会改变部分 checkpoint 键，因此本工程的 `--resume` 适用于本工程产生的 checkpoint；旧工程检测器 checkpoint 需先做键迁移，不能认为可以无损直接恢复。DINOv3 主干权重格式未改变。

## 验证范围

```bash
python -B -m unittest discover -s tests -v
```

测试覆盖 Python 语法、本地导入闭包、CLI、六域 COCO 合并、类别校验、缺图处理及从任意工作目录执行数据准备。整理环境未安装 PyTorch / Detectron2，未进行真实模型构建、GPU 前向反向或完整训练；这些仍需在具备数据、权重和训练依赖的环境验证。
