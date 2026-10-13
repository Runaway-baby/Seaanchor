"""Register the components used by the Seaanchor detector."""
from .rpn import SBRPN
from .backbone import DINOv3Backbone
from .rdm import ClipRCNNWithClipBackbone, ClipRCNNWithClipBackboneAugL1Dis
from .roi_head import ClipRes5ROIHeadsAttn
from .config import add_seaanchor_config
