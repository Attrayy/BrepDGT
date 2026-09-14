from brepdgt.models.detokenizers.base import Detokenizer, DetokenizerOutput
from brepdgt.models.detokenizers.pre_mask_uv_detokenizer_3d import (
    PreMaskUVDetokenizer3D,
    DedupDetokenizer3D,
)

DETOKENIZERS = {
    PreMaskUVDetokenizer3D.name: PreMaskUVDetokenizer3D,
    DedupDetokenizer3D.name: DedupDetokenizer3D,
}
