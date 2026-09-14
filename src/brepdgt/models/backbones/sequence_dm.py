from brepdgt.models.backbones.dit_1d import Dit1D, Dit1DCrossAttn, TwoDit1D


# TODO: add diffusion abstract interface for clarity


SEQUENCE_DIFFUSION_BACKBONES = {
    # WARNING: unet always sees ordered sequences due to convolutions!
    # Unet1D.name: Unet1D,
    Dit1D.name: Dit1D,
    Dit1DCrossAttn.name: Dit1DCrossAttn,
    TwoDit1D.name: TwoDit1D,
}
