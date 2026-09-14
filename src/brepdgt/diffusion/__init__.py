from brepdgt.diffusion.base import Diffusion, DiffusionSample
from brepdgt.diffusion.gaussian_diffusion import GaussianDiffusion1D
from brepdgt.diffusion.separate_gaussian_diffusion import SeparateGaussianDiffusion1D

DIFFUSION_PROCESSES = {
    GaussianDiffusion1D.name: GaussianDiffusion1D,
    SeparateGaussianDiffusion1D.name: SeparateGaussianDiffusion1D,
}
