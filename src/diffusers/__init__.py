<<<<<<< HEAD
__version__ = "0.10.2"

from .configuration_utils import ConfigMixin
from .onnx_utils import OnnxRuntimeModel
=======
__version__ = "0.12.0.dev0"

from .configuration_utils import ConfigMixin
>>>>>>> upstream/main
from .utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_inflect_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    logging,
)


try:
<<<<<<< HEAD
=======
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_onnx_objects import *  # noqa F403
else:
    from .pipelines import OnnxRuntimeModel

try:
>>>>>>> upstream/main
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_pt_objects import *  # noqa F403
else:
<<<<<<< HEAD
    from .modeling_utils import ModelMixin
    from .models import AutoencoderKL, Transformer2DModel, UNet1DModel, UNet2DConditionModel, UNet2DModel, VQModel
=======
    from .models import (
        AutoencoderKL,
        ModelMixin,
        PriorTransformer,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        VQModel,
    )
>>>>>>> upstream/main
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
<<<<<<< HEAD
    from .pipeline_utils import DiffusionPipeline
    from .pipelines import (
        DanceDiffusionPipeline,
        DDIMPipeline,
        DDPMPipeline,
=======
    from .pipelines import (
        AudioPipelineOutput,
        DanceDiffusionPipeline,
        DDIMPipeline,
        DDPMPipeline,
        DiffusionPipeline,
        ImagePipelineOutput,
>>>>>>> upstream/main
        KarrasVePipeline,
        LDMPipeline,
        LDMSuperResolutionPipeline,
        PNDMPipeline,
        RePaintPipeline,
        ScoreSdeVePipeline,
    )
    from .schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
<<<<<<< HEAD
=======
        UnCLIPScheduler,
>>>>>>> upstream/main
        VQDiffusionScheduler,
    )
    from .training_utils import EMAModel

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
else:
    from .schedulers import LMSDiscreteScheduler


try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        CycleDiffusionPipeline,
        LDMTextToImagePipeline,
        PaintByExamplePipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionPipeline,
        StableDiffusionPipelineSafe,
        StableDiffusionUpscalePipeline,
<<<<<<< HEAD
=======
        UnCLIPImageVariationPipeline,
        UnCLIPPipeline,
>>>>>>> upstream/main
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
        VQDiffusionPipeline,
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    from .pipelines import StableDiffusionKDiffusionPipeline

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    from .pipelines import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        StableDiffusionOnnxPipeline,
    )
<<<<<<< HEAD

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    from .pipelines import AudioDiffusionPipeline, Mel

try:
=======

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    from .pipelines import AudioDiffusionPipeline, Mel

try:
>>>>>>> upstream/main
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_flax_objects import *  # noqa F403
else:
<<<<<<< HEAD
    from .modeling_flax_utils import FlaxModelMixin
=======
    from .models.modeling_flax_utils import FlaxModelMixin
>>>>>>> upstream/main
    from .models.unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .models.vae_flax import FlaxAutoencoderKL
    from .pipelines import FlaxDiffusionPipeline
    from .schedulers import (
        FlaxDDIMScheduler,
        FlaxDDPMScheduler,
        FlaxDPMSolverMultistepScheduler,
        FlaxKarrasVeScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxPNDMScheduler,
        FlaxSchedulerMixin,
        FlaxScoreSdeVeScheduler,
    )

<<<<<<< HEAD
from .callbacks import LPIPSCallback, NormalDistLoss, ContrastLoss, LPIPSLoss, create_callbacks
=======
>>>>>>> upstream/main

try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
<<<<<<< HEAD
    from .pipelines import FlaxStableDiffusionPipeline
=======
    from .pipelines import FlaxStableDiffusionImg2ImgPipeline, FlaxStableDiffusionPipeline
>>>>>>> upstream/main
