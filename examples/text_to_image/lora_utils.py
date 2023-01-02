import torch

from lora_diffusion.lora import _find_modules_v2
from lora_diffusion import (
        extract_lora_ups_down,
        inject_trainable_lora,
        save_lora_weight,
        save_safeloras,
        LoraInjectedLinear,
    )


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}
DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE


def extract_loras(model, target_replace_module=DEFAULT_TARGET_REPLACE):
    # get LORA modules of unet
    loras = []
    for _m, _n, _child_module in _find_modules_v2(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        #loras.append((_child_module.lora_up, _child_module.lora_down))
        loras.append(_child_module)
    if len(loras) == 0:
        raise ValueError("No lora injected.")
    return loras


def create_expert_weights(loras, num_experts=5):
    # generate N sets of lora_up and lora_down, one for each expert
    expert_weights = []
    for i in range(num_experts):
        weights = torch.nn.ModuleList([torch.nn.ParameterList([torch.nn.Parameter(lora.lora_down.weight.clone()),
                                                               torch.nn.Parameter(lora.lora_up.weight.clone())]) 
                                       for lora in loras])
        expert_weights.append(weights)
    expert_weights = torch.nn.ModuleList(expert_weights)
    return expert_weights


def use_expert(loras, expert_weights, expert_idx):
    # replace all experts lora weights for one expert
    for lora_idx, lora in enumerate(loras):
        for weight_idx in range(2):
            if weight_idx == 0:
                layer = loras[lora_idx].lora_down
            else:
                layer = loras[lora_idx].lora_up
            expert_layer_weights = expert_weights[expert_idx][lora_idx][weight_idx]
            expert_layer_weights_copy = expert_layer_weights.clone()
            layer.weight = expert_layer_weights

            
def get_expert_index(timestep, num_experts, total_timesteps=1000):
    # assumes that
    timestep = timestep.flatten()[0]  # select the first timestep
    expert_size = total_timesteps // num_experts
    return timestep // expert_size


class UNet2DConditionOutput:
    sample: torch.FloatTensor
    
 # if we use Mixture of Denoising Experts (MoDE), then we have a set of weights per expert
    # we need to:
# 1. create a Unet variant that activates the correct LORA weights given the timestep
# 2. Initialize N LORA weights, one set per expert
# 3. Sample one timestep per batch and activate the expert for that timestep
    
class UNetLORAMoDE(torch.nn.Module):
    def __init__(self, unet, num_experts=5):
        super().__init__()
        self.unet = unet
        self.num_experts = num_experts
        # for compatibility
        self.in_channels = unet.in_channels
        self.config = unet.config
        self.device = unet.device
        self.dtype = torch.half

        # init LoRA
        # add LORA modules to unet
        unet.requires_grad_(False)
        unet_lora_params, _ = inject_trainable_lora(
            unet, r=4#, loras=args.resume_unet
        )
        # init experts
        self.lora_modules = extract_loras(unet)
        self.expert_weights = create_expert_weights(self.lora_modules, self.num_experts)
        
        self.fake_tensor = torch.rand(5)

    def forward(self, latent_model_input, t, encoder_hidden_states):
        # assign expert
        expert_idx = get_expert_index(t, self.num_experts)
        use_expert(self.lora_modules, self.expert_weights, expert_idx)
        # make prediction
        sample = self.unet(latent_model_input, t, encoder_hidden_states)[0]
        out = UNet2DConditionOutput()
        out.sample = sample
        return out

    def to(self, *args, **kwargs):
        # to get device and dtype right
        self.fake_tensor.to(*args, **kwargs)
        self.device = self.fake_tensor.device
        self.dtype = self.fake_tensor.dtype
        # send to
        self.unet.to(*args, **kwargs)
        self.expert_weights.to(*args, **kwargs)
        return self