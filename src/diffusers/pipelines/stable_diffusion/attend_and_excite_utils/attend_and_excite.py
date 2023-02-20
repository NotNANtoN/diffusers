from typing import List

import torch

from .gaussian_smoothing import GaussianSmoothing
from .ptp_utils import AttentionStore, aggregate_attention


def compute_max_attention_per_index(attention_maps: torch.Tensor,
                                     indices_to_alter: List[int],
                                     smooth_attentions: bool = False,
                                     sigma: float = 0.5,
                                     kernel_size: int = 3) -> List[torch.Tensor]:
    """ Computes the maximum attention value for each of the tokens we wish to alter. """
    attention_for_text = attention_maps[:, :, 1:-1]
    attention_for_text *= 100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

    # Shift indices since we removed the first token
    indices_to_alter = [index - 1 for index in indices_to_alter]

    # Extract the maximum values
    max_indices_list = []
    for i in indices_to_alter:
        image = attention_for_text[:, :, i]
        if smooth_attentions:
            smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            image = smoothing(input).squeeze(0).squeeze(0)
        max_indices_list.append(image.max())
    return max_indices_list


def aggregate_and_get_max_attention_per_token(attention_store: AttentionStore,
                                               indices_to_alter: List[int],
                                               attention_res: int = 16,
                                               smooth_attentions: bool = False,
                                               sigma: float = 0.5,
                                               kernel_size: int = 3):
    """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0)
    max_attention_per_index = compute_max_attention_per_index(
        attention_maps=attention_maps,
        indices_to_alter=indices_to_alter,
        smooth_attentions=smooth_attentions,
        sigma=sigma,
        kernel_size=kernel_size)
    return max_attention_per_index



def compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
    """ Computes the attend-and-excite loss using the maximum attention value for each token. """
    losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
    loss = max(losses)
    if return_losses:
        return loss, losses
    else:
        return loss


def update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
    """ Update the latent according to the computed loss. """
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
    latents = latents - step_size * grad_cond
    return latents


def perform_iterative_refinement_step(unet, tokenizer, 
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           verbose=True,
                                     ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size)

            loss, losses = compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = pdate_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            if verbose and text_input is not None:
                low_word = tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
                print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size)
        loss, losses = compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index