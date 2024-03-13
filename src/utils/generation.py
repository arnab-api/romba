import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

import baukit
import torch

import src.tokens as tokenization_utils
from src import functional, models
from src.data.dataclasses import PredictedToken, Relation
from src.utils.typing import Layer


@dataclass
class EditConfig:
    layers: list[Layer]
    intervention: Callable


# custom generate function for mamba
# ! batch generation will not give consistent results smaller members
# ! As PAD_embed != 0 and Mamba doesn't have a way to mask the attention to specific tokens.
# Will only give consistent results for the largest member.
@torch.inference_mode()
def mamba_generate(
    mt: models.ModelandTokenizer,
    prompts: Optional[list[str] | str] = None,
    input_ids: Optional[torch.Tensor] = None,
    max_out_len: int = 10,
    top_k: int = 5,
    edit_config: Optional[EditConfig] = None,
    n_gen_per_prompt: int = 1,
) -> list[str]:
    assert prompts is not None or input_ids is not None
    if isinstance(prompts, str):
        prompts = [prompts]

    prompts = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]

    if input_ids is None:
        with tokenization_utils.set_padding_side(mt.tokenizer, padding_side="left"):
            input_ids = (
                mt.tokenizer(prompts, return_tensors="pt", padding="longest")
                .to(models.determine_device(mt))
                .input_ids
            )

    predicted_tokens: list[PredictedToken] = []
    model_logits: torch.Tensor | None = None
    generated_tokens: list[list[int]] = [[] for _ in range(len(input_ids))]

    for i in range(max_out_len):
        if i == 0 and edit_config is not None:
            with baukit.Trace(
                module=mt.model,
                layer=edit_config.layers[0],
                edit_output=edit_config.intervention,
            ):
                outputs = mt(input_ids=input_ids)
        else:
            outputs = mt(input_ids=input_ids)

        logits = (
            outputs.logits[:, -1, :]
            if hasattr(outputs, "logits")
            else outputs[:, -1, :]
        )

        next_token_probs = logits.float().softmax(dim=-1)
        next_topk = logits.topk(dim=-1, k=top_k)
        next_token_probs_filtered_topk = next_topk.values.float().softmax(dim=-1)

        if i == 0:
            # save the logits and predicted tokens for the immidiate next token
            model_logits = logits[0].clone().cpu()
            for token_id in next_topk.indices[0]:
                predicted_tokens.append(
                    PredictedToken(
                        token=mt.tokenizer.decode(token_id),
                        prob=next_token_probs[0, token_id].item(),
                    )
                )

        # sample the next token
        next_token = torch.multinomial(next_token_probs_filtered_topk, num_samples=1)
        next_token = next_topk.indices.gather(dim=-1, index=next_token)

        for j in range(len(input_ids)):
            generated_tokens[j].append(next_token[j].item())

        # update the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_tokens = mt.tokenizer.batch_decode(generated_tokens)

    txt = [mt.tokenizer.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]
    functional.free_gpu_cache()
    return txt


@torch.inference_mode()
def generate_one_by_one(
    model, tok, prompts, n_gen_per_prompt=1, top_k=1, max_out_len=100
):
    txt = []
    for prompt in prompts:
        inp_tok = tok(prompt, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
        output_ids = model.generate(
            input_ids=input_ids.repeat(n_gen_per_prompt, 1),
            attention_mask=attention_mask.repeat(n_gen_per_prompt, 1),
            do_sample=True,
            top_k=top_k,
            max_length=max_out_len,
            pad_token_id=tok.eos_token_id,
        )
        txt.extend([tok.decode(out, skip_special_tokens=True) for out in output_ids])
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]
    functional.free_gpu_cache()
    return txt


def transformer_generate(
    mt: models.ModelandTokenizer,
    prompts: list[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """
    model, tok = mt.model, mt.tokenizer

    # ! Does not work for Mistral, weird attention error.
    # TODO(arnab): Debug
    if (
        "mistral" in model.config._name_or_path.lower()
        or "llama" in model.config._name_or_path.lower()
    ):
        return generate_one_by_one(
            model, tok, prompts, n_gen_per_prompt, top_k, max_out_len
        )

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # print(attention_mask)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]
    functional.free_gpu_cache()
    return txt


def generate_fast(mt: models.ModelandTokenizer, **kwargs) -> list[str]:
    functional.free_gpu_cache()
    if models.is_mamba_variant(mt.model):
        return mamba_generate(mt, **kwargs)
    else:
        return transformer_generate(mt, **kwargs)
