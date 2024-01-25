import copy
import logging
import re
from typing import Any, Callable, Literal, Optional, Union

import baukit
import torch

# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as MambaLMHeadModel
from transformers import AutoTokenizer

import src.utils.tokenizer_utils as tokenizer_utils
from mamba_minimal.model import Mamba
from src.data import Relation
from src.models import ModelandTokenizer
from src.utils.dataclasses import PredictedToken

logger = logging.getLogger(__name__)


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def predict_from_input(model, inp):
    if isinstance(model, Mamba):
        out = model(input_ids=inp["input_ids"])
    else:
        out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


import transformers


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[transformers.AutoTokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    # print(char_start, char_end)

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # Skip special tokens # ! Is this the proper way to do this?
        if token_char_start == token_char_end:
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def untuple(object: Any):
    if isinstance(object, tuple):
        return object[0]
    return object


######################### utils #########################
from src.models import unwrap_tokenizer


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | AutoTokenizer,
    logits: torch.Tensor,
    k: int = 10,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=k).values.squeeze().tolist()
    return [(tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)]


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    after_layer_norm: bool = False,
    interested_tokens: list[int] = [],
    get_proba: bool = False,
    k: int = 10,
) -> tuple[list[tuple[str, float]], dict]:
    lm_head = mt.lm_head if not after_layer_norm else mt.lm_head.lm_head
    h = untuple(h) if after_layer_norm else h
    logits = lm_head(h)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    candidates = interpret_logits(mt, logits, k=k)
    interested_logits = {
        t: (logits[t].item(), mt.tokenizer.decode(t)) for t in interested_tokens
    }
    return candidates, interested_logits


@torch.inference_mode()
def predict_next_token(
    mt: ModelandTokenizer,
    prompt: Union[str, list[str]],
    k: int = 5,
    batch_size: int = 8,
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
) -> Union[
    list[list[PredictedToken]],
    tuple[list[list[PredictedToken]], list[tuple[int, PredictedToken]]],
]:
    # ! Do not use batch decoding for LLaMA-2 models. Not working properly.
    # * Seems to be working properly for Mamba-S4. Consider batching for speed.
    # ? What the hack is happening with the subject `{Big Ben} is located in the city of`
    """Compute the next token."""
    if isinstance(prompt, str):
        prompt = [prompt]
        if token_of_interest is not None:
            token_of_interest = [token_of_interest]

    if token_of_interest is not None:
        assert len(token_of_interest) == len(prompt)
        track_interesting_tokens = []

    with tokenizer_utils.set_padding_side(mt.tokenizer, padding_side="left"):
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(
            mt.device
        )

    with torch.inference_mode():
        predictions = []
        for i in range(0, len(inputs.input_ids), batch_size):
            batch_inputs = {
                "input_ids": inputs.input_ids[i : i + batch_size],
            }
            if isinstance(mt.model, Mamba) == False:
                batch_inputs["attention_mask"] = inputs.attention_mask[
                    i : i + batch_size
                ]

            batch_outputs = mt.model(**batch_inputs)
            logits = (
                batch_outputs.logits[:, -1]
                if hasattr(batch_outputs, "logits")
                else batch_outputs[:, -1]
            )
            next_token_probs = logits.float().softmax(dim=-1)
            next_token_topk = next_token_probs.topk(dim=-1, k=k)

            for token_ids, token_probs in zip(
                next_token_topk.indices, next_token_topk.values
            ):
                predictions.append(
                    [
                        PredictedToken(
                            token=mt.tokenizer.decode(token_id),
                            # token_id=token_id.item(),
                            prob=prob.item(),
                        )
                        for token_id, prob in zip(token_ids, token_probs)
                    ]
                )
            if token_of_interest is not None:
                # print(batch_inputs["input_ids"].shape[0])
                for j in range(i, i + batch_inputs["input_ids"].shape[0]):
                    _t_idx = 0 if "llama" not in mt.model_name.lower() else 1
                    tok_id = (
                        mt.tokenizer(token_of_interest[j]).input_ids[_t_idx]
                        if type(token_of_interest[j]) == str
                        else token_of_interest[j]
                    )
                    # print(tok_id)
                    probs = next_token_probs[j - i]
                    order = probs.topk(dim=-1, k=probs.shape[-1]).indices.squeeze()
                    prob_tok = probs[tok_id]
                    rank = int(torch.where(order == tok_id)[0].item() + 1)
                    track_interesting_tokens.append(
                        (
                            rank,
                            PredictedToken(
                                token=mt.tokenizer.decode(tok_id),
                                # token_id=tok_id.item()
                                # if isinstance(tok_id, torch.Tensor)
                                # else tok_id,
                                prob=prob_tok.item(),
                            ),
                        )
                    )
    if token_of_interest is not None:
        return predictions, track_interesting_tokens
    return predictions


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


def make_icl_prompt(
    icl_examples: list, prompt_template: str, bos_token: str = "", subject: str = {}
):
    assert prompt_template.count("{}") == 1
    prompt = (
        bos_token
        + " "
        + "\n".join(
            [
                prompt_template.format(example[0]) + f" {example[1]}"
                for example in icl_examples
            ]
        )
    )
    prompt += "\n" + prompt_template.format(subject)
    return prompt


@torch.inference_mode()
def filter_samples_by_model_knowledge(
    mt: ModelandTokenizer, relation: Relation
) -> Relation:
    """Filter samples by model knowledge."""
    logger.debug(f'"{relation.name}" | filtering with {mt.model_name}')

    filtered_samples = []
    for i in range(len(relation.samples)):
        question, answer = relation[i]
        predictions = predict_next_token(mt, question, k=5)[0]
        top_pred = predictions[0]
        is_known = is_nontrivial_prefix(prediction=top_pred.token, target=answer)
        sample = relation.samples[i]
        if is_known:
            filtered_samples.append(sample)

        logger.debug(
            f"{sample.subject=} -> {answer=} | predicted = '{top_pred.token}'({top_pred.prob:.3f}) ==> ({get_tick_marker(is_known)})"
        )

    logger.info(
        f'filtered relation "{relation.name}" to {len(filtered_samples)} samples (with {len(relation._few_shot_samples)}-shots)'
    )

    relation.samples = filtered_samples
    return relation


def untuple(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def patch_repr(
    patch_layer: str,
    patch_idx: int,
    patch_vector: torch.Tensor,
    mode: Optional[Literal["ssm", "mlp"]] = None,  # will assume Mamba-S4 if provided
) -> Callable:
    def edit_repr(layer, repr):
        if layer != patch_layer:
            return repr
        if mode is None:
            untuple(repr)[
                :, patch_idx
            ] = patch_vector  # This should work for Transformers as well given proper module name `patch_layer`

        else:
            x_and_res = untuple(repr)[:, patch_idx]
            n_embd_x2 = x_and_res.shape[-1] // 2
            x, res = x_and_res.split([n_embd_x2, n_embd_x2], dim=-1)

            print(f"{x_and_res.shape=} {x.shape=} {res.shape=}")

            if mode == "ssm":
                x = (
                    patch_vector.reshape(x.shape)
                    .to(x_and_res.dtype)
                    .to(x_and_res.device)
                )
            elif mode == "mlp":
                res = (
                    patch_vector.reshape(res.shape)
                    .to(x_and_res.dtype)
                    .to(x_and_res.device)
                )
            else:
                raise AssertionError(f"{mode=} not supported")

            untuple(repr)[:, patch_idx] = torch.cat([x, res], dim=-1)

        return repr

    return edit_repr


@torch.inference_mode()
def get_h(
    mt: ModelandTokenizer,
    prompt: str,
    subject: str,
    layers: list[str],
    mode: Literal["input", "output"] = "output",
) -> dict[str, torch.Tensor]:
    # raise NotImplementedError("The function is not checked for Mamba-S4 yet")

    tokenized = mt.tokenizer(
        prompt, return_offsets_mapping=True, return_tensors="pt"
    ).to(mt.device)
    offset_mapping = tokenized.pop("offset_mapping")[0]
    if "token_type_ids" in tokenized:
        tokenized.pop("token_type_ids")

    subject_start, subject_end = find_token_range(
        prompt, subject, tokenizer=mt.tokenizer, offset_mapping=offset_mapping
    )

    subj_last_idx = subject_end - 1
    logger.debug(
        f"h_index={subj_last_idx} | h_token={mt.tokenizer.decode(tokenized['input_ids'][0][subj_last_idx])}"
    )

    if isinstance(mt.model, Mamba):
        tokenized.pop("attention_mask")

    retain_input = mode == "input"
    with baukit.TraceDict(
        module=mt.model, layers=layers, retain_input=retain_input
    ) as traces:
        mt.model(**tokenized)

    h = {
        layer: untuple(traces[layer].output)[:, subject_end - 1].squeeze()
        if mode == "output"
        else untuple(traces[layer].input)[:, subject_end - 1].squeeze()
        for layer in layers
    }
    return h
