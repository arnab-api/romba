import torch

from src.models import ModelandTokenizer


def perplexity(
    mt: ModelandTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    inputs = mt.tokenizer(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")
    output = mt(**inputs)
    logits = output.logits if hasattr(output, "logits") else output
    logits = torch.nn.functional.log_softmax(logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()
