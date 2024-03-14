import os

import matplotlib.pyplot as plt

from mamba_minimal.model import Mamba
from src.functional import guess_subject
from src.tracing import calculate_hidden_flow


def get_color_map(kind):
    if kind in [None, "None", ""]:
        return "Purples"
    if "mlp" in kind.lower():
        return "Greens"
    if "attn" in kind.lower():
        return "Reds"
    if "ssm" in kind.lower():
        return "Reds"
    return "Greys"


from src.functional import decode_tokens


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    alt_subject_tokens = (
        None if "alt_subject" not in result else list(result["alt_subject"])
    )
    subject_start = result["subject_range"][0]
    for i in range(*result["subject_range"]):
        if alt_subject_tokens is None:
            labels[i] = labels[i] + "*"
        else:
            labels[i] = f"{labels[i]}/{alt_subject_tokens[i - subject_start]}"

    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            "font.size": 10,
        }
    ):
        # fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        fig, ax = plt.subplots(figsize=(3.5, len(labels) * 0.08 + 1.8), dpi=200)
        h = ax.pcolor(
            differences,
            cmap=get_color_map(kind),
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = kind.upper()
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(
                f"p({str(answer).strip()})",
                # y=-len(labels) * 0.011,
                y=-0.13,
                fontsize=10,
            )
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


from typing import Literal


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    model_name=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        num_samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf, modelname=model_name)


def plot_all_flow(mt, prompt, subject=None, model_name=None):
    for kind in ["mlp", "ssm" if isinstance(mt.model, Mamba) else "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind, model_name=model_name)
