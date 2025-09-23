# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Generate paper-ready plots"""

import fractions
import inspect
import logging
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("Dark2")
SEQ_PALETTE = sns.color_palette("flare", as_cmap=True)

DISPLAY_NAMES = {
    "bits": "$b$",
    "block_size": "$B$",
    "qrmse_norm": "$R$",
    "LM": "Lloyd-Max",
    "rms": "RMS",
    "absmax": "Absmax",
    "signmax": "Signmax",
    "df": r"$\nu$",
    # Formats
    "BFLOAT16": r"\texttt{bfloat16}",
    "EXP8": r"\texttt{E8M0}",
    "E0M3": r"\texttt{INT4}",
    "E2M1": r"\texttt{E2M1}",
    "E3M0": r"\texttt{E3M0}",
    "NF4": r"\texttt{NF4}",
    # Experiments
    "bits_per_param": r"$b$",
    "kl_div": r"$\mathrm{D_{KL}}$",
    "kl_div_efficiency": r"$\rho$",
    # "kl_div_efficiency": r"$\mathrm{D_{KL}}\cdot 2^{2b}$",
    # "kl_div_efficiency_delta": r"Ratio of $\mathrm{mean}(\mathrm{D}\cdot 2^{2b})$",
    "allocation": "Bit allocation",
}
CRD_LABEL = r"$\sqrt[3]{\mathrm{p}}$"

# fmt: off
MODEL_ORDER_GRID = ["Llama-3.1-8B", "gemma-3-12b-pt", "phi-4",
                    "Llama-3.2-3B", "gemma-3-4b-pt", "Qwen2.5-7B",
                    "Llama-3.2-1B", "gemma-3-1b-pt", "Qwen2.5-3B"]
# fmt:on


def format_fraction(max_denominator: int = 10) -> Callable[[float, int], str]:
    def _format(x: float, n: int) -> str:
        if x == 0:
            return "0"
        f = fractions.Fraction.from_float(x).limit_denominator(max_denominator)
        if f.denominator == 1:
            return str(f.numerator)
        return f"$\\frac{{{f.numerator}}}{{{f.denominator}}}$"

    return _format


def drop_label(args: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in args.items() if k != "label"}


def transform_labels(
    items: list[tuple[Any, ...]], pattern: str, replacement: str
) -> list[tuple[Any, ...]]:
    return [
        (
            *p,
            {
                k: v.replace(pattern, replacement) if k == "label" else v
                for k, v in d.items()
            },
        )
        for *p, d in items
    ]


def display_name(t: str) -> str:
    if t in DISPLAY_NAMES:
        return DISPLAY_NAMES[t]
    if m := re.match(r"CRD-(N|L|T\[([\d.]+)\])-[RA]S", t):
        return (
            CRD_LABEL
            + dict(
                N=lambda: " Normal",
                L=lambda: " Laplace",
                T=lambda: f" t[$\\nu={m.group(2)}$]",
            )[m.group(1)[0]]()
        )
    return t


def configure(disable_tex_for_debug_speed: bool = False) -> None:
    """Place at the start of the notebook, to set up defaults."""
    print(
        "Recommend (Ubuntu):\n"
        "  sudo apt-get install cm-super dvipng fonts-cmu texlive-latex-extra"
    )
    logging.getLogger("matplotlib.texmanager").setLevel(logging.WARNING)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    sns.set_palette(PALETTE)
    font_name = "CMU Serif"
    matplotlib.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.serif": [font_name],
            "text.usetex": not disable_tex_for_debug_speed,
            # Latex
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{amsfonts}",
                    r"\usepackage{bm}",
                    r"\newcommand{\prob}{\mathrm{p}}",
                    r"\newcommand{\norm}[2]{\left \lVert #1 \right \rVert_{#2}}",
                    r"\newcommand{\expectation}[2]{\mathop{{}\mathbb{E}}_{#1}\left[#2\right]}",
                    r"\newcommand{\kl}{\mathrm{D_{KL}}}",
                    r"\newcommand{\kld}[2]{\kl\left(#1\|#2\right)}",
                ]
            ),
            # General
            "figure.figsize": (8, 3),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.edgecolor": "none",
            "legend.fontsize": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.markersize": 3,
        }
    )
    try:
        matplotlib.font_manager.findfont(
            font_name, rebuild_if_missing=True, fallback_to_default=False
        )
    except ValueError as e:
        print(
            f"Couldn't find font {font_name!r}.\nOn Ubuntu:\n"
            "  sudo apt install fonts-cmu\n"
            "  rm ~/.cache/matplotlib/fontlist-*.json\n"
            "  (restart kernel)\n"
            f"  (original error: {e!r})"
        )


def build_legend_handles(*rows: tuple[Any, ...] | dict[str, Any] | str) -> list[Any]:
    handles = []
    for row in rows:
        if isinstance(row, str):
            handles.append(matplotlib.patches.Patch(color="none", label=row))
        else:

            args = dict(row if isinstance(row, dict) else row[-1])
            args.setdefault("color", "k")
            handles.append(matplotlib.lines.Line2D([], [], **args))
    return handles


def set_figure_legend(
    figure: matplotlib.figure.Figure,
    handles: Any = None,
    labels: Any = None,
    build: list[tuple[Any, ...] | dict[str, Any] | str] = None,
    loc: str = "center left",
    bbox_to_anchor: tuple[float, float] = (0.98, 0.55),
    **args: Any,
) -> None:
    if build is not None:
        assert handles is None and labels is None
        handles = build_legend_handles(*build)
    figure.legend(
        handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, **args
    )
    if "left" in loc:
        extent = figure.legends[0].get_window_extent()
        figure.set_figwidth(2 * figure.get_figwidth() - extent.x1 / figure.dpi)


def share_legend(figure: matplotlib.figure.Figure, **args: Any) -> None:
    handles, labels = figure.axes[00].get_legend_handles_labels()
    for ax in figure.axes:
        assert ax.get_legend_handles_labels()[1] == labels
        if ax.legend_ is not None:
            ax.legend_.remove()
    set_figure_legend(figure, handles, labels, **args)


def tidy(figure: matplotlib.figure.Figure) -> None:
    figure.tight_layout()

    for ax in figure.axes:
        for label in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            label.set_text(display_name(label.get_text()))

    for legend in filter(None, [ax.legend_ for ax in figure.axes] + figure.legends):
        title = legend.get_title()
        title.set_text(display_name(title.get_text()))
        for text in legend.get_texts():
            text.set_text(display_name(text.get_text()))


# Subplot grids


@dataclass
class Grid:
    rows: list[str | None]
    cols: list[str | None]
    axes: np.ndarray[matplotlib.axes.Axes]  # [row, col]
    figure: matplotlib.figure.Figure

    def __iter__(self) -> Iterable[Any]:
        """Iterate through ((key,), Axes) tuples."""
        for row, axr in zip(self.rows, self.axes):
            for col, ax in zip(self.cols, axr):
                key = ()
                if row is not None:
                    key = (*key, row)
                if col is not None:
                    key = (*key, col)
                yield (key, ax)


def grid(
    rows: list[str | None] = [None],
    cols: list[str | None] = [None],
    sharex: bool = False,
    sharey: bool = False,
    height: float | None = None,
) -> Grid:
    """Create a grid of matplotlib plots (much like seaborn, but plainer if not simpler)."""
    figw, figh = matplotlib.rcParams["figure.figsize"]
    figure, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(cols),
        figsize=(figw, (height or figh) * len(rows)),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return Grid(rows=rows, cols=cols, axes=axes, figure=figure)


def fmt_latex_booktabs(df: pd.DataFrame, cols: dict[str, str]) -> str:
    """Format as a booktabs table."""

    def fmt_value(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3g}"
        else:
            return str(v)

    s = r"\begin{tabular}" + "{" + "l" * len(cols) + "}" + r" \toprule"
    s += "\n  " + " & ".join(map(fmt_value, cols.values())) + r" \\\midrule"
    for _, row in df.iterrows():
        s += "\n  " + " & ".join(fmt_value(row[col]) for col in cols) + r" \\"
    s += "\n" + r"\bottomrule"
    s += "\n" + r"\end{tabular}"
    return s


# Paper sync

OVERLEAF = Path(__file__).parent / "overleaf"
WARN_NO_OVERLEAF = False


def _check_overleaf_cloned() -> bool:
    if not OVERLEAF.exists():
        if WARN_NO_OVERLEAF:
            warnings.warn(
                f"Repository not found at {OVERLEAF}, disabling save-and-push"
            )
        return False
    return True


def push_to_paper() -> None:
    for git_cmd in [
        "add code/ fig/ tab/",
        "commit -m 'Update figures' --quiet",
        "pull --rebase --quiet",
        "push --quiet",
    ]:
        cmd = f"git -C {OVERLEAF} {git_cmd}"
        # print(f"$ {cmd}", file=sys.stderr)
        if subprocess.call(cmd, shell=True):
            print(f"Error running {cmd!r} -- aborting")
            return


def save(name: str, push: bool = True) -> None:
    """Save and push a figure to the paper."""
    if _check_overleaf_cloned():
        plt.savefig(OVERLEAF / "fig" / f"{name}.pdf", bbox_inches="tight")
        if push:
            push_to_paper()


def save_code(fn: Callable[..., Any], push: bool = True) -> None:
    body = inspect.getsource(fn).splitlines()[1:]
    body = [re.sub(r"^    ", "", x) for x in body]
    body = [x for x in body if "# IGNORE" not in x]
    code = "\n".join(body) + "\n"

    if _check_overleaf_cloned():
        (OVERLEAF / "code" / f"{fn.__name__}.py").write_text(code)
        if push:
            push_to_paper()


def save_table(
    name: str, df: pd.DataFrame, cols: dict[str, str], push: bool = True
) -> str:
    if _check_overleaf_cloned():
        (OVERLEAF / "tab" / f"{name}.tex").write_text(fmt_latex_booktabs(df, cols=cols))
        if push:
            push_to_paper()
