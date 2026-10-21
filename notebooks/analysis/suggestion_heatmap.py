"""Render one pool document's uncertainty scores as a suggestion heatmap, in HTML and LaTeX.

The heatmaps come from `scripts/suggestion_heatmap.py score`: every pool word's score
and predicted tag under one round's model snapshot, the words labelled so far, and the
next round's suggestions. Shading runs from white (certain) to blue at a score of 1.
Two consecutive rounds also draw the active learning loop: the model suggests and
guesses, the annotator labels, and the updated model suggests again.
"""

import html
import json
from collections.abc import Callable
from pathlib import Path

from seqeval.metrics.sequence_labeling import get_entities

from analysis.paper_results import tag_name, type_name

HEAT = (57, 135, 229)  # Sequential blue, step 400 (#3987E5).
GUESS = "#D6383A"
CORRECT = "#2E8540"
LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def heatmap_files(results_root: Path) -> dict[str, str]:
    """Map "run folder · round r" to each saved heatmap, in run and round order."""
    return {
        f"{path.parent.name} · round {int(path.stem.rsplit('_', 1)[1])}": str(path)
        for path in sorted((results_root / "suggestion_heatmap").glob("*/heatmap_round_*.json"))
    }


def loop_pairs(results_root: Path) -> dict[str, tuple[str, str]]:
    """Map "run folder · rounds r→r+1" to each pair of heatmaps from consecutive rounds."""
    files = heatmap_files(results_root)
    paths = {
        (Path(path).parent, int(Path(path).stem.rsplit("_", 1)[1])): path for path in files.values()
    }
    return {
        f"{folder.name} · rounds {round_idx}→{round_idx + 1}": (path, paths[folder, round_idx + 1])
        for (folder, round_idx), path in paths.items()
        if (folder, round_idx + 1) in paths
    }


def load_heatmap(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def document_sentences(heatmap: dict, document: str) -> list[dict]:
    return sorted(
        (s for s in heatmap["sentences"] if s["document_name"] == document),
        key=lambda s: s["sentence_id"],
    )


def document_names(heatmap: dict) -> list[str]:
    return sorted({s["document_name"] for s in heatmap["sentences"]})


def suggestion_counts(sentences: list[dict]) -> dict[str, int]:
    """Count a document's words, labelled words, and suggestions inside and outside entities."""
    suggested = [
        tag != "O"
        for s in sentences
        for tag, is_suggested in zip(s["tags"], s["suggested"], strict=True)
        if is_suggested
    ]
    return {
        "words": sum(len(s["tokens"]) for s in sentences),
        "labelled": sum(sum(s["labelled"]) for s in sentences),
        "suggested": len(suggested),
        "suggested_in_entities": sum(suggested),
    }


def _render(
    sentence: dict,
    word: Callable[[str, float | None, bool, bool], str],
    entity: Callable[[str, str], str],
) -> str:
    pieces = [
        word(token, score, labelled, suggested)
        for token, score, labelled, suggested in zip(
            sentence["tokens"],
            sentence["scores"],
            sentence["labelled"],
            sentence["suggested"],
            strict=True,
        )
    ]
    for kind, start, end in reversed(get_entities(sentence["tags"])):
        pieces[start : end + 1] = [entity(" ".join(pieces[start : end + 1]), kind)]
    return " ".join(pieces)


def heatmap_html(sentences: list[dict], uq_metric: str) -> str:
    """One paragraph per sentence, with a tooltip of each word's score and gold tag."""

    def word(token, score, labelled, suggested):
        text = html.escape(token)
        if score is None:
            return text
        style = f"background:rgba{(*HEAT, round(score, 3))};border-radius:3px;padding:1px 2px"
        if labelled:
            style = "opacity:.4;padding:1px 2px"
        elif suggested:
            style += ";outline:1.5px solid currentColor"
        return f'<span style="{style}" title="{uq_metric} {score:.3f}">{text}</span>'

    def entity(text, kind):
        return (
            f'<span style="border-bottom:2px solid currentColor;padding-bottom:1px" '
            f'title="{html.escape(kind)}">{text}</span>'
        )

    return "".join(
        f"<p style='line-height:2.1;margin:0 0 6px'>{_render(s, word, entity)}</p>"
        for s in sentences
    )


def legend_html() -> str:
    gradient = f"linear-gradient(90deg, rgba{(*HEAT, 0)}, rgba{(*HEAT, 1)})"
    return (
        "<div style='display:flex;gap:18px;align-items:center;flex-wrap:wrap;font-size:.85em'>"
        "<span>certain <span style='display:inline-block;width:120px;height:10px;"
        f"vertical-align:middle;border-radius:3px;background:{gradient}'></span> uncertain</span>"
        "<span><span style='outline:1.5px solid currentColor;padding:1px 4px'>word</span> "
        "suggested next round</span>"
        "<span><span style='opacity:.4'>word</span> already labelled</span>"
        "<span><span style='border-bottom:2px solid currentColor'>word</span> gold entity</span>"
        "</div>"
    )


def heatmap_latex(sentences: list[dict]) -> str:
    """A paragraph for a figure; needs xcolor, and shading in percent of the score."""

    def word(token, score, labelled, suggested):
        text = "".join(LATEX_SPECIALS.get(char, char) for char in token)
        if score is None:
            return text
        if labelled:
            return rf"\textcolor{{gray}}{{{text}}}"
        shade = f"uqheat!{round(100 * score)}"
        if suggested:
            return rf"\fcolorbox{{uqsuggest}}{{{shade}}}{{\strut {text}}}"
        return rf"\colorbox{{{shade}}}{{\strut {text}}}"

    def entity(text, kind):
        return rf"\underline{{{text}}}"

    body = " ".join(_render(s, word, entity) for s in sentences)
    return (
        "% Needs \\usepackage{xcolor}. Shading is the score in percent of uqheat.\n"
        "\\definecolor{uqheat}{HTML}{3987E5}\n"
        "\\definecolor{uqsuggest}{HTML}{0D366B}\n"
        "{\\setlength{\\fboxsep}{1pt}\\setlength{\\fboxrule}{0.8pt}%\n"
        f"{body}\n"
        "}\n"
    )


def loop_sentences(before: dict, after: dict) -> list[dict]:
    """Sentences with suggestions in both rounds, corrected guesses first, then shortest."""
    if after["run_dir"] != before["run_dir"] or after["round"] != before["round"] + 1:
        raise ValueError("the loop needs two consecutive rounds of one run")
    return sorted(
        (
            first
            for first, second in zip(before["sentences"], after["sentences"], strict=True)
            if any(first["suggested"]) and any(second["suggested"])
        ),
        key=lambda s: (not any(_corrected(s)), len(s["tokens"])),
    )


def _corrected(sentence: dict) -> list[bool]:
    """Which suggested words the model guessed wrong."""
    return [
        suggested and guess != tag
        for suggested, guess, tag in zip(
            sentence["suggested"], sentence["predicted"], sentence["tags"], strict=True
        )
    ]


def tag_label(tag: str, other: str) -> str:
    """A tag's type for display, keeping B-/I- only when that is all that tells it from other."""
    if tag == "O":
        return "O"
    if tag != other and tag[2:] == other[2:]:
        return tag_name(tag)
    return type_name(tag[2:])


def _stacked(label: str, word: str) -> str:
    """A word on the text line with a small label centred above it."""
    return (
        f"<span style='position:relative;display:inline-block'>{word}"
        "<span style='position:absolute;bottom:100%;left:50%;transform:translateX(-50%);"
        f"font-size:.72em;line-height:1.3;white-space:nowrap'>{label}</span></span>"
    )


def _loop_panel(sentence: dict, mode: str) -> str:
    words = []
    for token, score, tag, guess, labelled, suggested in zip(
        sentence["tokens"],
        sentence["scores"],
        sentence["tags"],
        sentence["predicted"],
        sentence["labelled"],
        sentence["suggested"],
        strict=True,
    ):
        text = html.escape(token)
        if labelled and mode == "model":
            words.append(f"<span style='opacity:.4'>{text}</span>")
            continue
        shade = (
            f"background:rgba{(*HEAT, round(score, 3))};"
            if mode == "model" and score is not None
            else ""
        )
        box = f"{shade}border-radius:3px;padding:0 2px"
        if not suggested:
            words.append(f'<span style="{box}">{text}</span>')
            continue
        box += ";outline:1.5px solid currentColor"
        word = f'<span style="{box}" title="{tag_name(tag)}">{text}</span>'
        if mode == "model":
            label = f"<span style='color:{GUESS}'>{html.escape(tag_label(guess, tag))}?</span>"
        elif guess == tag:
            label = f"<span style='color:{CORRECT}'>✓ {html.escape(tag_label(tag, guess))}</span>"
        else:
            label = (
                f"<span style='color:{GUESS}'>{html.escape(tag_label(guess, tag))}</span> → "
                f"<span style='color:{CORRECT}'>{html.escape(tag_label(tag, guess))}</span>"
            )
        words.append(_stacked(label, word))
    return " ".join(words)


MODEL_ICON = (
    "<svg width='44' height='40' viewBox='0 0 44 40' fill='none' stroke='currentColor' "
    "stroke-width='1.4'><path d='M6 8 22 4 38 10M6 8 22 20 38 10M6 8 22 36M6 22 22 4M6 22 "
    "22 20 38 30M6 22 22 36 38 30M6 34 22 20M6 34 22 36M22 4 38 30M22 36 38 10'/>"
    + "".join(
        f"<circle cx='{x}' cy='{y}' r='2.6' fill='currentColor'/>"
        for x, y in ((6, 8), (6, 22), (6, 34), (22, 4), (22, 20), (22, 36), (38, 10), (38, 30))
    )
    + "</svg>"
)
ANNOTATOR_ICON = (
    "<svg width='44' height='40' viewBox='0 0 44 40' fill='none' stroke='currentColor' "
    "stroke-width='1.8'><circle cx='22' cy='12' r='8'/><path d='M8 38c0-9 6-14 14-14s14 5 14 14'/>"
    "</svg>"
)


def loop_html(before: dict, after: dict, pool_idxs: list[int]) -> str:
    """Three panels over the same sentences: suggest and guess, label, suggest again."""
    loop_sentences(before, after)
    panels = (
        (MODEL_ICON, f"Model after round {before['round']} suggests", before, "model"),
        (ANNOTATOR_ICON, "Annotator labels the suggestions", before, "annotator"),
        (MODEL_ICON, f"Model after round {after['round']} suggests", after, "model"),
    )
    columns = []
    for index, (icon, title, heatmap, mode) in enumerate(panels):
        sentences = {s["pool_idx"]: s for s in heatmap["sentences"]}
        body = "".join(
            f"<p style='margin:0 0 14px;line-height:2.6'>{_loop_panel(sentences[i], mode)}</p>"
            for i in pool_idxs
        )
        border = "border-left:1.5px solid currentColor;" if index else ""
        columns.append(
            f"<div style='{border}padding:0 18px'><div style='text-align:center'>{icon}"
            f"<div style='font-size:.85em;opacity:.7;margin:2px 0 26px'>{title}</div></div>"
            f"{body}</div>"
        )
    arrow = (
        "<div style='display:flex;align-items:center;gap:8px;margin:0 18px 14px'>"
        "<div style='flex:1;border-top:1.5px solid currentColor'></div>"
        "<span style='font-size:1.05em'>Active learning loop</span>"
        "<div style='flex:1;border-top:1.5px solid currentColor'></div>"
        "<span style='margin-left:-12px;font-size:1.2em;line-height:0'>&#10148;</span></div>"
    )
    return (
        f"<div style='max-width:1300px'>{arrow}<div style='display:grid;"
        f"grid-template-columns:repeat(3,minmax(0,1fr))'>{''.join(columns)}</div></div>"
    )
