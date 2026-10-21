"""Check model snapshots, the heatmap they score, and its HTML and LaTeX rendering."""

import json

import pytest
import torch

import suggestion_heatmap as runner
from analysis import suggestion_heatmap as sh
from uq_pet import active_learning, experiment


@pytest.fixture
def snapshot_run(tiny_experiment, tmp_path, monkeypatch):
    splits, kwargs = tiny_experiment
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                key: kwargs[key]
                for key in (
                    "checkpoint",
                    "k",
                    "bootstrap_epochs",
                    "batch_size",
                    "score_batch_size",
                    "max_length",
                )
            }
        )
    )
    for module in (experiment, runner):
        monkeypatch.setattr(module, "load_splits", lambda *_: (splits, {}))
        monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    root = tmp_path / "run"
    runner.main(["train", "--config", str(config), "--rounds", "0", "1", "--out", str(root)])
    return root


@pytest.mark.parametrize("round_idx", [0, 1])
def test_snapshot_scores_reproduce_the_next_rounds_selection(snapshot_run, round_idx):
    assert runner.main(["score", str(snapshot_run), "--round", str(round_idx)]) == 0
    heatmap = json.loads((snapshot_run / f"heatmap_round_{round_idx:03d}.json").read_text())
    assert heatmap["max_score_difference"] < 1e-6
    assert (heatmap["uq_metric"], heatmap["model_seed"], heatmap["k"]) == ("margin", 0, 2)
    assert all(len(s["predicted"]) == len(s["tokens"]) for s in heatmap["sentences"])
    words = [
        (score, labelled, suggested)
        for s in heatmap["sentences"]
        for score, labelled, suggested in zip(
            s["scores"], s["labelled"], s["suggested"], strict=True
        )
    ]
    assert sum(labelled for _, labelled, _ in words) == 2 * round_idx
    suggestions = [score for score, _, suggested in words if suggested]
    unlabelled = sorted((score for score, labelled, _ in words if not labelled), reverse=True)
    assert sorted(suggestions, reverse=True) == unlabelled[:2]


def test_score_rejects_a_round_without_a_snapshot(snapshot_run):
    with pytest.raises(SystemExit):
        runner.main(["score", str(snapshot_run), "--round", "2"])


def test_train_refuses_to_overwrite_a_run(snapshot_run, tmp_path):
    with pytest.raises(SystemExit):
        runner.main(
            ["train", "--config", str(tmp_path / "settings.json"), "--out", str(snapshot_run)]
        )


def test_snapshot_rounds_need_a_folder_and_must_fit_the_schedule(tiny_experiment, tmp_path):
    splits, kwargs = tiny_experiment
    with pytest.raises(ValueError, match="needs snapshot_dir"):
        active_learning.run_active_learning(*splits, **kwargs, snapshot_rounds=[1])
    with pytest.raises(ValueError, match="must not exceed"):
        active_learning.run_active_learning(
            *splits,
            **{**kwargs, "model_seeds": [0]},
            snapshot_dir=tmp_path,
            snapshot_rounds=[99],
        )


SENTENCE = {
    "document_name": "doc-1",
    "sentence_id": 0,
    "tokens": ["The", "clerk", "checks", "50%", "."],
    "tags": ["O", "B-Actor", "I-Actor", "O", "O"],
    "scores": [0.1, 0.9, 0.5, 0.25, None],
    "predicted": ["O", "B-Activity", "I-Actor", "O", "O"],
    "labelled": [True, False, False, False, False],
    "suggested": [False, True, False, False, False],
}


def test_latex_shades_suggests_fades_and_underlines_entities():
    latex = sh.heatmap_latex([SENTENCE])
    assert (
        r"\textcolor{gray}{The} \underline{\fcolorbox{uqsuggest}{uqheat!90}{\strut clerk} "
        r"\colorbox{uqheat!50}{\strut checks}} \colorbox{uqheat!25}{\strut 50\%} ."
    ) in latex
    assert r"\definecolor{uqheat}{HTML}{3987E5}" in latex


def test_html_escapes_words_and_marks_suggestions():
    sentence = {**SENTENCE, "tokens": ["<a>", *SENTENCE["tokens"][1:]]}
    rendered = sh.heatmap_html([sentence], "margin")
    assert "&lt;a&gt;" in rendered and "<a>" not in rendered
    assert rendered.count("outline:") == 1
    assert 'title="Actor"' in rendered and 'title="margin 0.900"' in rendered


def test_counts_and_files_follow_the_saved_heatmaps(tmp_path):
    later = {**SENTENCE, "sentence_id": 1}
    heatmap = {"sentences": [later, SENTENCE, {**SENTENCE, "document_name": "doc-2"}]}
    sentences = sh.document_sentences(heatmap, "doc-1")
    assert [s["sentence_id"] for s in sentences] == [0, 1]
    assert sh.document_names(heatmap) == ["doc-1", "doc-2"]
    assert sh.suggestion_counts(sentences) == {
        "words": 10,
        "labelled": 2,
        "suggested": 2,
        "suggested_in_entities": 2,
    }
    run = tmp_path / "suggestion_heatmap" / "distilbert-margin-seed0"
    run.mkdir(parents=True)
    for round_idx in (10, 0):
        (run / f"heatmap_round_{round_idx:03d}.json").write_text("{}")
    assert list(sh.heatmap_files(tmp_path)) == [
        "distilbert-margin-seed0 · round 0",
        "distilbert-margin-seed0 · round 10",
    ]


def test_tag_labels_show_bio_only_when_it_is_the_difference():
    assert sh.tag_label("B-Actor", "I-Actor") == "B-Actor"
    assert sh.tag_label("B-Activity", "B-Actor") == "Activity"
    assert sh.tag_label("O", "B-Actor") == "O"


def loop_round(round_idx, suggested):
    sentence = {**SENTENCE, "pool_idx": 0, "suggested": suggested}
    short = {**sentence, "pool_idx": 1, "tokens": ["Go", "."], "tags": ["O", "O"]}
    short.update({key: sentence[key][:2] for key in ("scores", "predicted", "labelled")})
    short["suggested"] = [False, False]
    return {"run_dir": "run", "round": round_idx, "sentences": [sentence, short]}


def test_loop_guesses_then_corrects_then_suggests_again():
    before = loop_round(10, [False, True, False, False, False])
    after = loop_round(11, [False, False, True, False, False])
    assert [s["pool_idx"] for s in sh.loop_sentences(before, after)] == [0]
    # A correctly guessed suggestion ranks after a corrected one, however short.
    short = before["sentences"][1]
    short["suggested"], after["sentences"][1]["suggested"] = [True, False], [True, False]
    assert [s["pool_idx"] for s in sh.loop_sentences(before, after)] == [0, 1]
    short["suggested"], after["sentences"][1]["suggested"] = [False, False], [False, False]
    rendered = sh.loop_html(before, after, [0])
    assert "Activity?" in rendered
    assert ">Activity</span> → <span" in rendered and ">Actor</span>" in rendered
    assert "I-Actor?" not in rendered and "Actor?" in rendered
    assert "after round 10" in rendered and "after round 11" in rendered
    with pytest.raises(ValueError, match="consecutive"):
        sh.loop_sentences(before, loop_round(12, after["sentences"][0]["suggested"]))


def test_loop_pairs_need_consecutive_rounds(tmp_path):
    run = tmp_path / "suggestion_heatmap" / "distilbert-margin-seed0"
    run.mkdir(parents=True)
    for round_idx in (0, 10, 11, 25):
        (run / f"heatmap_round_{round_idx:03d}.json").write_text("{}")
    assert list(sh.loop_pairs(tmp_path)) == ["distilbert-margin-seed0 · rounds 10→11"]
