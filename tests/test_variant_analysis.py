import json

import polars as pl
import pytest

from utils.variant_analysis import (
    load_runs,
    matched_variants,
    paired_curves,
    round_jitter,
    seed_gaps,
    selection_redundancy,
)

SETTINGS = {"checkpoint": "bert", "uq_metric": "margin", "model_seeds": [0, 1], "k": 2}


def write_run(root, name, config, uq_f1):
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**SETTINGS, **config}))
    rows = [
        {"seed": seed, "round": r, "percent_acquired": 50.0 * r, "arm": arm, "entity_f1": f1}
        for seed in (0, 1)
        for r in range(3)
        for arm, f1 in (("random", 0.1 * r), ("uncertainty", 0.1 * r + uq_f1 * (r > 0)))
    ]
    pl.DataFrame(rows).write_csv(run / "results.csv")
    selections = [
        {"seed": 0, "round": 1, "arm": "uncertainty", "token": token, "label": label, "pool_idx": i}
        for i, (token, label) in enumerate((("the", "O"), ("The", "B-Actor"), ("dog", "O")))
    ]
    (run / "selections.json").write_text(json.dumps(selections))
    return run


def test_variants_pair_with_matching_top_k_runs_and_report_gaps(tmp_path):
    top_k = write_run(tmp_path, "plain", {}, uq_f1=0.0)
    gumbel = write_run(tmp_path, "gumbel", {"gumbel_noise": True}, uq_f1=0.1)
    write_run(tmp_path, "other_k", {"k": 3, "gumbel_noise": True}, uq_f1=0.1)

    runs = load_runs(tmp_path)
    assert sorted(runs["variant"]) == ["Gumbel", "Gumbel", "top-K"]
    pairs = matched_variants(runs)
    assert pairs.select("baseline_run", "run").rows() == [("plain", "gumbel")]

    curves = paired_curves(top_k, gumbel)
    assert (curves["random"] == curves["variant_random"]).all()
    gaps = seed_gaps(curves)
    # Gap is 0 at 0% and 0.1 at 50% and 100%: trapezoid AUC 0.075 over a 100-point span.
    assert gaps["variant_minus_top_k_auc"].to_list() == pytest.approx([0.075, 0.075])
    assert gaps["variant_minus_top_k_final"].to_list() == pytest.approx([0.1, 0.1])
    assert gaps["top_k_minus_random_auc"].to_list() == pytest.approx([0.0, 0.0])
    assert gaps["random_rerun_auc"].to_list() == pytest.approx([0.0, 0.0])

    jitter = round_jitter(curves).filter(pl.col("arm") == "variant")
    assert jitter["mean |step| (pp)"][0] == pytest.approx(15.0)

    redundancy = selection_redundancy(gumbel, "Gumbel").row(0, named=True)
    assert redundancy["selection"] == "Gumbel"
    assert redundancy["repeated_share"] == pytest.approx(1 / 3)
    assert redundancy["o_share"] == pytest.approx(2 / 3)
