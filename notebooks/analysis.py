import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    from collections import Counter

    import marimo as mo
    import pandas as pd
    import yaml
    from matplotlib.figure import Figure

    from uq_pet.config import NER_TAGS, RESULTS_DIR
    from uq_pet.dataset import sentence_key, split_dataset, to_examples

    return (
        Counter,
        Figure,
        NER_TAGS,
        RESULTS_DIR,
        json,
        mo,
        pd,
        re,
        sentence_key,
        split_dataset,
        to_examples,
        yaml,
    )


@app.cell
def _(RESULTS_DIR, re):
    # Accept both legacy second-resolution run names and current microsecond names.
    _stamp = re.compile(r"_\d{8}_\d{6}(?:_\d{6})?$")
    _needed = ("results.csv", "selection.json", "config.yaml")

    def available_runs():
        """Return the newest complete result directory for each config."""
        found = {}
        for path in sorted(RESULTS_DIR.glob("*/"), reverse=True):
            if all((path / name).exists() for name in _needed):
                found.setdefault(_stamp.sub("", path.name), path)
        return found

    latest = available_runs()
    return (latest,)


@app.cell
def _(latest, pd):
    FACTOR_RUNS = {
        "Number of shots": {
            "1": "nhr_gemma4_1shot",
            "2": "nhr_gemma4_2shot",
            "5": "nhr_gemma4_5shot",
            "10": "nhr_gemma4_10shot",
        },
        "LLM": {
            "Gemma 4": "nhr_gemma4_1shot",
            "GPT-OSS 120B": "nhr_gpt_oss_topk",
            "Kimi K2.6": "nhr_kimi2.6_topk",
            "Mistral 3.5": "nhr_mistral3.5",
        },
        "Few-shot seed": {
            "1": "nhr_gemma4_1shot_seed1",
            "7": "nhr_gemma4_1shot_seed7",
            "13": "nhr_gemma4_1shot",
        },
        "Encoder": {
            "DistilBERT cased": "nhr_gemma4_1shot",
            "BERT cased": "nhr_gemma4_bert",
            "RoBERTa": "nhr_gemma4_roberta",
            "DistilBERT uncased": "nhr_gemma4_distilbert_uncased",
        },
        "Stopping rule": {
            "20 fixed epochs": "nhr_gemma4_1shot",
            "early stopping": "nhr_gemma4_early_stopping",
        },
    }
    UQ_ARMS = {
        "avg_neg_logprob_filtered",
        "avg_neg_logprob_pure",
        "least_confident",
        "vote_entropy",
        "disagreement",
        "pairwise_f1_disagreement",
        "entity_count_std",
    }

    _required = {stem for settings in FACTOR_RUNS.values() for stem in settings.values()}
    _missing = sorted(_required - latest.keys())
    if _missing:
        raise FileNotFoundError(
            "Run the missing configs before opening this notebook: " + ", ".join(_missing)
        )

    run_frames = {}
    for _stem in _required:
        _frame = pd.read_csv(latest[_stem] / "results.csv")
        _frame["budget_pct"] = _frame["budget_pct"].astype(int)
        run_frames[_stem] = _frame

    shared_budgets = sorted(
        set.intersection(*(set(frame["budget_pct"]) for frame in run_frames.values()))
    )
    return FACTOR_RUNS, UQ_ARMS, run_frames, shared_budgets


@app.cell
def _(FACTOR_RUNS, UQ_ARMS, pd, run_frames):
    _rows = []
    for _factor, _settings in FACTOR_RUNS.items():
        for _setting, _stem in _settings.items():
            _means = run_frames[_stem].groupby(["budget_pct", "arm"])["entity_f1"].mean()
            for _budget in _means.index.get_level_values("budget_pct").unique():
                _at_budget = _means.loc[_budget]
                _random_f1 = float(_at_budget["random"])
                _uq = _at_budget[[arm.split(":", 1)[0] in UQ_ARMS for arm in _at_budget.index]]
                _gaps = _uq - _random_f1
                _rows.append(
                    {
                        "factor": _factor,
                        "setting": _setting,
                        "run": _stem,
                        "budget_pct": int(_budget),
                        "random_f1": _random_f1,
                        "mean_uq_f1": float(_uq.mean()),
                        "mean_gap": float(_gaps.mean()),
                        "best_gap": float(_gaps.max()),
                        "best_arm": str(_gaps.idxmax()),
                    }
                )

    effects = pd.DataFrame(_rows)
    return (effects,)


@app.cell
def _(mo, shared_budgets):
    budget = mo.ui.slider(
        steps=shared_budgets,
        value=10 if 10 in shared_budgets else shared_budgets[0],
        show_value=True,
        label="Budget (% of pool)",
        full_width=True,
    )
    return (budget,)


@app.cell(hide_code=True)
def _(budget, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                # What changes uncertainty sampling?

                PET process-extraction NER · entity-level micro F1 · means over training seeds

                **UQ lift = mean F1 of uncertainty-selected arms − random-selection F1.**
                Positive values favor uncertainty sampling within the same run.
                """
            ),
            budget,
        ]
    )
    return


@app.cell
def _(budget, effects):
    selected_effects = effects[effects["budget_pct"] == budget.value].copy()
    return (selected_effects,)


@app.cell(hide_code=True)
def _(Figure, budget, mo, selected_effects):
    _fig = Figure(figsize=(10.5, 7.4), layout="constrained")
    _axes = _fig.subplots(3, 2)

    for _ax, (_factor, _group) in zip(
        _axes.flat, selected_effects.groupby("factor", sort=False), strict=False
    ):
        _group = _group.sort_values("mean_gap")
        _colors = ["#4f7664" if value >= 0 else "#b56355" for value in _group["mean_gap"]]
        _positions = range(len(_group))
        _ax.axvline(0, color="#8b8b88", linewidth=0.8)
        _ax.scatter(_group["mean_gap"], _positions, color=_colors, s=45, zorder=3)
        for _position, _value in zip(_positions, _group["mean_gap"], strict=True):
            _ax.plot([0, _value], [_position, _position], color="#cbc9c2", linewidth=1.2)
        _ax.set_yticks(list(_positions), _group["setting"], fontsize=8)
        _ax.set_title(_factor, loc="left", fontsize=11, fontweight="semibold")
        _ax.set_xlabel("mean UQ lift (Δ F1)")
        _ax.grid(axis="x", color="#e8e7e2", linewidth=0.7)
        _ax.spines[["top", "right", "left"]].set_visible(False)

    _fig.delaxes(_axes.flat[-1])
    _fig.suptitle(f"Observed effects at a {budget.value}% budget", fontsize=13)
    mo.as_html(_fig)
    return


@app.cell
def _(selected_effects):
    effect_table = selected_effects[
        ["factor", "setting", "random_f1", "mean_uq_f1", "mean_gap", "best_gap", "best_arm"]
    ].copy()
    for _column in ("random_f1", "mean_uq_f1"):
        effect_table[_column] = effect_table[_column].map(lambda value: f"{value:.3f}")
    for _column in ("mean_gap", "best_gap"):
        effect_table[_column] = effect_table[_column].map(lambda value: f"{value:+.3f}")
    effect_table = effect_table.rename(
        columns={
            "factor": "choice",
            "random_f1": "random F1",
            "mean_uq_f1": "mean UQ F1",
            "mean_gap": "mean UQ Δ",
            "best_gap": "best UQ Δ",
            "best_arm": "best arm",
        }
    )
    return (effect_table,)


@app.cell(hide_code=True)
def _(effect_table, mo):
    mo.ui.table(
        effect_table,
        selection=None,
        pagination=False,
        show_search=False,
        show_download=False,
        show_data_types=False,
        freeze_columns_left=["choice"],
    )
    return


@app.cell
def _(budget, run_frames, selected_effects):
    _takeaways = []
    for _factor, _group in selected_effects.groupby("factor", sort=False):
        _best = _group.loc[_group["mean_gap"].idxmax()]
        _worst = _group.loc[_group["mean_gap"].idxmin()]
        _takeaways.append(
            f"**{_factor}:** {_best['setting']} has the largest observed lift "
            f"({_best['mean_gap']:+.3f}); {_worst['setting']} the smallest "
            f"({_worst['mean_gap']:+.3f})."
        )

    _encoder = selected_effects[selected_effects["factor"] == "Encoder"]
    _best_encoder = _encoder.loc[_encoder["random_f1"].idxmax()]
    _takeaways.append(
        f"**Encoder level:** {_best_encoder['setting']} has the highest random baseline "
        f"({float(_best_encoder['random_f1']):.3f})."
    )

    _early = run_frames["nhr_gemma4_early_stopping"]
    _early = _early[_early["budget_pct"] == budget.value]["best_epoch"].dropna()
    early_epoch_range = (int(_early.min()), float(_early.median()), int(_early.max()))
    takeaways = _takeaways
    return early_epoch_range, takeaways


@app.cell(hide_code=True)
def _(early_epoch_range, mo, takeaways):
    mo.md(
        "\n".join(f"- {line}" for line in takeaways)
        + (
            "\n- **Stopping detail:** early stopping selected epochs "
            f"{early_epoch_range[0]}–{early_epoch_range[2]} "
            f"(median {early_epoch_range[1]:.0f})."
        )
        + "\n\nThe early-stopping run holds out 20% of each selected set, so compare its "
        "within-run UQ lift; its raw F1 is not a like-for-like fixed-epoch comparison."
    ).callout(kind="neutral")
    return


@app.cell
def _(
    Counter,
    NER_TAGS,
    UQ_ARMS,
    json,
    latest,
    pd,
    sentence_key,
    split_dataset,
    to_examples,
    yaml,
):
    _primary_dir = latest["nhr_gemma4_1shot"]
    _config = yaml.safe_load((_primary_dir / "config.yaml").read_text())
    _selection = json.loads((_primary_dir / "selection.json").read_text())
    _, _pool, _ = split_dataset(
        n_few_shot=_config["n_few_shot"], few_shot_seed=_config["few_shot_seed"]
    )
    _pool_examples = to_examples(_pool)
    _pool_by_key = {sentence_key(example): example for example in _pool_examples}
    _pool_counts = Counter(tag_id for example in _pool_examples for tag_id in example["ner-tags"])
    _pool_tokens = sum(_pool_counts.values())
    pool_lengths = [len(example["tokens"]) for example in _pool_examples]

    _distribution_rows = []
    _distance_rows = []
    _length_rows = []
    for _budget, _arms in _selection.items():
        for _arm, _keys in _arms.items():
            _selected_lengths = [len(_pool_by_key[key]["tokens"]) for key in _keys]
            _selected_counts = Counter(
                tag_id for key in _keys for tag_id in _pool_by_key[key]["ner-tags"]
            )
            _selected_tokens = sum(_selected_counts.values())
            _absolute_shifts = []
            for _tag_id, _tag in enumerate(NER_TAGS):
                _pool_share = _pool_counts[_tag_id] / _pool_tokens
                _selected_share = _selected_counts[_tag_id] / _selected_tokens
                _shift_pp = 100 * (_selected_share - _pool_share)
                _absolute_shifts.append(abs(_shift_pp))
                _distribution_rows.append(
                    {
                        "budget_pct": int(_budget),
                        "arm": _arm,
                        "tag": _tag,
                        "pool_share": _pool_share,
                        "selected_share": _selected_share,
                        "shift_pp": _shift_pp,
                    }
                )
            _distance_rows.append(
                {
                    "budget_pct": int(_budget),
                    "arm": _arm,
                    "family": "UQ" if _arm.split(":", 1)[0] in UQ_ARMS else "control",
                    "selected_tokens": _selected_tokens,
                    "total_variation": sum(_absolute_shifts) / 200,
                }
            )
            _length_rows.append(
                {
                    "budget_pct": int(_budget),
                    "arm": _arm,
                    "family": "UQ" if _arm.split(":", 1)[0] in UQ_ARMS else "control",
                    "mean_sentence_tokens": sum(_selected_lengths) / len(_selected_lengths),
                    "median_sentence_tokens": float(pd.Series(_selected_lengths).median()),
                    "selected_tokens": sum(_selected_lengths),
                }
            )

    bio_distribution = pd.DataFrame(_distribution_rows)
    bio_distance = pd.DataFrame(_distance_rows)
    length_selection = pd.DataFrame(_length_rows)
    return bio_distance, bio_distribution, length_selection, pool_lengths


@app.cell
def _(length_selection, run_frames):
    _performance = (
        run_frames["nhr_gemma4_1shot"]
        .groupby(["budget_pct", "arm"], as_index=False)["entity_f1"]
        .mean()
    )
    _random = _performance[_performance["arm"] == "random"][["budget_pct", "entity_f1"]]
    _random = _random.rename(columns={"entity_f1": "random_f1"})
    length_performance = length_selection.merge(_performance, on=["budget_pct", "arm"])
    length_performance = length_performance.merge(_random, on="budget_pct")
    length_performance["f1_gap"] = length_performance["entity_f1"] - length_performance["random_f1"]
    return (length_performance,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Sentence length

    A sentence budget can hide a token budget. The left panel shows which arms select
    longer sentences; the right checks whether that extra length tracks F1 lift in the
    primary Gemma 4 experiment. The dedicated `length` arm is a control, not UQ.
    """
            )
        ]
    )
    return


@app.cell(hide_code=True)
def _(Figure, budget, length_performance, mo, pool_lengths):
    _arm_labels = {
        "random": "random",
        "avg_neg_logprob_filtered": "avg NLP",
        "vote_entropy": "vote entropy",
        "pairwise_f1_disagreement": "pairwise F1",
        "entity_count_std": "entity-count std",
        "least_confident": "least confident",
        "length": "length control",
        "confident:avg_neg_logprob_filtered": "most confident",
    }
    _at_budget = length_performance[length_performance["budget_pct"] == budget.value].copy()
    _at_budget["label"] = _at_budget["arm"].map(_arm_labels)
    _at_budget = _at_budget.sort_values("mean_sentence_tokens")
    _pool_mean = sum(pool_lengths) / len(pool_lengths)

    _fig = Figure(figsize=(11.5, 5.2), layout="constrained")
    _length_ax, _effect_ax = _fig.subplots(1, 2)

    _positions = range(len(_at_budget))
    _colors = ["#557663" if family == "UQ" else "#8b8b88" for family in _at_budget["family"]]
    _length_ax.axvline(_pool_mean, color="#b56355", linewidth=1.1, label="pool mean")
    _length_ax.scatter(
        _at_budget["mean_sentence_tokens"], _positions, color=_colors, s=45, zorder=3
    )
    _length_ax.set_yticks(list(_positions), _at_budget["label"], fontsize=8)
    _length_ax.set_xlabel("selected tokens per sentence (mean)")
    _length_ax.set_title("Length of selected sentences", loc="left", fontweight="semibold")
    _length_ax.legend(frameon=False, fontsize=8)
    _length_ax.grid(axis="x", color="#e8e7e2", linewidth=0.7)
    _length_ax.spines[["top", "right", "left"]].set_visible(False)

    _effect_ax.axhline(0, color="#8b8b88", linewidth=0.8)
    _effect_ax.axvline(_pool_mean, color="#b56355", linewidth=1.1)
    _effect_ax.scatter(
        _at_budget["mean_sentence_tokens"],
        _at_budget["f1_gap"],
        color=_colors,
        s=45,
        zorder=3,
    )
    for _row in _at_budget.itertuples():
        _effect_ax.annotate(
            _row.label,
            (_row.mean_sentence_tokens, _row.f1_gap),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    _effect_ax.set_xlabel("selected tokens per sentence (mean)")
    _effect_ax.set_ylabel("Δ F1 vs random")
    _effect_ax.set_title("Length versus downstream lift", loc="left", fontweight="semibold")
    _effect_ax.grid(color="#e8e7e2", linewidth=0.7)
    _effect_ax.spines[["top", "right"]].set_visible(False)
    mo.as_html(_fig)
    return


@app.cell
def _(budget, length_performance, pd, pool_lengths):
    _at_budget = length_performance[length_performance["budget_pct"] == budget.value]
    _uq = _at_budget[_at_budget["family"] == "UQ"]
    _longest = _at_budget.loc[_at_budget["mean_sentence_tokens"].idxmax()]
    _length_control = _at_budget[_at_budget["arm"] == "length"].iloc[0]
    length_summary = {
        "pool_mean": sum(pool_lengths) / len(pool_lengths),
        "pool_median": float(pd.Series(pool_lengths).median()),
        "uq_min": float(_uq["mean_sentence_tokens"].min()),
        "uq_max": float(_uq["mean_sentence_tokens"].max()),
        "longest_arm": str(_longest["arm"]),
        "longest_mean": float(_longest["mean_sentence_tokens"]),
        "length_gap": float(_length_control["f1_gap"]),
    }
    return (length_summary,)


@app.cell(hide_code=True)
def _(budget, length_summary, mo):
    mo.md(
        f"At **{budget.value}%**, the pool averages **{length_summary['pool_mean']:.1f} tokens** "
        f"per sentence (median {length_summary['pool_median']:.0f}); UQ selections average "
        f"**{length_summary['uq_min']:.1f}–{length_summary['uq_max']:.1f}**. "
        f"The longest selection is **{length_summary['longest_arm']}** at "
        f"**{length_summary['longest_mean']:.1f}**, and the pure length control changes F1 by "
        f"**{length_summary['length_gap']:+.3f}** versus random."
    ).callout(kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ## BIO-tag distribution

    The pool chart shows gold-token prevalence. The heatmap shows how each selected set
    moves that distribution in percentage points. All arms receive the same number of
    sentences, but not necessarily the same number or mix of tokens.
    """
            )
        ]
    )
    return


@app.cell(hide_code=True)
def _(Figure, NER_TAGS, bio_distribution, budget, mo):
    ARM_LABELS = {
        "random": "random",
        "avg_neg_logprob_filtered": "avg NLP",
        "vote_entropy": "vote entropy",
        "pairwise_f1_disagreement": "pairwise F1",
        "entity_count_std": "entity-count std",
        "least_confident": "least confident",
        "length": "length",
        "confident:avg_neg_logprob_filtered": "most confident",
    }
    _at_budget = bio_distribution[bio_distribution["budget_pct"] == budget.value]
    _pool_shares = (
        _at_budget.drop_duplicates("tag").set_index("tag").loc[NER_TAGS, "pool_share"] * 100
    )
    _shift = _at_budget.pivot(index="arm", columns="tag", values="shift_pp").reindex(
        columns=NER_TAGS
    )
    _shift = _shift.reindex([arm for arm in ARM_LABELS if arm in _shift.index])
    _limit = max(1.0, float(abs(_shift.to_numpy()).max()))

    _fig = Figure(figsize=(12, 6.1), layout="constrained")
    _grid = _fig.add_gridspec(1, 2, width_ratios=(1.0, 2.35))
    _pool_ax = _fig.add_subplot(_grid[0, 0])
    _heat_ax = _fig.add_subplot(_grid[0, 1])

    _positions = list(range(len(NER_TAGS)))
    _pool_ax.barh(_positions, _pool_shares, color="#557663")
    _pool_ax.set_yticks(_positions, NER_TAGS, fontsize=8)
    _pool_ax.invert_yaxis()
    _pool_ax.set_xlabel("pool token share (%)")
    _pool_ax.set_title("Eligible pool", loc="left", fontsize=11, fontweight="semibold")
    _pool_ax.grid(axis="x", color="#e8e7e2", linewidth=0.7)
    _pool_ax.spines[["top", "right", "left"]].set_visible(False)

    _image = _heat_ax.imshow(
        _shift.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-_limit, vmax=_limit
    )
    _heat_ax.set_xticks(range(len(NER_TAGS)), NER_TAGS, rotation=55, ha="right", fontsize=8)
    _heat_ax.set_yticks(
        range(len(_shift.index)), [ARM_LABELS[arm] for arm in _shift.index], fontsize=8
    )
    _heat_ax.set_title(
        f"Selection shift · {budget.value}% budget",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    _heat_ax.set_xlabel("gold BIO tag")
    _fig.colorbar(_image, ax=_heat_ax, label="difference from pool (pp)", shrink=0.8)
    mo.as_html(_fig)
    return


@app.cell
def _(bio_distance, bio_distribution, budget):
    _distance = bio_distance[bio_distance["budget_pct"] == budget.value].copy()
    _uq = _distance[_distance["family"] == "UQ"].sort_values("total_variation")
    closest_uq = _uq.iloc[0]
    farthest_uq = _uq.iloc[-1]
    random_distance = float(_distance.loc[_distance["arm"] == "random", "total_variation"].iloc[0])
    token_range = (
        int(_distance["selected_tokens"].min()),
        int(_distance["selected_tokens"].max()),
    )

    _farthest_shifts = bio_distribution[
        (bio_distribution["budget_pct"] == budget.value)
        & (bio_distribution["arm"] == farthest_uq["arm"])
    ]
    _largest = _farthest_shifts.loc[_farthest_shifts["shift_pp"].abs().idxmax()]
    farthest_shift = (str(_largest["tag"]), float(_largest["shift_pp"]))
    return (
        closest_uq,
        farthest_shift,
        farthest_uq,
        random_distance,
        token_range,
    )


@app.cell(hide_code=True)
def _(
    budget,
    closest_uq,
    farthest_shift,
    farthest_uq,
    mo,
    random_distance,
    token_range,
):
    mo.hstack(
        [
            mo.stat(f"{random_distance:.3f}", "random TV distance"),
            mo.stat(
                f"{closest_uq['total_variation']:.3f}",
                "closest UQ arm",
                str(closest_uq["arm"]),
            ),
            mo.stat(
                f"{farthest_uq['total_variation']:.3f}",
                "farthest UQ arm",
                str(farthest_uq["arm"]),
            ),
        ],
        widths="equal",
        gap=1,
    )
    mo.md(
        f"At **{budget.value}%**, selections contain **{token_range[0]:,}–"
        f"{token_range[1]:,} labelled tokens**. The largest shift in the farthest UQ arm is "
        f"**{farthest_shift[0]} ({farthest_shift[1]:+.2f} pp)**."
    )
    return


if __name__ == "__main__":
    app.run()
