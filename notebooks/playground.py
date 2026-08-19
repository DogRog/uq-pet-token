import marimo

__generated_with = "0.23.16"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import json
    from collections import Counter

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from uq_pet.config import NER_TAGS, RAW_DATASET_PATH

    return RAW_DATASET_PATH, alt, mo, np, pl


@app.cell
def _(RAW_DATASET_PATH, pl):
    df = pl.read_ndjson(RAW_DATASET_PATH)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df, np, pl):
    sentence_lengths = df["tokens"].map_elements(len).to_list()
    mean_length = np.mean(sentence_lengths)

    plot_df = pl.DataFrame({
        "sentence_length": sentence_lengths
    })

    hist = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "sentence_length:Q",
                bin=alt.Bin(maxbins=40),
                title="Number of Tokens",
            ),
            y=alt.Y("count()", title="Frequency"),
        )
    )

    mean_line = (
        alt.Chart(pl.DataFrame({"mean": [mean_length]}))
        .mark_rule(color="red", strokeDash=[6, 4], size=2)
        .encode(x="mean:Q")
    )

    mean_text = (
        alt.Chart(pl.DataFrame({"mean": [mean_length]}))
        .mark_text(
            text=f"Mean = {mean_length:.1f}",
            color="red",
            align="left",
            dx=5,
            dy=-10,
        )
        .encode(x="mean:Q", y=alt.value(15))
    )

    (hist + mean_line + mean_text).properties(
        title="Sentence Length Distribution",
        width=700,
        height=400,
    )
    return


if __name__ == "__main__":
    app.run()
