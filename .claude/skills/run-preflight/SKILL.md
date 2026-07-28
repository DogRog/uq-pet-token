---
name: run-preflight
description: Run ruff lint and format checks, the offline pytest suite, and a push-safety scan before committing or pushing uq-pet to GitHub. Use when asked to lint, run ruff, format the code, check the repo before a push/commit/PR, or verify it is safe to upload.
---

# Preflight: lint + test + push safety

The gate that decides whether uq-pet is safe to push. One command runs
everything: `ruff check`, `ruff format --check`, the offline pytest suite,
and a scan for things you don't want on GitHub (API keys, `.env`, run
artifacts, the missing prompt file).

All paths below are relative to the repo root (`uq-pet/`). Everything runs
through `uv run` — no separate venv activation.

## Run (agent path)

```bash
.claude/skills/run-preflight/preflight.sh
```

Exit `0` = `ready to push`. Exit `1` = `do NOT push yet`, with a `fail:`
line per blocking check. Takes ~20 s (most of it is pytest).

Flags:

| Flag | Effect |
|---|---|
| `--fix` | `ruff check --fix` + `ruff format` on `src tests` first, then re-check |
| `--no-tests` | lint/format/safety only (~5 s) — use when iterating on style |
| `--strict` | notebook lint/format failures become blocking too |

What it checks, in order:

1. `ruff check src tests` — **blocking**
2. `ruff format --check src tests` — **blocking**
3. `ruff check` + `format --check` on `notebooks/` — **advisory** (see Gotchas)
4. `uv run pytest -q --tb=line` — **blocking**
5. Push safety — **blocking**: `.env` untracked, no API-key patterns in
   tracked files, nothing tracked under `data/` or `results/`,
   `prompts/ner_v1.txt` present. Tracked files over 1 MB are a warning.

Typical green run:

```
== verdict
  warn: ruff notebooks — not blocking; see SKILL.md > Gotchas
ready to push
```

## Run the checks individually

Each of these was run standalone and is what the driver wraps:

```bash
uv run ruff check src tests --output-format concise   # lint, one line per finding
uv run ruff check --fix src tests                     # autofix what's fixable
uv run ruff format src tests                          # reformat in place
uv run ruff format --check src tests                  # verify only, no writes
uv run ruff check . --statistics                      # rule-code histogram
uv run pytest -q --tb=line                            # offline suite, 74 tests
```

## Gotchas

- **`ruff check .` fails on this repo, and it's not your code.** All 10
  findings are `E402` inside `notebooks/test.ipynb` — ruff lints `.ipynb`
  cells by default, and cell 11 re-imports at the top of a cell that isn't
  the first. `src` and `tests` are clean. That's why the driver scopes the
  blocking check to `src tests` and reports notebooks separately.
- **`E402` is not autofixable.** `--fix` will not clear the notebook
  findings; `--statistics` shows 10 errors and `[*]` never appears. To make
  `ruff check .` genuinely green, append this to the existing
  `[tool.ruff.lint.per-file-ignores]` table in `pyproject.toml` (verified —
  drops it to `All checks passed!`):

  ```toml
  "notebooks/*.ipynb" = ["E402"]
  ```

  `ruff format` would still rewrite `notebooks/test.ipynb` (it wants a blank
  line before a def in cell 5). Reformatting a notebook rewrites JSON that
  carries cell outputs — decide deliberately, don't let the gate do it. The
  driver never touches `notebooks/`.
- **Editing `pyproject.toml` triggers a rebuild.** The next `uv run` prints
  `Built uq-pet` / `Installed 1 package` before any output. Harmless, but it
  means the first post-edit run is a few seconds slower.
- **The `.env` check is about tracking, not content.** `.env` holds
  `OPENROUTER_API_KEY` and `NHR_FAU_API_KEY` and is correctly gitignored. The
  key-pattern scan covers tracked files including notebook JSON, since a
  saved cell output can carry a key that never appears in source.
- **`data/` and `results/` are gitignored** (`/data/`, `/results/`), so the
  artifact check only fires if someone used `git add -f`.
- **No CI exists.** There is no `.github/workflows/`, so nothing runs these
  checks after you push. This driver is the only gate.

## Troubleshooting

**`prompts/ner_v1.txt MISSING` and 6 tests fail** — this is the repo's
current state on `experiment`. The file was deleted in `6255c1b` ("Part 1
update"); it was last present in `ad080f1`. Its absence breaks
`load_prompt_template()`, so `test_config.py::test_prompt_path_points_into_prompts_dir`
and five `test_llm_scoring.py` tests fail with
`FileNotFoundError: No prompt template .../prompts/ner_v1.txt. Available: []`.
Restore it (verified: 74 passed):

```bash
git checkout ad080f1 -- prompts/ner_v1.txt
```

Do **not** retype the file from scratch. Its rendered text is fingerprinted
into every existing LLM score cache header (`prompt_fingerprint`, golden
value `a92de2863f0cbb3c`), and any byte-level difference makes `score-pool`
reject those caches.

**`Would reformat: notebooks/test.ipynb`** — expected, advisory. See Gotchas.

**`command not found: timeout`** — macOS has no `timeout(1)`. Don't wrap
these commands in it; they all finish in seconds.

## Human path

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Same tools, but it fails on the notebook findings above and skips the
push-safety scan. Prefer the driver.
