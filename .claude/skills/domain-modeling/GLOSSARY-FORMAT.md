# GLOSSARY.md Format

## Structure

```md
# {Context Name}

{One or two sentence description of what this context is and why it exists.}

## Language

**Candidate**:
{A one or two sentence description of the term}
_Avoid_: Sample, instance

**Arm**:
One of the two models updated side by side from the same bootstrap model, one acquiring by uncertainty and one at random.
_Avoid_: Branch, condition

**Round**:
One acquisition step in which each arm selects up to K new candidates and updates on them with replay.
_Avoid_: Iteration, epoch
```

## Rules

- **Be opinionated.** When multiple words exist for the same concept, pick the best one and list the others under `_Avoid_`.
- **Keep definitions tight.** One or two sentences max. Define what it IS, not what it does.
- **Only include terms specific to this project's context.** General programming concepts (timeouts, error types, utility patterns) don't belong even if the project uses them extensively. Before adding a term, ask: is this a concept unique to this context, or a general programming concept? Only the former belongs.
- **Group terms under subheadings** when natural clusters emerge. If all terms belong to a single cohesive area, a flat list is fine.

## Single vs multi-context repos

**Single context (most repos):** One `GLOSSARY.md` at the repo root.

**Multiple contexts:** A `GLOSSARY-MAP.md` at the repo root lists the contexts, where they live, and how they relate to each other. (Illustrative split; this repo is single-context today.)

```md
# Glossary Map

## Contexts

- [Data](./src/uq_pet/data/GLOSSARY.md): builds the seed, pool, and test splits
- [Acquisition](./src/uq_pet/acquisition/GLOSSARY.md): runs paired arms round by round
- [Search](./src/uq_pet/search/GLOSSARY.md): plans, resumes, and summarises configuration sweeps

## Relationships

- **Data → Acquisition**: Data hands over label-free pool inputs; Acquisition reveals a pool label only after selecting the candidate
- **Search → Acquisition**: Search freezes each configuration in a plan; Acquisition runs it and returns evaluation rows and selections
- **Data ↔ Search**: Shared `dataset` and `dataset_percent` settings, so a resumed plan rebuilds the same pool
```

The skill infers which structure applies:

- If `GLOSSARY-MAP.md` exists, read it to find contexts
- If only a root `GLOSSARY.md` exists, single context
- If neither exists, create a root `GLOSSARY.md` lazily when the first term is resolved

When multiple contexts exist, infer which one the current topic relates to. If unclear, ask.
