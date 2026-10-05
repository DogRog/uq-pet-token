# ADR Format

ADRs live in `docs/adr/` and use sequential numbering: `0001-slug.md`, `0002-slug.md`, etc.

Create the `docs/adr/` directory lazily: only when the first ADR is needed.

## Template

```md
# {Short title of the decision}

{1-3 sentences: what's the context, what did we decide, and why.}
```

That's it. An ADR can be a single paragraph. The value is in recording *that* a decision was made and *why*, not in filling out sections.

## Optional sections

Only include these when they add genuine value. Most ADRs won't need them.

- **Status** frontmatter (`proposed | accepted | deprecated | superseded by ADR-NNNN`): useful when decisions are revisited
- **Considered Options**: only when the rejected alternatives are worth remembering
- **Consequences**: only when non-obvious downstream effects need to be called out

## Numbering

Scan `docs/adr/` for the highest existing number and increment by one.

## When to offer an ADR

All three of these must be true:

1. **Hard to reverse**: the cost of changing your mind later is meaningful
2. **Surprising without context**: a future reader will look at the code and wonder "why on earth did they do it this way?"
3. **The result of a real trade-off**: there were genuine alternatives and you picked one for specific reasons

If a decision is easy to reverse, skip it: you'll just reverse it. If it's not surprising, nobody will wonder why. If there was no real alternative, there's nothing to record beyond "we did the obvious thing."

### What qualifies

- **Architectural shape.** "Sentence-level LLM selection lives in a separate repository; no code is ported between the two." "Both arms clone one bootstrap-trained model and optimizer state, then keep their own state across rounds."
- **Integration patterns between contexts.** "Seed workers run in spawned processes and publish progress only after both arms finish a round, never a half-finished one."
- **Technology choices that carry lock-in.** Dataset library pins, experiment tracker, encoder family, compute target. Not every library: just the ones that would invalidate saved results or take weeks to swap out. "`datasets` stays pinned to 2.19.2 so the splits never shift."
- **Boundary and scope decisions.** "The test split is evaluation-only; tuning reads only the validation holdout." The explicit no-s are as valuable as the yes-s.
- **Deliberate deviations from the obvious path.** "We mask unselected positions with `-100` instead of replacing input words with `[MASK]`, because X." Anything where a reasonable reader would assume the opposite. These stop the next engineer from "fixing" something that was deliberate.
- **Constraints not visible in the code.** "Saved plans leave the full PET pool implicit, so sweeps saved before the `dataset` setting existed still resume."
- **Rejected alternatives when the rejection is non-obvious.** If you considered YAML configs or a CLI framework and kept validated settings in `config.py`, record why; otherwise someone will suggest YAML again in six months.
