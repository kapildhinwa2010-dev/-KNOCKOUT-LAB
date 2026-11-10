# KNOCKOUT-LAB

Remove one block at a time and rank every component by the damage it prevents.

KNOCKOUT-LAB is an ablation lab for stream-additive models. It scores the intact run, removes each block in turn, and ranks blocks by score damage. A block whose removal costs nothing is dead weight; a block whose removal helps is a parasite.

## What it does

- Baseline score of the intact block stack.
- Per-block damage and importance when knocked out.
- Full ranking, plus a fragile-blocks shortcut.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from knockout_lab.lab import knockout_rank, fragile_blocks

blocks = [lambda y: y, lambda y: y * 0.5, lambda y: np.roll(y, 1)]
x = np.array([1.0, 0.0, 0.0, 0.0])
score = lambda out: float(out[0])

report = knockout_rank(blocks, x, score)
print(report["ranking"])
print(fragile_blocks(report, fraction=0.5))
```

## How it works

Blocks follow stream-additive semantics: state = state + block(state). Removing block k skips its write entirely. Damage is the baseline score minus the score without the block, normalized by the baseline magnitude for importance.

## Honest scope

- One-at-a-time ablation misses interaction effects between blocks.
- Damage is measured against one input and one score function; averages over corpora are the caller job.
- Tests use a tiny synthetic block stack, so no model download is needed.

## License

MIT
