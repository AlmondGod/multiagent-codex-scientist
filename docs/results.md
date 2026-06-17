# Results

All standard deviations are sample standard deviations across seeds.
Decision Changed is the fraction of agents whose second action changed after the communication checkpoint.

## Controlled Critique Ablation

Seeds: 0, 1, 2, 4, 5.

| Condition | Mean Final Score | Mean Improvement | Decision Changed | Cross-Agent Transfer |
| --- | ---: | ---: | ---: | ---: |
| self critique | 0.974884 +/- 0.001591 | 0.019771 +/- 0.001510 | 1.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| peer critique | 0.981672 +/- 0.000351 | 0.025250 +/- 0.001448 | 1.000000 +/- 0.000000 | 0.666667 +/- 0.000000 |
| peer + roles | 0.981181 +/- 0.000912 | 0.024143 +/- 0.000852 | 1.000000 +/- 0.000000 | 0.666667 +/- 0.000000 |

Peer critique improved final score by +0.006788 absolute on average, about 0.70% relative. Mean improvement increased by +0.005479, about 27.7% relative.

## Live Transfer Ablation

Seeds: 6, 7, 8, 9, 10.

| Condition | Mean Final Score | Mean Improvement | Decision Changed | Cross-Agent Transfer |
| --- | ---: | ---: | ---: | ---: |
| self critique | 0.971327 +/- 0.007087 | 0.013115 +/- 0.000647 | 1.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| peer critique | 0.981843 +/- 0.000960 | 0.022194 +/- 0.004039 | 1.000000 +/- 0.000000 | 0.733334 +/- 0.149071 |

Live peer transfer improved final score by +0.010516 absolute on average, about 1.08% relative. Mean improvement increased by +0.009079, about 69.2% relative.

## Compact CSVs

- `results/controlled_critique_5seed.csv`
- `results/live_transfer_5seed.csv`
