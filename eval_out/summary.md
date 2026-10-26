# Baseline comparison (%, mean ± SD over seeds)

closed = best of the pairs in C = S ∪ U; open = top crop + top disease (any of 55 × 175).
harmonic = 2 × seen × unseen / (seen + unseen).

| Method | Backbone | Seeds | closed_seen_top1 | closed_unseen_top1 | closed_harmonic_top1 | closed_seen_top5 | closed_unseen_top5 | closed_harmonic_top5 | open_seen_top1 | open_unseen_top1 | open_harmonic_top1 | open_seen_top5 | open_unseen_top5 | open_harmonic_top5 | crop_seen_top1 | crop_unseen_top1 | crop_seen_top5 | crop_unseen_top5 | disease_seen_top1 | disease_unseen_top1 | disease_seen_top5 | disease_unseen_top5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| independent | convnext_tiny | 3 | 70.5 ± 0.5 | 7.3 ± 0.5 | 13.2 ± 0.8 | 89.0 ± 0.4 | 36.3 ± 2.9 | 51.6 ± 3.0 | 68.7 ± 0.5 | 4.5 ± 0.6 | 8.5 ± 1.0 | 85.0 ± 0.3 | 17.9 ± 0.9 | 29.6 ± 1.2 | 90.5 ± 0.5 | 68.9 ± 1.5 | 96.4 ± 0.2 | 85.0 ± 0.7 | 71.3 ± 0.3 | 11.3 ± 1.3 | 89.4 ± 0.5 | 28.3 ± 1.2 |
| independent | efficientnet_b3 | 3 | 69.3 ± 0.3 | 7.8 ± 0.9 | 14.0 ± 1.4 | 90.0 ± 0.4 | 42.6 ± 2.3 | 57.8 ± 2.2 | 67.7 ± 0.3 | 5.2 ± 0.3 | 9.6 ± 0.5 | 85.8 ± 0.1 | 24.7 ± 3.2 | 38.3 ± 3.9 | 90.9 ± 0.4 | 74.7 ± 1.0 | 96.9 ± 0.1 | 85.6 ± 0.9 | 70.2 ± 0.1 | 10.0 ± 0.7 | 90.0 ± 0.1 | 33.3 ± 4.0 |
| independent | resnet50 | 3 | 68.5 ± 0.5 | 7.0 ± 0.8 | 12.7 ± 1.4 | 89.1 ± 0.0 | 34.9 ± 3.1 | 50.0 ± 3.2 | 67.0 ± 0.3 | 5.0 ± 0.8 | 9.3 ± 1.4 | 84.6 ± 0.2 | 19.5 ± 2.1 | 31.7 ± 2.7 | 90.2 ± 0.2 | 69.8 ± 0.7 | 96.2 ± 0.3 | 83.4 ± 0.4 | 69.5 ± 0.4 | 10.9 ± 1.7 | 89.6 ± 0.2 | 30.1 ± 1.9 |
