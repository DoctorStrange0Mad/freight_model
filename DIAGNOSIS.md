# Model diagnosis & improvement

## 1. Why "R2 > 0.90" is not reachable on this data
- 1.41% of training labels are corrupted: 0.71% are x2-5 too high, 0.70% are x0.2-0.5 too low.
- They are unpredictable (AUC 0.51; uniform over lane, equipment, month, weekday, distance, weight, signals; Poisson per day).
- They carry 99.5% of squared error. On clean loads R2 = 0.996-0.998. On all loads the ceiling is ~0.84.
- Random 9.5k subsets of the data reach R2 >= 0.90 only 0.5% of the time. Do not chase it; do not drop these rows from a holdout to hit it.

## 2. Other problems found
- Shipped validation_predictions.csv was broken: ~40% of rows had rate/mile from $0.08 to $100 (real range 1.67-3.18). Regenerated.
- Random K-fold understates real error by ~2x (MAE 27.7 vs 56.1 forward with day-of-year). Day-of-year does help forward, but only as a crude flat-extrapolated level; the explicit drift forecast replaces it and did better (clean MAE ~39-41 vs 56). Validate only time-forward.
- Lane priors included each row's own label at training time (target leakage). Removed.
- August labels are 3x noisier (resid std 4.3% vs 1.5%), uniformly. Inflates any window containing August.
- A slow market drift (+~1-3% over the year) is not in any feature and trees cannot extrapolate it. This was the main forward error.

## 3. Fix
Huber GBM (threshold 0.06) on rate/mile (bag of 6, noisy months down-weighted by measured out-of-fold noise) x exp(forecast drift); drift = robust linear trend + last-14-day level from week-blocked OOF residuals.

## 4. Results, forward split (train Jan-Aug, test Sep-Oct)
| | Old | v1 (LAD) | v2 (Huber + month weights) |
|---|---|---|---|
| MAE | 112.7 | 97.0 | 94.9 |
| MedAPE | 2.08% | 1.32% | 1.17% |
| MAPE | 4.93% | 4.12% | 4.11% |
| Loads within +/-5% | 89.3% | 96.4% | 94.4% |
| MAE, clean loads | 56.8 | 41.0 | 38.9 |
| R2 all loads | 0.826 | 0.829 | 0.829 |
Other windows (v2): MAE 98.6 / 111.6, MedAPE 1.09% / 1.32%.
Tradeoff: Huber lowers typical error but puts slightly fewer loads inside +/-5% than LAD.
