# IQN HOLD-Collapse — Cross-Experiment Comparison Report

Generated from `copilot-diagnostics/results/experiment_*/iqn_reward_action_diagnostic_by_seed.csv`.

Higher `hold_share` (close to 1.0) and positive `hold_score_minus_buy_score` indicate HOLD-collapse.
Experiments that reduce these values isolate the responsible factor.

## Effective configuration per experiment

| Experiment | Description |
|------------|-------------|
| `experiment_a_no_cvar` | risk_lambda=0.0 (no CVaR penalty in training/eval). Same period, same training length, same seeds as baseline. |
| `experiment_b_more_training` | 2x training steps (50k) and slower epsilon decay (40k). Same period, same seeds. |
| `experiment_c_longer_window` | Wider train window 2015-2022 (instead of 2018-2023). More regime variation in replay. Same seeds and training length. |
| `experiment_d_zero_cost` | buy_cost_pct=0 and sell_cost_pct=0. Removes per-step asymmetry where BUY is guaranteed negative reward vs HOLD=0 cost. |
| `experiment_e_lower_grad_clip` | max_norm=1.0 (was 10.0). Tests whether gradient explosion is the primary driver of Q-value divergence in seeds 2 and 5. |
| `experiment_f_state_norm` | state_norm_scale=1000. Tests whether raw FinRL state scale (cash=1M) is the root cause of Q-value divergence and HOLD-collapse in seeds 2 and 5. |
| `experiment_g_layer_norm` | use_layer_norm=true. Tests whether LayerNorm in the IQN state encoder fixes Q-value divergence by normalizing hidden activations regardless of raw input scale. |

## Seed status (active_trading / no_trade)

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | active_trading | no_trade | no_trade | active_trading | no_trade | no_trade | active_trading |
| 2 | no_trade | no_trade | no_trade | no_trade | no_trade | no_trade | active_trading |
| 3 | active_trading | active_trading | active_trading | active_trading | active_trading | no_trade | active_trading |
| 4 | active_trading | no_trade | active_trading | active_trading | no_trade | no_trade | active_trading |
| 5 | no_trade | no_trade | no_trade | no_trade | no_trade | no_trade | active_trading |

## Final training loss (lower = stable)

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 206,544 | 2,573,089 | 162,634 | 140,305 | 211,185 | 243.070 | 82.836 |
| 2 | 18,185,800 | 1,490,525,824 | 17,685,978 | 18,843,992 | 18,473,236 | 10,431 | 80.999 |
| 3 | 10,007 | 163,829 | 47,208 | 36,419 | 10,673 | 1,536 | 79.845 |
| 4 | 8,216 | 15,146 | 42,131 | 4,789 | 107,174 | 2,319 | 69.579 |
| 5 | 4,666,403 | 1,910,197,248 | 1,213,829 | 15,116,172 | 4,352,964 | 3,951 | 86.544 |

## Final total return %

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 1.052 | 0.0000 | 0.0000 | 2.119 | 0.0000 | 0.0000 | 92.750 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 89.106 |
| 3 | 93.504 | 44.108 | 29.711 | 78.672 | 93.504 | 0.0000 | 12.169 |
| 4 | 93.504 | 0.0000 | 28.450 | 28.653 | 0.0000 | 0.0000 | 89.629 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 93.504 |

## Number of trades

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 13.000 | 0.0000 | 0.0000 | 68.000 | 0.0000 | 0.0000 | 18.000 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 22.000 |
| 3 | 8.000 | 176.000 | 359.000 | 106.000 | 8.000 | 0.0000 | 244.000 |
| 4 | 8.000 | 0.0000 | 40.000 | 22.000 | 0.0000 | 0.0000 | 62.000 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.000 |

## q50(HOLD) - q50(BUY) (positive = HOLD preferred)

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 1,012 | 3,285 | 25,488 | 395.448 | 1,295 | 15.457 | -0.4549 |
| 2 | 1,083,892 | 129,610,701 | 940,436 | 1,135,279 | 1,091,186 | 663.654 | -0.6134 |
| 3 | -18.111 | 10.530 | -23.313 | -9.571 | -7.833 | 23.156 | 0.0370 |
| 4 | -15.864 | 73.694 | -49.710 | 12.747 | 225.148 | 127.115 | -0.2379 |
| 5 | 354,864 | 149,593,823 | 125,575 | 981,134 | 333,590 | 389.796 | -0.1599 |

## q50 of HOLD action

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 53,516 | 76,196 | 179,516 | 28,561 | 60,113 | 162.415 | 11.555 |
| 2 | 7,934,384 | 750,434,873 | 7,305,607 | 7,879,326 | 8,057,827 | 4,639 | 12.406 |
| 3 | -243.936 | 5,073 | -59.591 | 330.972 | -256.288 | 170.616 | 6.919 |
| 4 | -298.365 | -310.109 | -606.644 | -63.569 | 990.270 | 1,363 | 3.674 |
| 5 | 2,590,072 | 833,759,256 | 840,634 | 6,919,209 | 2,457,499 | 2,874 | 10.351 |

## q50 of BUY action

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 52,504 | 72,911 | 154,028 | 28,165 | 58,818 | 146.958 | 12.010 |
| 2 | 6,850,492 | 620,824,171 | 6,365,171 | 6,744,047 | 6,966,641 | 3,975 | 13.020 |
| 3 | -225.826 | 5,063 | -36.278 | 340.543 | -248.456 | 147.459 | 6.882 |
| 4 | -282.501 | -383.803 | -556.934 | -76.316 | 765.122 | 1,236 | 3.912 |
| 5 | 2,235,209 | 684,165,433 | 715,059 | 5,938,075 | 2,123,909 | 2,484 | 10.510 |

## Mean reward when action=BUY

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 0.0619 | 0.0780 | 0.0614 | 0.0997 | 0.0520 | 0.0392 | 0.1577 |
| 2 | 0.0446 | 0.0309 | 0.0820 | 0.0896 | 0.0445 | 0.0450 | 0.1763 |
| 3 | 0.0590 | 0.1014 | 0.2091 | 0.1392 | 0.1101 | 0.0528 | 0.1279 |
| 4 | 0.0710 | 0.1247 | 0.1333 | 0.1071 | 0.0624 | 0.0612 | 0.1179 |
| 5 | 0.0291 | 0.0165 | 0.0695 | 0.0508 | 0.0294 | 0.0267 | 0.1454 |

## Cash-only share during training

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 0.0344 | 0.2640 | 0.1334 | 0.0206 | 0.0445 | 0.1460 | 0.0004 |
| 2 | 0.1520 | 0.2913 | 0.1489 | 0.1505 | 0.1554 | 0.1559 | 0.0003 |
| 3 | 0.0009 | 0.1952 | 0.0014 | 0.0006 | 0.0009 | 0.1498 | 0.0007 |
| 4 | 0.0014 | 0.0020 | 0.0017 | 0.0010 | 0.1144 | 0.1357 | 0.0005 |
| 5 | 0.1399 | 0.2960 | 0.1471 | 0.1428 | 0.1376 | 0.1369 | 0.0003 |

## SELL action count during training

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 12,793 | 24,234 | 14,034 | 11,626 | 12,934 | 14,248 | 5,015 |
| 2 | 14,617 | 26,122 | 14,739 | 14,541 | 14,646 | 14,632 | 4,703 |
| 3 | 8,008 | 15,543 | 7,805 | 6,934 | 9,584 | 14,415 | 5,214 |
| 4 | 7,931 | 15,653 | 10,864 | 6,845 | 10,011 | 14,332 | 6,497 |
| 5 | 14,641 | 26,020 | 14,453 | 14,621 | 14,578 | 14,571 | 5,365 |

## Final epsilon

| Seed | experiment_a_no_cvar | experiment_b_more_training | experiment_c_longer_window | experiment_d_zero_cost | experiment_e_lower_grad_clip | experiment_f_state_norm | experiment_g_layer_norm |
|------|------|------|------|------|------|------|------|
| 1 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| 2 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| 3 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| 4 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| 5 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |

## How to read this

- **loss_final > 1,000,000** for a seed → Q-value divergence / gradient explosion (primary driver of HOLD-collapse).
- **hold_minus_buy_score >> 0** for a seed → HOLD dominates BUY in Q-space, agent never buys.
- **training_sell_count > 12,000** (>48% of 25k steps) → diverged policy over-sells during training, draining holdings, leaving agent cash-only at backtest start.
- If `loss_final` drops sharply in **experiment_e_lower_grad_clip** → gradient explosion was the root cause; max_norm fix resolves it.
- If `loss_final` stays high in experiment_e → need reward normalization or lower LR.
