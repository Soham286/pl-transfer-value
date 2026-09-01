# PL Transfer Value — Model Card

## Model overview

This model estimates a professional football player's market value in euros
from a completed domestic-league season.

The estimator is an XGBoost gradient-boosted tree regressor trained on
`log1p(market_value_in_eur)`. Predictions are converted back to euros with
`expm1`.

## Intended use

The model is intended for:

- exploratory football analytics
- preliminary player valuation
- scouting support
- identifying players who may deserve further investigation

It should not be treated as a transfer-fee quote or used as the only basis
for a recruitment decision.

## Training and evaluation window

- Final training seasons: 2012–2025
- Evaluation training seasons: 2012–2024
- Held-out test season: 2025
- Final training rows: 64,622
- Held-out test rows: 4,977

The held-out test season was not used during hyperparameter tuning.

## Held-out performance

- MAE: €2.76M
- RMSE: €6.67M
- R²: 0.7984

The MAE is the most readable headline: predictions differ from the observed
market value by approximately €2.76M on average.

The larger RMSE shows that errors are substantially greater for some
high-value players.

## Target

`y = log1p(market_value_in_eur)`

Training on the logarithmic target reduces the influence of a small number
of extremely valuable players. Deployment predictions are inverted with
`expm1`.

## Exact feature order

The following order must be preserved during prediction:

1. `goals`
2. `assists`
3. `minutes_played`
4. `age`
5. `age_squared`
6. `goals_per_90`
7. `assists_per_90`
8. `goal_contributions_per_90`
9. `appearances`
10. `minutes_share`
11. `is_prime_age`
12. `club_strength_log`
13. `club_strength_available`
14. `position_Attack`
15. `position_Defender`
16. `position_Goalkeeper`
17. `position_Midfield`
18. `league_BE1`
19. `league_DK1`
20. `league_ES1`
21. `league_FR1`
22. `league_GB1`
23. `league_GR1`
24. `league_IT1`
25. `league_L1`
26. `league_NL1`
27. `league_PO1`
28. `league_RU1`
29. `league_SC1`
30. `league_TR1`
31. `league_UKR1`
32. `club_tier_Elite`
33. `club_tier_Lower`
34. `club_tier_Middle`
35. `club_tier_Unknown`
36. `club_tier_Upper`

A feature-order mismatch can produce plausible-looking but incorrect
predictions without necessarily raising an error. The same ordered list is
therefore stored inside `model.pkl` and in `feature_columns.json`.

## Best hyperparameters

- `colsample_bytree`: 1.0
- `learning_rate`: 0.05
- `max_depth`: 4
- `min_child_weight`: 1
- `n_estimators`: 600
- `reg_alpha`: 0.01
- `reg_lambda`: 2.0
- `subsample`: 0.7

## Contract-data policy

Contract features are excluded from the primary historical model.

The dataset contains current contract-expiration information, but does not
provide reliable contract snapshots for every historical player-season.
Attaching a current contract date to an old season would leak future
information into training.

Contract information may be displayed as contextual or experimental
information in the application, but this saved model does not silently
pretend that complete historical contract data exists.

## Important limitations

- Transfermarkt values are estimates, not confirmed transfer fees.
- Historical contract information is incomplete.
- Injuries and injury history are not fully represented.
- Reputation, nationality, commercial value, and negotiation conditions are
  not modeled directly.
- The data has stronger coverage for well-documented competitions.
- Market value changes over time, so model performance can decay.
- Predictions for unusual player profiles are less reliable.
- Holding-out one recent season is useful but is not the same as continuous
  production monitoring.

## Responsible interpretation

This model should generate evidence for discussion, not replace scouts,
analysts, medical staff, or contract specialists.

Generated: 2026-09-01T01:08:12.831889+00:00
