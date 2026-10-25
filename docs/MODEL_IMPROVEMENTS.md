# Model improvements for malaria and cholera (version 0.4)

This note records why the model changed between 0.3 and 0.4, how the changes
were evaluated, what the evaluation does and does not show, and what should come
next. Version 0.4 also renames the service from `chap-malaria-forecasting` to
`chap-disease-forecasting`, because the same model now targets both diseases.

## 1. What the literature points to

| Finding | Source | Consequence for the model |
|---|---|---|
| Recent case history ("autoregressive memory") is the strongest predictor of malaria incidence; rainfall acts with a lag and saturates non-linearly | [Ghana, SEM + SHAP](https://link.springer.com/article/10.1186/s12982-026-02163-y) | keep rich case lags; trees capture saturation |
| Malaria climate lags vary by climate zone, from weeks to about 3 months; one fixed lag for all variables is a known limitation | [Zambia](https://link.springer.com/article/10.1186/s12936-025-05644-8), [Kenya](https://www.nature.com/articles/s41598-017-02560-z), [Ghana, Amenfi West](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5307250/), [Nigeria](https://link.springer.com/article/10.1186/s12879-026-14196-4) | many lags (0 to 12 weeks) and windows up to 16 weeks; let the model choose per zone |
| Cholera responds to rainfall and temperature at short lags (up to about 3 weeks), differently by district | [Ethiopia early warning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12447798/), [ANN study](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0224813) | rainfall in the origin week (lag 0) and rainfall anomalies against each zone's own history |
| Neighbouring districts carry information about each other's cholera risk | [Uganda, Bayesian networks](https://jquinn.air.ug/files/Mubangizi_2009_Cholera.pdf) | neighbour incidence, level and growth as features |
| Cholera forecasts benefit from spatial information and rainfall | [real-time cholera projections](https://www.sciencedirect.com/science/article/pii/S0309170816305218), [ML review](https://www.researchgate.net/publication/385635048_Recent_Advances_in_Machine-Learning_Driven_Cholera_Research_A_Review) | same as above |
| Case counts are overdispersed with many zeros; Gaussian likelihoods are inappropriate; Tweedie trains far faster than a custom negative binomial in LightGBM | [retail GBDT forecasting](https://arxiv.org/pdf/2311.00993), [spatio-temporal boosting](https://pmc.ncbi.nlm.nih.gov/articles/PMC12173814) | Tweedie objective |
| Probabilistic gradient boosting improves CRPS and interval coverage over point models at equal accuracy | [probabilistic GBDT for claims](https://pith.science/paper/2412.14916) | uncertainty that adapts to the forecast level |

No study found combines all of these elements for the DRC; the design below
combines them and the evaluation tests them together and one at a time.

## 2. What changed

| | 0.3 | 0.4 |
|---|---|---|
| Uncertainty | one pool of absolute errors per horizon, shared by all zones | errors on the log scale, per horizon and per forecast level (3 classes, falls back to 1 when thin) |
| Climate | lags 1 to 12 weeks, means over 4, 8, 12 weeks | lags 0 to 12 weeks, means over 4, 8, 12, 16 weeks, 4-week anomalies against the zone's history |
| Case dynamics | lags, rolling means and spread, 1-to-4-week change | plus log growth over 1 and 4 weeks and current incidence |
| Space | none | 5 nearest neighbours from the CHAP polygons: their incidence, 4-week level, growth, and the pressure they put on the zone |
| Objective | Poisson | Tweedie (power 1.5), Poisson as an option |

Why the uncertainty change matters most for CHAP's metrics: with one absolute
error distribution, a zone forecast at 5 cases and one at 2,000 got the same
margin. In a controlled test (unit test
`test_log_scale_intervals_keep_their_coverage_for_large_counts`), the 80 %
interval for a zone at 2,000 cases covered about 80 % of new observations on
the log scale, and under 40 % with the old method.

All features are causal: they use data up to the forecast origin only, and the
neighbour features use the neighbours' data up to the same week (unit tests
`test_neighbour_features_only_use_the_neighbours_past`,
`test_future_mutation_cannot_change_past_features`). The neighbour graph is
stored with the trained model, so prediction does not need the polygons.

## 3. How it was evaluated

`chap-disease-compare` (module `evaluation/comparison.py`) replays the
service's logic for each variant, out of sample:

1. 4 outer rolling origins near the end of the series; for each one, only data
   up to the origin is used;
2. an inner rolling-origin backtest (3 splits, from week 104) on that data gives
   the errors behind the predictive samples, exactly as the service does;
3. the model is trained on that data and forecasts the next 3 weeks with 200
   samples per row;
4. the forecasts are scored against the observed weeks with MAE, RMSE, CRPS,
   80 % and 50 % interval coverage, and the 80 % interval score.

Data: synthetic, 100 zones on a grid, 196 weeks, XGBoost, 120 trees.

- *Malaria-like*: seasonal, rainfall and temperature lagged 2 to 3 weeks,
  moderate overdispersion, independent zones.
- *Cholera-like*: 20 % endemic hotspots, about 59 % zero weeks, heavy-tailed
  peaks (up to ~300 cases a week), outbreaks triggered by recent rainfall and
  spreading to the four grid neighbours, strong overdispersion.

These processes exist to exercise the software. They are not evidence about
real malaria or cholera, and the gains on real data may be smaller or larger.

## 4. Results

Each row adds one change to the row above (seed 20261025). Lower is better,
except coverage (targets 0.80 and 0.50).

**Cholera-like**

| Variant | MAE | CRPS | Coverage 80 % | Coverage 50 % | Interval score 80 % |
|---|---|---|---|---|---|
| 0.3 model | 2.449 | 1.835 | 0.763 | 0.490 | 15.28 |
| + log-scale uncertainty | 2.449 | 1.669 | 0.769 | 0.503 | 12.97 |
| + dynamics and climate | 2.463 | 1.661 | 0.768 | 0.511 | 12.72 |
| + neighbours | 2.407 | 1.647 | 0.774 | 0.523 | 12.57 |
| + Tweedie (0.4) | **2.311** | **1.641** | 0.772 | 0.506 | **12.19** |

**Malaria-like**

| Variant | MAE | CRPS | Coverage 80 % | Coverage 50 % | Interval score 80 % |
|---|---|---|---|---|---|
| 0.3 model | 4.751 | 3.472 | 0.776 | 0.483 | 23.57 |
| + log-scale uncertainty | 4.751 | 3.429 | 0.787 | 0.463 | 22.15 |
| + dynamics and climate | 4.739 | 3.436 | 0.792 | 0.466 | 22.37 |
| + neighbours | 4.740 | 3.421 | 0.793 | 0.471 | 22.19 |
| + Tweedie (0.4) | **4.701** | **3.408** | 0.796 | 0.471 | **22.24** |

**Second seed (4242), 0.3 against 0.4**

| | MAE | CRPS | Interval score 80 % |
|---|---|---|---|
| Cholera-like | 1.388 → 1.256 (-9.5 %) | 0.956 → 0.856 (-10.5 %) | 7.73 → 6.26 (-19 %) |
| Malaria-like | 5.338 → 5.276 (-1.2 %) | 3.982 → 3.905 (-1.9 %) | 28.33 → 25.62 (-9.6 %) |

Reading:

- The log-scale uncertainty gives the largest single gain in CRPS and interval
  score, without changing the point forecast.
- Neighbours help where spread exists (cholera-like) and do no harm where it
  does not (malaria-like zones are independent by construction).
- Tweedie improved MAE and interval score in all four runs and CRPS in three of
  four (cholera seed 4242: 0.856 against 0.839 with Poisson), hence the default.
- Gains are larger for cholera, whose zeros, peaks and spread the 0.3 model did
  not represent.

Reproduce:

```bash
uv run chap-disease-compare --disease cholera --zones 100 --weeks 196
uv run chap-disease-compare --disease malaria --zones 100 --weeks 196
```

## 5. Limits

- Synthetic data only. The decisive test is a CHAP backtest (7 splits, horizon
  3) on real malaria and real cholera data, compared with the 0.3 model and the
  current reference models.
- Intervals are better shaped but still empirical, not calibrated; check
  coverage per province in CHAP.
- Neighbours come from centroid distance; zones linked by rivers or roads but
  far apart are not neighbours.
- A dataset without polygons gets no neighbour features; everything else still
  works.

## 6. Roadmap

Ordered by expected value for DRC surveillance.

1. **Real-data validation per disease.** CHAP backtests for malaria and cholera
   at 10, 100 and 519 zones; per-province coverage; comparison with 0.3 and
   reference models; decision on `assessed_status`.
2. **Outbreak alert probabilities.** From the samples, the probability that
   next week's cases exceed an epidemic threshold per zone, for CHAP's alert
   policies (present in Core 2.4.0). Most useful for cholera.
3. **Calibrated intervals.** Rolling conformal recalibration of the log-scale
   errors per horizon and level, for coverage guarantees under drift.
4. **Weather forecasts.** CHAP can pass future climate from a weather provider;
   using it for horizons beyond 2 weeks, only when supplied, without leakage.
5. **Disease-specific tuning.** Limited, time-series-validated search of the
   Tweedie power, tree depth and learning rate per disease, stored as named
   configurations (for example `cholera` and `malaria`).
6. **Hierarchy.** Province (`parent`) aggregates as features and reconciled
   forecasts that sum to provincial totals.
7. **Reporting delay and completeness.** Nowcasting of the last weeks when DHIS2
   completeness is low.
8. **More covariates.** Humidity, vegetation index, water and sanitation
   coverage, through CHAP's additional continuous covariates.
9. **Speed.** Feature caching across the selection backtest and parallel
   engine training, to cut national training time.
