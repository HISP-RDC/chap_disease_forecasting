# Rapport d'implémentation — CHAP Disease Forecasting RDC (paludisme, choléra)

État au 2026-10-25, version `0.2.0`. Chaque affirmation de succès ci-dessous correspond à
une commande réellement exécutée (journal détaillé : `WORKLOG.md`). Ce qui n'a pas été
exécuté est signalé comme tel.

## 1. Architecture réalisée

Un service CHAPKit Python autonome (`chap-disease-forecasting`), une image, un template CHAP,
trois configured models (`xgboost`, `lightgbm`, `auto`) via l'option utilisateur `engine`.

- Bibliothèque métier sans dépendance CHAPKit : `data` (validation, semaines ISO),
  `features` (variables causales par zone), `models` (XGBoost, LightGBM, naïfs, sélection et
  fallback), `forecasting` (prévision directe 1–3 semaines, échantillonneur de résidus),
  `evaluation` (backtest rolling-origin, métriques, benchmark), `artifacts` (bundles natifs
  vérifiés SHA-256), `synthetic`.
- Adaptateur unique `chapkit_adapter/service.py` : `on_train` / `on_predict` /
  `on_validate_*`, calcul lourd hors boucle asyncio, descripteur d'entraînement autonome.
- Déploiement : entrée marketplace pour `chap-admin` (standard CHAP 2.4.0) ou overlay Compose
  `compose.chap-disease-forecasting.yml`.

## 2. Versions vérifiées

| Élément | Version | Preuve |
|---|---|---|
| CHAP Core cible | 2.4.0 (commit `7c95b2a7…`) | sources taguées lues |
| ChapKit client de Core | 2.1.0 (lock Core) | `uv.lock` Core |
| ChapKit / ServiceKit du service | 2.4.1 / 3.0.1 | `uv.lock`, code installé lu |
| Python | 3.13.13 local, 3.13 image | `uv run python -V` |
| XGBoost / LightGBM | 3.4.1 / 4.7.0 | imports |
| Image de base | `chapkit-py@sha256:0cd473d5…` | `docker buildx imagetools inspect` |

Détail et liens vers les lignes de code : `docs/compatibility-matrix.md`.

## 3. Choix techniques et justifications

- **Prévision directe** (un estimateur par horizon) : pas de réinjection de prédictions.
- **Poisson** (`count:poisson`, `poisson`) : cible de comptage non négative; la surdispersion
  est reflétée par les résidus empiriques, pas par une hypothèse paramétrique.
- **Climat retardé uniquement** (1–12 semaines) : aucune dépendance à des prévisions
  météorologiques; les valeurs futures envoyées par CHAP sont acceptées mais ignorées.
- **Sélection `auto`** sur backtest de l'historique d'entraînement uniquement, seuil
  d'observations, hystérésis; champion retenu et tracé si preuves insuffisantes.
- **Incertitude** : résidus hors échantillon par horizon; `sample_0` seul si résidus
  insuffisants; jamais présentée comme calibrée.
- **Descripteur autonome** (session 2) : ChapKit copie le cwd dans chaque workspace et pickle
  le résultat; embarquer les fichiers natifs vérifiés évite la croissance des artefacts et
  fait de la base SQLite l'unique état.
- **`asyncio.to_thread`** (session 2) : ChapKit attend les callbacks sur la boucle qui sert
  `/health` et le keepalive (TTL Core 30 s).
- **`chap-admin`** (session 2) : chemin documenté par CHAP 2.4.0; il vérifie identité et
  commit, crée les configured models et sait annuler un update raté.

## 4. Structure du dépôt

Voir README, section « Develop and release ». Fichiers de déploiement : `Dockerfile`,
`deploy/` (start, healthcheck, env, `pin_release.py`), `marketplace/`,
`compose.chap-disease-forecasting.yml`, `.github/workflows/ci.yml`.

## 5. Commandes exactes exécutées (session 2)

```bash
export UV_CACHE_DIR=/private/tmp/chap-disease-uv-cache OMP_NUM_THREADS=1
uv lock --check
uv run --frozen ruff format --check . && uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest -q
uv export --frozen --no-dev --no-emit-project --format requirements-txt --output-file req.txt
uv run --frozen pip-audit --require-hashes --disable-pip -r req.txt
docker compose -p chapcheck -f compose.ghcr.yml -f compose.chap-disease-forecasting.yml config --quiet   # fichiers chap-core v2.4.0
docker compose -p chapcheck -f compose.yml -f compose.chap-disease-forecasting.yml config --quiet
OMP_NUM_THREADS=4 uv run chap-disease-benchmark --zones 514 --engine {xgboost,lightgbm,auto} \
  --through-service --n-estimators 120 --threads 4
uv run python deploy/pin_release.py --entry <copie> --commit 0123456789abcdef0123456789abcdef01234567
```

## 6. Tests réussis et échoués

- **43 réussis, 0 échoué** (`pytest -q`, 10 s). Ruff, format, mypy strict : OK.
  `pip-audit` : « No known vulnerabilities found ». Compose fusionné : OK.
- Échecs rencontrés puis corrigés pendant la session : `auto` plantait sans preuve
  historique (défaut réel, corrigé); attente de test erronée sur le nombre d'échantillons
  (2 zones → 4 résidus < 10 : comportement correct, test ajusté).

## 7. Mesures CPU / RAM / disque

macOS arm64, données synthétiques, via les callbacks du service (`--through-service`) :

| Zones × semaines | Configuration | Train | Predict | Pic RSS | Artefact picklé |
|---|---|---|---|---|---|
| 10 × 52 | auto, 120 arbres, 1 thread | 3,1 s | 0,09 s | 252 Mio | 2,1 Mio |
| 514 × 196 | xgboost, 120 arbres, 4 threads | 40,1 s | 3,9 s | 525 Mio | 1,5 Mio |
| 514 × 196 | lightgbm, 120 arbres, 4 threads | 42,9 s | 4,0 s | 477 Mio | 1,3 Mio |
| 514 × 196 | auto, 120 arbres, 4 threads | 82,4 s | 4,1 s | 519 Mio | 2,7 Mio |

Mesures de session 1 (pipeline seul) : features 514 × 196 en 3,4 s, 169 Mio. Disque local
libre : 12 Gio. Aucune mesure sur le VPS.

## 8. Couverture des 514 zones

Simulation 514 × 196 (100 744 lignes, 89 variables) **exécutée** pour les trois
configurations : 1 542 prédictions (514 × 3), 100 échantillons, aucune erreur
d'entraînement. Aucune limite fonctionnelle sur le nombre de zones dans le code. Le palier
1 000 zones n'a pas été exécuté.

## 9. Validation CHAPKit

- Contrat HTTP complet rejoué comme Core 2.4.0, pour les trois configurations, y compris
  prédiction après redémarrage sur la même base (`test_core_style_train_predict_and_restart`).
- Schéma de configuration, `/health`, `/api/v1/info`, `git_revision` depuis `GIT_REVISION`.
- Compatibilité fil 2.4.1 ↔ 2.1.0 vérifiée par inspection des deux tags.
- Règles `chap-admin` reproduites par `test_marketplace_entry.py`.
- **Non exécuté** : Core 2.4.0 réel, `chap-admin` réel, DHIS2 Modeling App.

## 10. État de la construction Docker

Dockerfile multi-arch sur base épinglée par digest, non-root, healthcheck, labels OCI,
`GIT_REVISION` obligatoire (40 hex). **Build non exécuté localement** (disque limité, image
non nécessaire aux tests). La CI construit `linux/amd64`, fait un smoke test sans réseau en
lecture seule, puis publie `sha-<7>`/`sha-<40>` avec SBOM et provenance. La CI n'a pas encore
tourné : aucun dépôt Git n'est initialisé.

## 11. Préparation du déploiement VPS

`docs/PRODUCTION_DEPLOYMENT.md` : diagnostics en lecture seule, chemins A (`chap-admin`) et B
(overlay), création des configured models, vérification, acceptation 10/100/514, mise à jour.
`docs/OPERATIONS.md` : métriques, sauvegarde/restauration de la base, incidents, sécurité.
Aucune commande n'a été exécutée sur le VPS.

## 12. Procédure de rollback

Chemin A : rollback automatique de `chap-admin` si le nouveau service ne s'enregistre pas;
sinon remettre `channels.stable` sur la version précédente et `chap-admin update`. Chemin B :
remettre l'image précédente et `up -d --no-deps`. Restaurer image et base ensemble si le
format de descripteur change. Jamais `down -v` ni suppression de volume.

## 13. Risques et limitations

- Aucune validation scientifique sur données réelles; statut auto-évalué orange.
- Incertitude non calibrée; à contrôler par couverture/CRPS dans les backtests CHAP.
- Historique contigu par zone exigé; zones absentes de l'historique non prévisibles.
- Comparaison EWARS non faite : aucun résultat EWARS compatible disponible, aucun chiffre
  inventé.
- API du service sans authentification : l'isolation réseau est le contrôle.
- ChapKit pickle l'artefact d'entraînement (limite de la bibliothèque).
- Un backtest CHAP à 7 splits sur 514 zones en `auto` est estimé à ~10 min (7 × train +
  predict), à confirmer sur le VPS.
- `chap-admin`, Core et la Modeling App n'ont pas été exercés en réel.

## 14. Traçabilité exigences → tests → preuves

| Exigence (prompt) | Test(s) | Preuve |
|---|---|---|
| §6 semaines ISO, chevauchement d'année | `test_iso_week_crosses_year_correctly` | pytest vert |
| §6 zéro ≠ manquant, jamais imputé à 0 | `test_validation_preserves_zero_and_missing_cases` | pytest vert |
| §6 valeurs impossibles, doublons, continuité | `test_validation_rejects_impossible_values`, `test_validation_rejects_duplicate_and_missing_week` | pytest vert |
| §6 zones inconnues | `test_unknown_location_policy_is_explicit` | pytest vert |
| §7 anti-fuite temporelle | `test_future_mutation_cannot_change_past_features`, `test_rolling_origin_backtest_uses_only_past_training_data` | pytest vert |
| §7 séparation des zones | `test_zone_mutation_does_not_leak_between_locations`, `test_required_features_exist_and_are_per_zone` | pytest vert |
| §8 trois horizons directs | `test_real_engine_trains_three_horizons_and_is_deterministic`, `test_naive_models_produce_three_non_negative_horizons` | pytest vert |
| §9 XGBoost et LightGBM réels, déterminisme | `test_real_engine_trains_three_horizons_and_is_deterministic` | pytest vert |
| §9 sérialisation sûre, falsification détectée | `test_native_artifact_roundtrip_and_tamper_detection`, `test_in_memory_bundle_roundtrip_and_rejections` | pytest vert |
| §10 champion/challenger, sélection sans futur | `test_challenger_requires_promotion_margin`, `test_auto_selection_ignores_future_evidence_and_records_reason` | pytest vert |
| §10 ensemble pondéré hors échantillon | `test_weighted_ensemble_uses_historical_weights` | pytest vert |
| §10 fallback tracé, échec structuré | `test_fallback_is_traceable_and_total_failure_is_structured`, `test_auto_engine_retains_champion_when_evidence_is_insufficient` | pytest vert |
| §11 échantillons honnêtes | `test_residual_samples_are_reproducible_non_identical_and_labelled_honestly`, `test_in_sample_or_insufficient_residuals_are_rejected` | pytest vert |
| §12 métriques MAE/RMSE/MAPE/CRPS/couverture | `test_point_and_probabilistic_metrics` | pytest vert |
| §14 générateur synthétique et paliers | `test_synthetic_generator_is_reproducible_and_has_required_columns`, `test_required_scale_presets` | pytest vert |
| §15 performances | `test_ten_zone_feature_benchmark_stays_bounded`; benchmark 514 | section 7 |
| §16 contrat CHAPKit | `test_service_contract_exposes_health_info_and_config_schema`, `test_callbacks_train_real_model_and_return_chap_samples`, `test_core_style_train_predict_and_restart` | pytest vert |
| §16 persistance et rechargement | `test_core_style_train_predict_and_restart`, `test_predict_rejects_descriptor_from_previous_format` | pytest vert |
| §16 trois modèles sélectionnables, IDs stables | `test_identity_matches_the_running_service`, `test_each_configuration_is_stored_unchanged_by_the_service` | pytest vert |
| §16 enregistrement durable (keepalive) | `test_training_does_not_block_the_event_loop` | pytest vert |
| §17 image versionnée, release immuable | `test_pin_release_produces_a_pin_chap_admin_accepts`, `test_service_version_matches_the_package_version` | pytest vert; build CI non exécuté |
| §18 Compose compatible, pas de port, réseau CHAP | — | `docker compose config` fusionné v2.4.0 OK |
| §22 dépendances auditées | — | `pip-audit` : aucune vulnérabilité |
| §20 production, rollback | — | guides écrits, non exécutés |

La réussite des tests locaux ne vaut ni validation scientifique ni autorisation de
déploiement national.

## Addendum — version 0.4.0 (paludisme et choléra)

- **Renommage** : `chap-malaria-forecasting` → `chap-disease-forecasting` (service, paquet
  Python `chap_disease_forecasting`, image, overlay, entrée marketplace
  `chap_disease_forecasting`, CLI `chap-disease-benchmark` / `chap-disease-compare`). Nom
  affiché : « Weekly Disease Forecast (Gradient Boosting) ». Nouveau modèle dans CHAP;
  migration décrite dans `PRODUCTION_DEPLOYMENT.md`, section 8.
- **Améliorations** (détail, sources et résultats : `docs/MODEL_IMPROVEMENTS.md`) :
  incertitude en échelle log par horizon et par niveau; variables climatiques lag 0,
  fenêtre 16 semaines et anomalies; croissance et incidence courante; voisinage à partir des
  polygones envoyés par CHAP (graphe stocké dans le modèle); objectif Tweedie par défaut.
  Format de descripteur 3.
- **Preuves** : banc `chap-disease-compare` hors échantillon (100 zones × 196 semaines,
  4 origines, horizon 3), deux graines. Choléra synthétique : CRPS −10,6 % / −10,5 %, MAE
  −5,6 % / −9,5 %, score d'intervalle −20 % / −19 %. Paludisme synthétique : CRPS −1,8 % /
  −1,9 %, MAE −1,1 % / −1,2 %, score d'intervalle −5,6 % / −9,6 %. Données synthétiques
  seulement; validation sur données réelles à faire dans CHAP.
- **Tests** : 58 réussis, 1 ignoré (aucune release épinglée pour le nouveau modèle);
  nouveaux tests de voisinage (causalité, signal de flambée voisine, persistance), de
  couverture de l'incertitude, du banc de comparaison, et de bout en bout avec polygones.
