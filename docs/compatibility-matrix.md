# Matrice de compatibilité CHAP Core / CHAPKit

Date de vérification : 2026-10-25  
Cible de référence : CHAP Core `v2.4.0`  
Périmètre : service autonome de prévision hebdomadaire XGBoost, LightGBM et sélection/ensemble.

## Verdict

L'intégration primaire à retenir est un **service CHAPKit autonome**, enregistré auprès de CHAP Core. Le tag CHAP Core `v2.4.0` (commit `7c95b2a7df62e32cc19393468ab6b111926655ce`) déclare `chapkit>=2.1.0,<3`. Son fichier `uv.lock` résout `chapkit==2.1.0`; cette version est donc la baseline exacte pour reproduire le tag, et non la seule version compatible. Le scaffold courant observé le 2026-10-25 avec le paquet officiel ChapKit a produit `chapkit>=2.4.1,<3`, ce qui reste dans la plage déclarée par Core 2.4.0, mais doit encore passer les tests contractuels contre l'instance cible.

L'architecture retenue est **un service / un template**, avec un champ de configuration `engine` et trois `ConfiguredModels` Core : `xgboost`, `lightgbm` et `auto`. Ce choix est compatible avec le contrat Core 2.4.0 à condition que le JSON Schema exposé par le service déclare bien `engine` et ses trois valeurs. L'enregistrement du service ne crée pas ces configurations automatiquement.

La version réellement déployée sur le VPS n'a pas été interrogée. Cette matrice prouve le contrat des sources officielles taguées; elle ne remplace pas la recette sur la cible.

Légende : ✅ confirmé dans le tag/code/test officiel; ⚠️ déclaré compatible mais non exécuté ici; ⛔ non établi pour cette cible.

## Versions et génération du service

| Sujet | Statut | Contrat confirmé | Décision pour ce projet |
|---|---:|---|---|
| CHAP Core | ✅ | `pyproject.toml` : version `2.4.0`, Python `>=3.13,<3.15`, ChapKit `>=2.1.0,<3` ([source](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/pyproject.toml#L1-L6), [dépendance](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/pyproject.toml#L54-L62)). | Tester et documenter contre l'image/tag exact `2.4.0`; vérifier `/system/info` sur la cible avant installation. |
| Baseline reproductible | ✅ | Le lock officiel de Core 2.4.0 résout ChapKit `2.1.0` ([uv.lock](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/uv.lock#L563-L577)). Le tag ChapKit `v2.1.0` correspond au commit `1ae98cf03f0bdc6abc6295a6f9ba4f84a8ae6744`. | `chapkit==2.1.0` sert à reproduire exactement le graphe Core; ce n'est pas une borne maximale. |
| ChapKit plus récent | ⚠️ | `2.4.1` appartient à la plage `>=2.1.0,<3`. Le scaffold courant observé produit `chapkit>=2.4.1,<3` et conserve les callbacks ci-dessous. | Acceptable seulement après la suite contractuelle Core 2.4.0. Verrouiller ensuite la version et le hash dans `uv.lock`; ne pas utiliser une image ou une dépendance flottante. |
| Python | ✅ | Core 2.4.0 impose Python 3.13 ou 3.14; ChapKit 2.1.0 impose Python `>=3.13` ([ChapKit pyproject](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/pyproject.toml#L1-L31)). | Python 3.13 est la cible commune prudente. |
| Scaffold | ✅ | `chapkit init` accepte `fn-py` et `--with-validation`, calcule l'identifiant de service et rend `main.py`, `pyproject.toml`, Dockerfile et Compose ([init.py](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/cli/init.py#L72-L159)). | Base fonctionnelle : `uvx --from chapkit==2.1.0 chapkit init <projet> --template fn-py --with-validation` pour reproduire la baseline. Le code généré est un point de départ, pas une preuve de compatibilité du modèle final. |

## Contrat fonctionnel train / predict

Le runner fonctionnel attend des fonctions `async` avec ces signatures exactes :

```python
async def on_train(config, data, geo=None) -> Any: ...


async def on_predict(config, model, historic, future, geo=None) -> DataFrame: ...


async def on_validate_train(config, data, geo=None) -> list[ValidationDiagnostic]: ...


async def on_validate_predict(config, historic, future, geo=None) -> list[ValidationDiagnostic]: ...
```

Ces signatures sont définies dans [ChapKit `runner.py`](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/ml/runner.py#L32-L44), mises en œuvre par le scaffold ([train/predict](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/cli/templates/main.py.jinja2#L49-L115)) et reprises par la fixture d'intégration de Core 2.4.0 ([fixture officielle](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/tests/fixtures/chapkit_test_model/main.py#L50-L98)).

| Opération | Route | Entrée | Réponse immédiate | Résultat final |
|---|---|---|---|---|
| Info | `GET /api/v1/info` | — | `MLServiceInfo` | Contrat/capacités du template |
| Schéma config | `GET /api/v1/configs/$schema` | — | JSON Schema | Source de `userOptions` côté Core |
| Création config | `POST /api/v1/configs` | nom + données de config | config avec ULID | ULID utilisé par `$train` |
| Train | `POST /api/v1/ml/$train` | `config_id`, `data`, `run_info`, `geo?` | `job_id`, `artifact_id`, message | artefact d'entraînement après polling du job |
| Predict | `POST /api/v1/ml/$predict` | `artifact_id`, `historic`, `future`, `run_info`, `geo?` | `job_id`, `artifact_id`, message | artefact de prédiction DataFrame |
| Validation | `POST /api/v1/ml/$validate` | union discriminée `type=train|predict` | `valid`, `diagnostics[]` | diagnostics `error|warning|info` |

Les schémas ChapKit sont documentés dans [`ml/schemas.py`](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/ml/schemas.py#L38-L168). Le wrapper Core 2.4.0 appelle les routes info/config/job/artifact et envoie `$train`/`$predict` comme ci-dessus ([wrapper, routes](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/models/chapkit_rest_api_wrapper.py#L148-L201), [payloads](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/models/chapkit_rest_api_wrapper.py#L237-L281)). Les traitements sont asynchrones : un HTTP 202 ne signifie pas que l'entraînement ou la prédiction a réussi; Core sonde le job jusqu'à un état terminal.

### DataFrames d'entrée

Sur le fil, un DataFrame est représenté par :

```json
{
  "columns": ["location", "time_period", "disease_cases"],
  "data": [["OU_1", "2026-W01", 12.0]]
}
```

Le tableau `data` est orienté lignes; `schema` est facultatif ([ChapKit DataFrame](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/data/dataframe.py#L146-L171)). Core convertit `time_period` en chaîne et les `NaN` en `null` avant l'appel ([Core `_prepare_dataframe`](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/models/chapkit_rest_api_wrapper.py#L67-L75)).

Contrat applicatif minimal à valider :

- `location` et `time_period` identifient une série et une période; `disease_cases` est la cible dans l'historique d'entraînement;
- les covariables effectivement obligatoires doivent être listées dans `MLServiceInfo.required_covariates`; les covariables libres ne sont acceptées que si `allow_free_additional_continuous_covariates=True`;
- `historic` et `future` sont des arguments distincts au predict; la sortie doit suivre strictement les lignes de `future` sans perte, duplication ni réordonnancement ambigu;
- `geo` est une `FeatureCollection` optionnelle. Déclarer `requires_geo=False` tant que le modèle ne consomme pas réellement les polygones;
- `parent` n'est pas un champ générique garanti par ChapKit. Il peut traverser le DataFrame comme colonne supplémentaire, mais sa présence et sa sémantique doivent être testées sur les datasets Core/DHIS2 cibles.

Le `run_info.prediction_periods` est le nom canonique du nombre de périodes; `prediction_length` reste accepté comme alias historique. Il ne faut pas coder l'horizon à 3 dans le runner. Le test d'intégration officiel Core vérifie qu'une demande de 5 périodes atteint le modèle ([test E2E](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/tests/integration/test_chapkit_e2e.py#L233-L275)). Pour ce projet, `MLServiceInfo` doit annoncer `period_type=weekly`, `min_prediction_periods=1`, `max_prediction_periods=3`.

### Sortie probabiliste

La sortie doit contenir les colonnes d'index de la grille future et des tirages contigus `sample_0`, `sample_1`, ..., sans trou. Un modèle déterministe peut techniquement ne produire que `sample_0` ([contrat runner](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/docs/guides/shell-runner-contract.md#L104-L114)). Core dénombre les colonnes commençant par `sample_`, relit strictement `sample_0..sample_n` et rejette les valeurs non finies ([`Samples.from_pandas`](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/datatypes.py#L321-L354)).

Exigences de ce service, plus strictes que le minimum technique :

- exactement une ligne par paire `(location, time_period)` demandée;
- toutes les valeurs finies et positives ou nulles après la politique explicite de clipping;
- plusieurs vrais tirages lorsque le mode probabiliste est annoncé; dupliquer une prédiction ponctuelle dans plusieurs colonnes satisfait le format mais ne constitue pas une incertitude valide;
- graine contrôlée et provenance enregistrée pour les tests reproductibles;
- tests de calibration/couverture et CRPS avant toute promotion du statut de maturité.

CHAPKit ne prouve ni la non-négativité, ni la calibration, ni l'absence de fuite temporelle : ce sont des responsabilités du modèle.

## Configuration, identités et trois moteurs

### `ModelConfig`

Une classe dérivée de `BaseConfig` est le contrat de configuration du service. `BaseConfig` autorise les champs supplémentaires, réserve `prediction_periods` et `additional_continuous_covariates`, et remonte les valeurs reçues dans `user_option_values` vers les champs typés ([source](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/config/schemas.py#L13-L47)). Le service doit au minimum déclarer :

```python
engine: Literal["xgboost", "lightgbm", "auto"]
```

et typer/borner tous les hyperparamètres exposés. `auto` doit représenter la logique de sélection/ensemble documentée, pas un quatrième moteur implicite.

### Une identité de service, trois `ConfiguredModels`

Les identifiants ne doivent pas être confondus :

| Identité | Type / origine | Stabilité attendue |
|---|---|---|
| `MLServiceInfo.id` | slug texte du service | stable entre déploiements; ici `chap-disease-forecasting` |
| `ModelTemplate.id` | entier attribué par la base Core | dynamique; toujours le lire par API |
| nom du `ConfiguredModel` | `<template>:<stub>` calculé par Core | stable si le template et le stub restent stables |
| `ConfiguredModel.id` | entier attribué par Core | dynamique |
| `config_id` ChapKit | ULID du service | opaque; géré par Core/le service |
| `artifact_id` ChapKit | ULID du service | opaque; l'artefact train alimente predict |

L'enregistrement crée/synchronise le template, **pas** les `ConfiguredModels`; le test officiel l'affirme explicitement ([test self-registration](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/tests/integration/rest_api/test_chapkit_self_registration.py#L118-L127)). Le endpoint de création est `POST /v1/crud/configured-models`; son modèle contient exactement `name`, `modelTemplateId`, `userOptionValues` et `additionalContinuousCovariates` ([schéma](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/data_models.py#L381-L393), [route](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/v1/routers/crud.py#L966-L1000)).

Procédure sans ID inventé :

1. faire `GET /v1/crud/model-templates` avec le bearer token Core si activé;
2. sélectionner le template dont `name`, `version`, `usesChapkit` et `sourceDigest` correspondent au service actif;
3. reprendre son `id` dans les trois POST suivants;
4. relire `GET /v1/crud/configured-models` et conserver les IDs retournés.

```json
{
  "name": "xgboost",
  "modelTemplateId": 123,
  "userOptionValues": {"engine": "xgboost"},
  "additionalContinuousCovariates": []
}
```

Répéter avec les stubs `lightgbm` et `auto`. La valeur `123` est uniquement illustrative et ne doit jamais être copiée dans un script. Core préfixe un stub non `default` avec le nom du template ([calcul du nom](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/database/database.py#L238-L268)); il ne faut donc pas envoyer le nom complet souhaité comme stub. Core valide les valeurs contre `userOptions` avec `additionalProperties: false` ([validation](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/database/model_templates_and_config_tables.py#L275-L286)) : tant que `engine` n'apparaît pas dans le template lu par GET, ces POST ne sont pas valides.

## Artefacts, persistance et sécurité

| Point | Statut | Constat | Exigence |
|---|---:|---|---|
| Artefact train | ✅ | Le `FunctionalModelRunner` sérialise le modèle dans `model.pickle`, puis conserve le workspace ([source](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/ml/runner.py#L406-L435)). | L'objet retourné par `on_train` doit être pickleable avec ce runner. |
| Artefact predict | ✅ | Le runner recrée un workspace, écrit config, entrées, modèle et `predictions.csv` ([source](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/ml/runner.py#L437-L472)). En 2.4.1, l'`artifact_id` renvoyé contient un `ml_prediction` dont `content` est le DataFrame; le zip n'est qu'un artefact de debug. | Garder le stockage ChapKit persistant pour pouvoir prédire et auditer après redémarrage. |
| Copie du cwd | ✅ (2.4.1) | `prepare_workspace(Path.cwd(), ...)` copie tout le répertoire de travail (`/work`) dans chaque workspace zippé, en excluant `*.db`, `.venv`, caches. | Ne rien écrire sous le cwd hors de la base : sinon chaque train/predict recopie les fichiers précédents dans SQLite. |
| Base ChapKit | ✅ | Le scaffold utilise `DATABASE_URL`, par défaut `sqlite+aiosqlite:///data/chapkit.db` ([template](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/cli/templates/main.py.jinja2#L238-L258)). `chap-admin` monte le volume sur `<WORKDIR>/data`. | Un volume distinct et sauvegardé pour la base du service; ne pas la confondre avec PostgreSQL Core. |
| Sérialisations natives | ✅ (implémenté) | Le runner pickle l'objet englobant; ce service lui renvoie un descripteur `dict` (format 2) qui embarque les fichiers natifs XGBoost JSON / LightGBM texte avec SHA-256 (`artifacts.pack_forecaster`). | La base SQLite est l'unique état; un ancien format est refusé avec un message « retrain ». |
| Exécution des callbacks | ✅ (2.4.1) | `MLManager._train_task` fait `await runner.on_train(...)` sur la boucle asyncio qui sert `/health` et le keepalive ServiceKit (tâche asyncio, TTL Core 30 s). Ordonnanceur ChapKit : `max_concurrency=1` par défaut. | Le calcul lourd est exécuté via `asyncio.to_thread` (test `test_training_does_not_block_the_event_loop`). |
| Confiance | ✅ | `pickle` peut exécuter du code à la désérialisation. | Base/volume privés, aucun upload d'artefact non fiable, image/version de classes verrouillées, tests de restauration avant upgrade/rollback. |

La compatibilité binaire des artefacts entre deux versions de l'image n'est pas garantie par CHAPKit. Un rollback doit restaurer ensemble l'image, le lock de dépendances et, si nécessaire, la sauvegarde de la base ChapKit.

## Health, validation, métriques et démarrage

`MLServiceBuilder` ajoute automatiquement health, system, configs, artefacts, jobs et ML ([builder](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/api/service_builder.py#L426-L474)). Le scaffold ajoute `.with_monitoring().with_registration()` ([template](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/cli/templates/main.py.jinja2#L246-L258)).

Commande et healthcheck officiels du Dockerfile scaffoldé :

```text
uvicorn main:app --host 0.0.0.0 --port 8000
curl --fail http://localhost:8000/health
```

Voir le [Dockerfile template](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/src/chapkit/cli/templates/Dockerfile.jinja2#L67-L75). `/metrics` vient de `.with_monitoring()`; sa collecte par l'infrastructure cible n'est pas automatique et n'a pas été vérifiée.

Le endpoint `$validate` et ses hooks sont confirmés dans ChapKit 2.1.0, mais aucune invocation de `/api/v1/ml/$validate` par Core 2.4.0 n'a été trouvée dans le tag inspecté. La recette doit donc :

- tester `$validate` directement;
- refaire les validations critiques dans `on_train` et `on_predict` afin qu'elles ne puissent pas être contournées;
- vérifier les colonnes, doublons, fréquence hebdomadaire, horizon 1–3, historique minimal, valeurs finies, moteurs et covariables;
- renvoyer des codes de diagnostic stables et actionnables, sans données sensibles.

## Enregistrement, authentification et visibilité Modeling App

### Variables d'environnement officielles

| Variable | Rôle | Statut |
|---|---|---:|
| `SERVICEKIT_ORCHESTRATOR_URL` | active l'auto-registration; valeur Compose typique `http://chap:8000/v2/services/$$register` | ✅ |
| `SERVICEKIT_REGISTRATION_KEY` | secret commun envoyé par ServiceKit en `X-Service-Key`, si Core l'exige | ✅ |
| `SERVICEKIT_HOST`, `SERVICEKIT_PORT` | adresse annoncée à Core; optionnels, port par défaut 8000 | ✅ |
| `DATABASE_URL` | SQLite persistante du service | ✅ |
| `GIT_REVISION` | commit de build reporté par `/api/v1/info` | ✅ |
| `CHAP_API_TOKEN` | bearer token général de Core, distinct du secret d'enregistrement | ✅ Core |

Les trois variables ServiceKit sont documentées par le [guide de déploiement ChapKit](https://github.com/dhis2-chap/chapkit/blob/v2.1.0/docs/guides/deploying-to-chap-core.md#L45-L63). Core accepte le bearer token sur toutes les routes protégées et limite la clé d'enregistrement à `/v2/services`; `/health`, `/health/ready` et `/system/info` restent publics ([middleware d'authentification](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/auth.py#L23-L43), [règles](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/auth.py#L86-L131)). Ne jamais injecter `CHAP_API_TOKEN` dans le conteneur modèle si seule la clé d'enregistrement est nécessaire.

L'orchestrateur conserve une inscription 30 secondes et ServiceKit doit la maintenir par ping ([orchestrateur](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/services/orchestrator.py#L28-L59), [register/ping](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/services/orchestrator.py#L75-L181)). Un conteneur sain mais non visible dans `GET /v2/services` indique donc un problème d'URL, de réseau, de secret ou de keepalive.

### Chaîne de visibilité à vérifier

1. `GET <service>/health` renvoie HTTP 200 et un statut sain.
2. `GET <service>/api/v1/info` contient le bon `id`, la bonne version, `period_type=weekly`, les bornes 1–3, le commit et le schéma de capacités.
3. `GET <core>/v2/services` contient le service avec son URL joignable depuis Core.
4. `GET <core>/v1/crud/model-templates` contient le template ChapKit actif, sans `revision_mismatch`.
5. Les trois `ConfiguredModels` sont créés avec l'ID de template lu à l'étape 4.
6. `GET <core>/v1/crud/configured-models` retourne les trois entrées non archivées et `healthStatus=live`; cette route alimente les model pickers ([route Core](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/rest_api/v1/routers/crud.py#L912-L938)).
7. La Modeling App cible, authentifiée vers ce Core, affiche les trois variantes et peut lancer au moins un backtest puis une prédiction par variante.

Core refuse de considérer comme identique une révision différente publiée sous la même version. Chaque modification de modèle doit embarquer un `GIT_REVISION` de 40 caractères et incrémenter `MLServiceInfo.version`. La visibilité réelle dans la version de Modeling App installée n'est pas prouvée par la seule inspection de Core : l'étape 7 reste un test d'acceptation obligatoire.

## Compatibilité fil ChapKit 2.4.1 (service) ↔ 2.1.0 (client Core 2.4.0)

Core 2.4.0 relit la prédiction par `GET /api/v1/artifacts/{id}`, valide
`MLPredictionArtifactData` puis construit `chapkit.data.DataFrame(**content)` avec sa propre
ChapKit 2.1.0 ([wrapper](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/models/chapkit_rest_api_wrapper.py#L197-L201)).
ChapKit 2.4.1 ajoute un `schema` au DataFrame (`with_schema()`); le `DataFrame` 2.1.0 déclare
`table_schema` avec l'alias `schema` et `BaseArtifactData` 2.1.0 (`extra=forbid`) attend les
mêmes clés `type/metadata/content/content_type/content_size`. ✅ par inspection des deux tags;
le test `tests/integration/test_chapkit_http_lifecycle.py` rejoue les payloads de Core 2.4.0
(options à plat dans `data`, `time_period` `YYYY-Wnn`, `NaN` → `null`, polling jusqu'à
`completed|failed|canceled`) contre ChapKit 2.4.1. Core vérifie aussi que le service stocke
exactement les options envoyées (`_check_stored_config`); le test le vérifie.

## Déploiement standard : `chap-admin` et marketplace (Core 2.4.0)

Sources : [`cli_endpoints/marketplace.py`](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/cli_endpoints/marketplace.py),
[`services/model_marketplace.py`](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/chap_core/services/model_marketplace.py),
[docs « Running your own model »](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/docs/modeling-app/running-your-own-model.md).

| Règle `chap-admin` | Statut | Ce service |
|---|---:|---|
| Registre `schema_version: 2`, URL surchargeable par `CHAP_MARKETPLACE_URL` | ✅ | `marketplace/registry.yaml` |
| `id` `[a-z0-9][a-z0-9_]*`, `service_id == id.replace("_", "-")` | ✅ | `chap_disease_forecasting` / `chap-disease-forecasting` = `MLServiceInfo.id` |
| Seul `channels.stable` avec `status: verified` est installé | ✅ | épinglé par `deploy/pin_release.py` après CI + recette |
| `image_tag` = `sha-<préfixe ≥ 7 du commit>` | ✅ | CI publie `sha-<7>` et `sha-<40>` |
| Le service enregistré doit annoncer `git_revision == commit` | ✅ | `GIT_REVISION` (build arg CI) lu par ChapKit |
| Un configured model par entrée `configurations` (clés réservées retirées) | ✅ | `xgboost`, `lightgbm`, `auto` |
| Registre ≠ officiel → `--accept-risk` obligatoire | ✅ | documenté |
| Service `marketplace-<id avec ->`, volume `<service>-data:<WORKDIR>/data`, tmpfs `/tmp`, réseau `default` | ✅ | `WORKDIR /work`, utilisateur `chapkit` |
| Update raté → service précédent restauré, enregistrement annulé | ✅ | rollback automatique |

`tests/contracts/test_marketplace_entry.py` reproduit ces règles (identité, épinglage,
stockage inchangé des options, cohérence `MLServiceInfo`). `chap-admin` lui-même n'a pas été
exécuté (pas de stack CHAP locale).

Réseaux Compose 2.4.0 : `chap` et `worker` sur `default` + `backend`, PostgreSQL et Valkey sur
`backend` seulement; les services modèles n'ont pas de clé `networks` et restent sur
`default` ([compose.ghcr.yml](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/compose.ghcr.yml)).
`compose.worker.yml` n'existe pas dans le tag : c'est un fichier local du VPS.

## MLproject historique

Trois chemins existent dans l'écosystème officiel : ancien MLproject exécuté dans Core/worker, MLproject servi par ChapKit, et service ChapKit direct. Le premier couple le modèle à l'environnement Core et ne respecte pas la décision d'isolation; le troisième est donc le chemin principal ici.

ChapKit 2.1.0 contient des commandes `chapkit mlproject run` et `migrate`, mais la [documentation officielle courante du tutoriel MLproject](https://chap.dhis2.org/chap-modeling-platform/external_models/mlproject_tutorial/) demande **CHAP Core 2.5.0 ou plus récent** pour le flux MLproject servi par ChapKit. Aucun test E2E officiel Core 2.4.0 de ce flux n'a été identifié pendant cette recherche. Il est donc ⛔ non accepté comme solution de repli sur une cible 2.4.0 sans essai contractuel dédié.

Les anciens entry points MLproject utilisent notamment `train_data`, `historic_data`, `future_data`, `model`, `out_file`, avec `model_config` et `polygons` optionnels; la sortie reste une table avec `sample_0...`. Ce contrat est utile pour migrer un modèle existant, mais il ne doit pas contaminer l'API interne du nouveau service.

## Tests d'acceptation obligatoires

Avant de qualifier la combinaison ChapKit choisie avec Core 2.4.0 :

- relever la version/image/digest Core cible et sauvegarder `GET /system/info`;
- figer Python, ChapKit, ServiceKit et les bibliothèques ML dans le lock et l'image;
- valider `/health`, `/api/v1/info`, `/api/v1/configs/$schema` et `/metrics`;
- valider directement `$validate` sur payload sain et sur chaque erreur critique;
- exécuter train → polling job → lecture artefact → predict pour `engine=xgboost`, `lightgbm`, `auto` et horizons 1, 2, 3;
- vérifier la grille de sortie, la continuité `sample_0..n`, valeurs finies/non négatives et reproductibilité;
- redémarrer le service et prouver qu'un artefact train antérieur reste prédictible;
- tester une révision/version identique puis un bump de version, l'expiration d'enregistrement et le retour automatique;
- créer les trois `ConfiguredModels` à partir de l'ID lu, puis lancer backtest/prédiction depuis la Modeling App;
- tester sauvegarde/restauration et rollback image + base ChapKit;
- exécuter les tests d'intégration sur l'architecture CPU de production, en particulier LightGBM/XGBoost.

## Limites et points non vérifiés

- Aucun accès au VPS, à DHIS2, à la Modeling App ou aux secrets de production n'a été effectué.
- Les tests des dépôts officiels ont été inspectés, mais n'ont pas été exécutés dans cette recherche; aucune stack Core n'a été installée.
- La plage `chapkit>=2.1.0,<3` est une déclaration de compatibilité de Core 2.4.0. Elle ne garantit pas qu'une future 2.x soit exempte de régression pour ce service; le lock et la recette restent nécessaires.
- La présence des covariables futures (météo, population), de `parent`, la profondeur d'historique et les conventions DHIS2 doivent être constatées sur les vrais datasets.
- La validation scientifique, les seuils EWARS, la qualité des intervalles et la supériorité de `auto` ne sont pas prouvés par le contrat logiciel.
- Les routes `/v2/services` sont encore présentées comme expérimentales dans la [documentation officielle d'enregistrement](https://chap.dhis2.org/chap-modeling-platform/webapi/service-registration/); conserver des tests de non-régression à chaque upgrade.

## Références officielles principales

- [CHAP Core v2.4.0 — source taguée](https://github.com/dhis2-chap/chap-core/tree/v2.4.0)
- [ChapKit v2.1.0 — source taguée](https://github.com/dhis2-chap/chapkit/tree/v2.1.0)
- [Test E2E Core ↔ service ChapKit réel](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/tests/integration/test_chapkit_e2e.py)
- [Tests Core d'auto-registration](https://github.com/dhis2-chap/chap-core/blob/v2.4.0/tests/integration/rest_api/test_chapkit_self_registration.py)
- [Guide officiel : exécuter son propre modèle](https://chap.dhis2.org/chap-modeling-platform/modeling-app/running-your-own-model/)
- [Guide officiel : service registration](https://chap.dhis2.org/chap-modeling-platform/webapi/service-registration/)
