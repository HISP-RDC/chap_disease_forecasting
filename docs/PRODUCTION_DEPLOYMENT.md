# Déploiement de production — VPS CHAP RDC

Ce guide ajoute le service `chap-disease-forecasting` au déploiement CHAP existant
(`compose.ghcr.yml` + `compose.worker.yml`). Il ne modifie ni CHAP Core, ni le worker, ni
PostgreSQL, ni Valkey. **Aucune de ces commandes n'a encore été exécutée sur le VPS** : les
valeurs `A_CONFIRMER` doivent être relevées par les diagnostics de la section 2.

Deux chemins sont supportés par CHAP 2.4.0 (voir le README, « Deploy in CHAP ») :

- **A — `chap-admin` + entrée marketplace (recommandé)** : service écrit dans
  `compose.marketplace.yml`, trois configured models créés automatiquement, image refusée si
  son commit ne correspond pas à l'épinglage, rollback automatique d'un update raté.
- **B — overlay `compose.chap-disease-forecasting.yml`** : limites CPU/RAM incluses, configured
  models créés par trois appels API.

Ne jamais utiliser les deux : ils enregistrent le même `service_id`.

## 1. Préconditions bloquantes

1. `uname -m` = `x86_64`.
2. Version de CHAP Core confirmée = 2.4.0 (sinon relire `docs/compatibility-matrix.md`).
3. Nom du projet Compose et liste exacte des fichiers `-f` connus.
4. Sauvegarde PostgreSQL restaurable selon la procédure déjà approuvée.
5. Image publiée par la CI (`sha-<7>` et digest) et release épinglée dans
   `marketplace/models/chap_disease_forecasting.yaml` (chemin A).
6. Fenêtre de changement autorisée pour les mutations (sections 4 à 6).

## 2. Diagnostic en lecture seule

Depuis le répertoire qui contient `compose.ghcr.yml` :

```bash
FILES=(-f compose.ghcr.yml -f compose.worker.yml)   # ajouter compose.marketplace.yml s'il existe
CHAP=http://127.0.0.1:8000

uname -m; uname -sr
docker version --format '{{.Server.Version}}'; docker compose version
docker info --format 'CPUs={{.NCPU}} Memory={{.MemTotal}} Root={{.DockerRootDir}}'
free -h; df -hT / /var/lib/docker; swapon --show

docker compose "${FILES[@]}" config --services
docker compose "${FILES[@]}" ps
docker compose "${FILES[@]}" images
docker compose "${FILES[@]}" config --format json | jq -r .name     # nom du projet
ls -l compose*.yml .env

curl -fsS "$CHAP/health"
curl -fsS "$CHAP/v2/services" | jq '.services[] | {id, url}'
curl -fsS "$CHAP/v1/crud/model-templates" | jq '.[] | {id, name, version, healthStatus}'
grep -c '^SERVICEKIT_REGISTRATION_KEY=' .env; grep -c '^CHAP_API_TOKEN=' .env   # présence seulement
```

Si `CHAP_API_TOKEN` est défini, ajouter `-H "Authorization: Bearer $CHAP_API_TOKEN"` aux
appels `curl` (exporter la variable sans l'afficher). Ne pas imprimer `.Config.Env`, `.env`
ni `docker compose config` sans `--quiet`/filtre : ces sorties contiennent des secrets.

Vérifier en particulier que `chap` est sur les réseaux `default` et `backend`, et
PostgreSQL/Valkey sur `backend` seulement (configuration standard 2.4.0) :

```bash
docker compose "${FILES[@]}" config --format json \
  | jq '.services | to_entries[] | {service: .key, networks: (.value.networks | keys)}'
```

## 3. Préparation

```bash
# Image privée : jeton GHCR limité à read:packages, lu sur l'entrée standard
read -r -s -p 'GHCR token: ' GHCR_READ_TOKEN; echo
printf '%s' "$GHCR_READ_TOKEN" | docker login ghcr.io --username A_CONFIRMER --password-stdin
unset GHCR_READ_TOKEN

# Code du service (entrée marketplace, overlay, exemple d'environnement)
git clone https://github.com/HISP-RDC/chap_disease_forecasting.git ~/chap-disease-forecasting
git -C ~/chap-disease-forecasting log -1 --format=%H
```

## 4A. Installation avec `chap-admin` (recommandé)

```bash
uv tool install chap-core==2.4.0 --python 3.13       # même version que CHAP Core
chap-admin install --help                            # vérifier la syntaxe de --compose-file

python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory ~/chap-disease-forecasting/marketplace &
MARKETPLACE_PID=$!
export CHAP_MARKETPLACE_URL=http://127.0.0.1:8765

# Si CHAP_API_TOKEN est défini : chap-admin ne lit pas .env, et le service doit présenter
# une clé à POST /v2/services/$register (sinon 401, service jamais enregistré, rollback).
# Voir README, section « Authentication ». Correctif rapide sans redémarrer CHAP :
# export CHAP_API_TOKEN="$(grep '^CHAP_API_TOKEN=' .env | cut -d= -f2-)"
# sed -i '/^SERVICEKIT_REGISTRATION_KEY=$/d' .env   # drop an empty definition first
# grep -q '^SERVICEKIT_REGISTRATION_KEY=' .env || echo 'SERVICEKIT_REGISTRATION_KEY=${CHAP_API_TOKEN}' >> .env

chap-admin install chap_disease_forecasting \
  --compose-file compose.ghcr.yml --compose-file compose.worker.yml --accept-risk

kill "$MARKETPLACE_PID"
```

`chap-admin` refuse l'entrée tant qu'aucune release vérifiée n'est épinglée. En cas d'échec
(image, démarrage, enregistrement, commit), il retire le service et ne publie pas
`compose.marketplace.yml`. Ensuite, **toutes** les commandes Compose de ce déploiement
doivent inclure `-f compose.marketplace.yml`.

Limites de ressources : ajouter au service `marketplace-chap-disease-forecasting` de
`compose.marketplace.yml` (conservé par `chap-admin update`), puis recréer ce seul service :

```yaml
cpus: "4.0"
mem_limit: 16g
pids_limit: 512
environment:
  OMP_NUM_THREADS: "4"
  MALLOC_ARENA_MAX: "2"
logging:
  driver: json-file
  options: { max-size: 20m, max-file: "5" }
```

```bash
FILES=(-f compose.ghcr.yml -f compose.worker.yml -f compose.marketplace.yml)
docker compose "${FILES[@]}" config --quiet
docker compose "${FILES[@]}" up -d --no-deps marketplace-chap-disease-forecasting
```

`chap-admin` crée `xgboost`, `lightgbm` et `auto`; passer directement à la section 5.

## 4B. Installation avec l'overlay

```bash
cp ~/chap-disease-forecasting/compose.chap-disease-forecasting.yml .
cat ~/chap-disease-forecasting/deploy/runtime.env.example >> .env
sudoedit .env    # DISEASE_FORECASTING_IMAGE=ghcr.io/hisp-rdc/chap_disease_forecasting@sha256:<digest CI>

FILES=(-f compose.ghcr.yml -f compose.worker.yml -f compose.chap-disease-forecasting.yml)
docker compose "${FILES[@]}" config --quiet
docker compose "${FILES[@]}" pull chap-disease-forecasting
docker compose "${FILES[@]}" up -d --no-deps chap-disease-forecasting
```

Création des configured models, avec l'ID de template lu (jamais codé en dur) :

```bash
AUTH=()   # AUTH=(-H "Authorization: Bearer $CHAP_API_TOKEN")
TEMPLATE_ID=$(curl -fsS "${AUTH[@]}" "$CHAP/v1/crud/model-templates" \
  | jq '[.[] | select(.name == "chap-disease-forecasting")] | max_by(.id) | .id')
curl -fsS "${AUTH[@]}" "$CHAP/v1/crud/model-templates" \
  | jq --argjson id "$TEMPLATE_ID" '.[] | select(.id == $id) | .userOptions.properties.engine'

for ENGINE in xgboost lightgbm auto; do
  curl -fsS "${AUTH[@]}" -X POST "$CHAP/v1/crud/configured-models" \
    -H 'Content-Type: application/json' \
    -d "{\"name\": \"$ENGINE\", \"modelTemplateId\": $TEMPLATE_ID, \"userOptionValues\": {\"engine\": \"$ENGINE\"}}"
done
```

## 5. Vérification

```bash
SERVICE=marketplace-chap-disease-forecasting     # chemin B : chap-disease-forecasting
CID=$(docker compose "${FILES[@]}" ps -q "$SERVICE"); test -n "$CID"

docker inspect "$CID" --format 'status={{.State.Status}} health={{.State.Health.Status}} user={{.Config.User}} ro={{.HostConfig.ReadonlyRootfs}} privileged={{.HostConfig.Privileged}}'
docker inspect "$CID" --format '{{json .NetworkSettings.Ports}}'          # aucune liaison hôte
docker inspect "$CID" --format '{{range $n, $_ := .NetworkSettings.Networks}}{{println $n}}{{end}}'  # <projet>_default seulement
docker compose "${FILES[@]}" logs --tail=200 "$SERVICE" | grep -E 'registration|keepalive|service_start'

curl -fsS "${AUTH[@]}" "$CHAP/v2/services" | jq '.services[] | select(.id == "chap-disease-forecasting")'
curl -fsS "${AUTH[@]}" "$CHAP/v1/crud/model-templates" \
  | jq '.[] | select(.name == "chap-disease-forecasting") | {id, version, healthStatus}'
curl -fsS "${AUTH[@]}" "$CHAP/v1/crud/configured-models" | jq -r '.[].name' | grep chap-disease-forecasting
```

Attendu : service présent, template `healthStatus: live`, trois configured models
`chap-disease-forecasting:{xgboost,lightgbm,best_of_both}`. Puis, dans DHIS2 Modeling App :
rafraîchir la liste, vérifier les trois modèles et lancer un petit backtest pour chacun. Les
logs du service doivent contenir `disease_models_trained` puis `disease_predictions_created`
avec le moteur sélectionné et `used_fallback`.

## 6. Acceptation graduelle

Pour chaque palier, conserver : ID du job CHAP, digest/commit de l'image, configured model,
zones × semaines, splits, horizon, durée, pic mémoire, CPU, erreurs. Passer au palier suivant
seulement après succès.

| Palier                   | But                     | Critères bloquants                                                      |
| ------------------------ | ----------------------- | ----------------------------------------------------------------------- |
| 10 zones                 | contrat et flux complet | 3 horizons, sorties finies ≥ 0, aucune zone perdue, santé verte         |
| 100 zones × 196 semaines | ressources              | 7 splits, aucune erreur, sous les limites Compose                       |
| 514 zones × 196 semaines | capacité cible          | 7 splits, 514 zones couvertes, aucun OOM/restart, enregistrement stable |

Référence hors CHAP (synthétique, macOS arm64, 120 arbres, 4 threads, par appel train) :
`xgboost` 40 s, `lightgbm` 43 s, `auto` 82 s; predict ≈ 4 s; pic RSS ≤ 525 Mio. Un backtest à
7 splits enchaîne 7 train + predict : prévoir de l'ordre de 10 minutes pour `auto` à 514 zones,
à remesurer sur le VPS, service inactif (la mesure partage ses CPU et sa mémoire) :

```bash
docker compose "${FILES[@]}" exec -T "$SERVICE" \
  chap-disease-benchmark --zones 514 --engine auto --through-service --n-estimators 120 --threads 4
```

Surveillance pendant chaque palier :

```bash
docker stats --no-stream "$CID"
docker inspect "$CID" --format 'oom={{.State.OOMKilled}} restarts={{.RestartCount}} health={{.State.Health.Status}}'
docker compose "${FILES[@]}" logs --since=30m "$SERVICE" | grep -E 'error|keepalive.ping_failed|fallback'
curl -fsS "${AUTH[@]}" "$CHAP/v2/services" | jq '.services[].id'
```

Un service qui disparaît de `/v2/services` pendant un entraînement indique un keepalive
bloqué ou un OOM : c'est bloquant.

## 7. Mise à jour et rollback

**Chemin A.** Épingler la nouvelle release (`deploy/pin_release.py`), mettre à jour le clone,
servir `marketplace/`, puis :

```bash
chap-admin update chap_disease_forecasting \
  --compose-file compose.ghcr.yml --compose-file compose.worker.yml --accept-risk
```

Si le nouveau conteneur ne démarre pas ou ne s'enregistre pas, `chap-admin` restaure
l'ancien. Pour revenir en arrière sur une release fonctionnelle mais mauvaise, remettre
`channels.stable` sur la version précédente dans l'entrée et relancer `update`. Les versions
antérieures du template et leurs backtests restent dans CHAP.

**Chemin B.** Sauvegarder le volume (OPERATIONS.md), changer `DISEASE_FORECASTING_IMAGE`,
`config --quiet`, `pull`, `up -d --no-deps chap-disease-forecasting`. Rollback : remettre le
digest précédent et relancer la même commande.

Dans les deux cas : un changement de format de descripteur (ex. 0.1.0 → 0.2.0) oblige à
réentraîner; le service refuse un ancien artefact avec un message explicite. Vérifier ensuite
`/health`, l'enregistrement, les configured models et une prédiction ciblée.

Interdits : `docker compose down -v`, suppression des volumes CHAP/PostgreSQL/Valkey ou du
volume du modèle, `chap-admin uninstall --delete-data`.

## 8. Migration depuis `chap-malaria-forecasting` (0.3) vers `chap-disease-forecasting` (0.4)

La 0.4 est un **nouveau modèle** dans CHAP (identifiant `chap-disease-forecasting`), pas une
mise à jour de l'ancien. Les deux peuvent tourner côte à côte le temps de les comparer.

1. **GitHub** (une fois) : le code vit dans le dépôt de l'organisation
   [`HISP-RDC/chap_disease_forecasting`](https://github.com/HISP-RDC/chap_disease_forecasting).
   Sa CI publie `ghcr.io/hisp-rdc/chap_disease_forecasting`. Publié avec `GITHUB_TOKEN`, le
   paquet hérite normalement de la visibilité du dépôt (public), sauf si la politique de
   l'organisation interdit les paquets publics. Le vérifier après la première publication
   (commande ci-dessous); s'il est privé : paquet → Package settings → Change visibility,
   ou `docker login ghcr.io` sur le VPS avec un jeton `read:packages`.

   ```bash
   TOKEN=$(curl -fsS "https://ghcr.io/token?scope=repository:hisp-rdc/chap_disease_forecasting:pull" | jq -r .token)
   curl -s -o /dev/null -w 'HTTP %{http_code}\n' -H "Authorization: Bearer $TOKEN" \
     -H 'Accept: application/vnd.oci.image.index.v1+json' \
     https://ghcr.io/v2/hisp-rdc/chap_disease_forecasting/manifests/sha-<7 caractères>
   # HTTP 200 = public, téléchargeable sans authentification
   ```

   Vérifier aussi que GitHub Actions peut publier des paquets (Settings → Actions →
   Workflow permissions : « Read and write »).
2. **Release** : CI verte, puis `uv run python deploy/pin_release.py --commit <sha>` et push.
3. **Installation** à côté de l'ancien :

   ```bash
   cd /home/meleck/chap-core
   export CHAP_API_TOKEN="$(grep '^CHAP_API_TOKEN=' .env | cut -d= -f2-)"
   export CHAP_MARKETPLACE_URL=https://raw.githubusercontent.com/HISP-RDC/chap_disease_forecasting/main/marketplace
   chap-admin install chap_disease_forecasting \
     --compose-file compose.ghcr.yml --compose-file compose.worker.yml --accept-risk
   ```

   Reporter les limites CPU/RAM (section 4A) sur `marketplace-chap-disease-forecasting`.
4. **Comparaison** : mêmes backtests (paludisme, puis choléra) avec l'ancien et le nouveau
   modèle; comparer CRPS, MAE, couvertures et score d'intervalle dans la Modeling App.
5. **Retrait de l'ancien**, une fois la comparaison concluante :

   ```bash
   chap-admin uninstall chap_malaria_forecasting \
     --compose-file compose.ghcr.yml --compose-file compose.worker.yml
   ```

   Le template et les configured models sont retirés des listes; leurs backtests et
   prédictions restent consultables. Le volume
   `chap-core_marketplace-chap-malaria-forecasting-data` est conservé : ne pas utiliser
   `--delete-data` avant d'être certain de ne plus en avoir besoin.

Pour des données choléra, aucune configuration particulière n'est nécessaire : le jeu de
données choisi dans la Modeling App détermine la maladie. Les polygones des zones de santé,
envoyés par CHAP, activent les variables de voisinage, importantes pour le choléra.
