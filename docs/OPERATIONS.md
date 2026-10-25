# Exploitation et sécurité

Variables utilisées ci-dessous (voir `PRODUCTION_DEPLOYMENT.md`, section 2) :

```bash
FILES=(-f compose.ghcr.yml -f compose.worker.yml -f compose.marketplace.yml)  # chemin B : compose.chap-disease-forecasting.yml
SERVICE=marketplace-chap-disease-forecasting                                 # chemin B : chap-disease-forecasting
PROJECT=$(docker compose "${FILES[@]}" config --format json | jq -r .name)
MODEL_VOLUME="${PROJECT}_${SERVICE}-data"                                    # vérifier avec docker volume ls
CID=$(docker compose "${FILES[@]}" ps -q "$SERVICE")
```

## Indicateurs et alertes minimales

`/metrics` (Prometheus) n'est joignable que sur le réseau Docker interne; ne pas publier de
port pour le collecter. Événements applicatifs structurés à suivre dans les logs :

- `disease_models_trained` : moteurs entraînés, décision de sélection, incertitude disponible;
- `model_backtest_failed`, `model_training_failed` : preuve ou moteur indisponible;
- `disease_predictions_created` : modèle demandé/sélectionné, `used_fallback`, raison;
- `primary_artifact_unavailable` : bascule vers un modèle de secours;
- `keepalive.ping_failed`, `keepalive.re_registered` : santé de l'enregistrement CHAP.

Bloquant : tout OOM, sortie non finie, zone perdue, fallback inexpliqué, disparition de
`/v2/services`. Les seuils chiffrés de durée et mémoire se fixent à partir des paliers
10/100/514 mesurés sur le VPS (référence macOS dans `PRODUCTION_DEPLOYMENT.md`).

Commandes sans mutation :

```bash
docker compose "${FILES[@]}" ps "$SERVICE"
docker stats --no-stream "$CID"
docker inspect "$CID" --format 'oom={{.State.OOMKilled}} restarts={{.RestartCount}} health={{.State.Health.Status}}'
docker system df -v | grep -E "VOLUME NAME|${SERVICE}"
docker compose "${FILES[@]}" logs --since=15m --tail=500 "$SERVICE"
```

La file Celery appartient à CHAP Core et son worker, pas au service de modèle. Le service
exécute un seul job ChapKit à la fois (limite ChapKit liée à SQLite).

## Sauvegarde et restauration

Le volume du service ne contient que la base SQLite ChapKit (`/work/data/chapkit.db`) :
configurations, artefacts d'entraînement (modèles natifs embarqués et vérifiés par SHA-256)
et prédictions. Arrêter le seul service de modèle pour une copie cohérente; ne pas arrêter
CHAP, PostgreSQL ni Valkey.

```bash
BACKUP_HELPER_IMAGE='docker.io/library/alpine:3.22@sha256:dc92f017beea8e7f43263073d717f7bc011228c95d5312e440deb4d035bb4eec'
BACKUP_DIR=/srv/backups/chap-disease
BACKUP_NAME="chap-disease-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
sudo install -d -m 0750 "$BACKUP_DIR"

docker compose "${FILES[@]}" stop --timeout 90 "$SERVICE"
docker run --rm --read-only --network none \
  --mount type=volume,src="$MODEL_VOLUME",dst=/source,readonly \
  --mount type=bind,src="$BACKUP_DIR",dst=/backup \
  "$BACKUP_HELPER_IMAGE" tar -C /source -czf "/backup/$BACKUP_NAME" .
sha256sum "$BACKUP_DIR/$BACKUP_NAME" > "$BACKUP_DIR/$BACKUP_NAME.sha256"
docker compose "${FILES[@]}" up -d --no-deps "$SERVICE"
```

Le digest Alpine 3.22.7 (Linux AMD64) a été inspecté le 2026-10-25 sans téléchargement de
layer; le réévaluer explicitement, ne jamais le remplacer par un tag mutable.

Restauration : faire d'abord une sauvegarde de l'état courant (ci-dessus), puis :

```bash
RESTORE=chap-disease-YYYYmmddTHHMMSSZ.tar.gz      # sauvegarde à restaurer
sha256sum --check "$BACKUP_DIR/$RESTORE.sha256"
docker compose "${FILES[@]}" stop --timeout 90 "$SERVICE"
docker run --rm --read-only --network none \
  --mount type=bind,src="$BACKUP_DIR",dst=/backup,readonly \
  --mount type=volume,src="$MODEL_VOLUME",dst=/restore \
  "$BACKUP_HELPER_IMAGE" sh -c 'rm -f /restore/chapkit.db /restore/chapkit.db-wal /restore/chapkit.db-shm && tar -C /restore -xzf "/backup/$0"' "$RESTORE"
docker compose "${FILES[@]}" up -d --no-deps "$SERVICE"
```

Vérifier santé, enregistrement, configured models et une prédiction ciblée. Une sauvegarde
n'est compatible qu'avec une image de même format de descripteur (format 2 depuis la
version 0.2.0) : restaurer l'image et la base ensemble. Tester périodiquement une
restauration sur une machine de recette. Les sauvegardes PostgreSQL de CHAP suivent la
procédure existante; ce service n'a aucun accès réseau à PostgreSQL.

## Diagnostic d'incident

1. Relever image/commit (`GET /api/v1/info` via `docker compose exec`), heure, job CHAP,
   configured model.
2. Vérifier santé, OOM, redémarrages, CPU/RSS/disque.
3. Lire les logs bornés dans le temps, sans les publier s'ils contiennent des données.
4. Vérifier `/v2/services`, `healthStatus` du template et les configured models.
5. Classer : enregistrement, contrat de données (message `DataValidationError`), moteur,
   artefact (checksum, format ancien → réentraîner), capacité.
6. Reproduire sur données synthétiques avec la même image (`chap-disease-benchmark`).
7. Rollback si la régression vient de la nouvelle image.

Un modèle indisponible n'est jamais remplacé silencieusement : le fallback apparaît dans les
logs, sinon le job échoue avec une erreur structurée.

## Mises à jour de CHAP Core

- Lire la plage ChapKit et les notes de version de la nouvelle version de Core.
- Sauvegarder PostgreSQL et le volume du modèle.
- Mettre `chap-admin` à la même version : `uv tool install chap-core==<version> --python 3.13`.
- Valider sur une instance de recette : enregistrement, `userOptions`, configured models,
  backtest et prédiction pour les trois configurations.
- Conserver l'image précédente jusqu'à la fin de la période d'observation.
- Passer toujours les mêmes fichiers `-f` (dont `compose.marketplace.yml`), sinon le service
  est silencieusement retiré.

## Garde-fous sécurité

- Aucun port publié : CHAP joint le service par son nom DNS sur le réseau `default`;
  PostgreSQL et Valkey restent sur `backend`, hors de portée du service.
- Utilisateur `chapkit` non-root, rootfs en lecture seule, capabilities supprimées,
  `no-new-privileges`, `/tmp` en tmpfs.
- L'API du service n'a pas d'authentification propre : l'isolation réseau est le contrôle.
- `SERVICEKIT_REGISTRATION_KEY` est requise dès que CHAP définit `CHAP_API_TOKEN` (égale au
  token, ou clé dédiée transmise aussi au conteneur `chap`); elle vit dans le `.env`
  de CHAP (`chmod 600`). `deploy/start.sh` accepte aussi un secret monté en
  `/run/secrets/servicekit_registration_key`.
- Ne jamais committer `.env`, clés, données DHIS2, bases SQLite réelles ou logs sensibles.
- ChapKit pickle l'artefact d'entraînement : n'importer aucune base ou artefact d'origine
  non maîtrisée. Les modèles natifs embarqués sont vérifiés par SHA-256 avant chargement.
- Vérifier digest, SBOM et rapport `pip-audit` de la CI avant promotion.

## Rotation et rétention

Logs Docker : 20 Mio × 5 fichiers (overlay; à ajouter au service généré par `chap-admin`).
La base SQLite grossit à chaque entraînement/prédiction (≈ 1,5 à 3 Mio par entraînement à
514 zones, plus le workspace zippé par ChapKit). Surveiller la taille du volume; purger des
artefacts uniquement via l'API ChapKit (`DELETE /api/v1/artifacts/{id}`) pour des versions
inactives, non nécessaires au rollback et sauvegardées. Toute purge de volume en bloc est
interdite.
