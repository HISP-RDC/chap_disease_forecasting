# syntax=docker/dockerfile:1.7

# Manifest list inspected on 2026-10-25. Override only with another immutable
# ghcr.io/dhis2-chap/chapkit-py reference (tag@sha256 or digest).
ARG CHAPKIT_BASE_IMAGE=ghcr.io/dhis2-chap/chapkit-py@sha256:0cd473d5f665aae6e268bac4a60d905162d4f2984ea8c7ea615d06831fe67a16
FROM ${CHAPKIT_BASE_IMAGE}

ARG GIT_REVISION
ARG APP_VERSION=0.0.0-dev

LABEL org.opencontainers.image.title="CHAP weekly disease forecasting service" \
      org.opencontainers.image.description="XGBoost, LightGBM and controlled ensemble forecasting for CHAP" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/HISP-RDC/chap_disease_forecasting" \
      org.opencontainers.image.revision="${GIT_REVISION}"

USER root
WORKDIR /work

# LightGBM wheels link against the system OpenMP runtime (libgomp.so.1), which
# the slim chapkit-py base does not ship; XGBoost vendors its own copy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The base image provides Python 3.13 and uv. The lockfile remains the sole
# dependency resolution source; production builds intentionally exclude dev deps.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY main.py ./main.py

RUN --mount=type=cache,target=/root/.cache/uv \
    test "${#GIT_REVISION}" -eq 40 && \
    case "${GIT_REVISION}" in *[!0-9a-fA-F]*) \
      echo "GIT_REVISION must be a full 40-character hexadecimal commit SHA" >&2; exit 64 ;; \
    esac && \
    UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --frozen --no-dev && \
    if ! getent group chapkit >/dev/null 2>&1; then groupadd --gid 1000 chapkit; fi && \
    if ! id chapkit >/dev/null 2>&1; then useradd --uid 1000 --gid chapkit --no-create-home --home-dir /tmp --shell /usr/sbin/nologin chapkit; fi && \
    install -d -o chapkit -g chapkit -m 0750 /work/data /tmp/.cache

COPY --chown=chapkit:chapkit --chmod=0555 deploy/start.sh /usr/local/bin/start-disease-model
COPY --chown=chapkit:chapkit --chmod=0444 deploy/healthcheck.py /usr/local/lib/chap-disease-healthcheck.py

ENV PATH="/app/.venv/bin:${PATH}" \
    GIT_REVISION="${GIT_REVISION}" \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONUNBUFFERED=1 \
    DISEASE_HEALTHCHECK_URL=http://127.0.0.1:8000/health

USER chapkit:chapkit
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/start-disease-model"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "/usr/local/lib/chap-disease-healthcheck.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
