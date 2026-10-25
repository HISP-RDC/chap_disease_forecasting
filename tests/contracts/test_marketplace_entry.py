"""The marketplace entry must pass the checks `chap-admin` (CHAP Core 2.4.0) applies.

Rules mirrored from chap_core/services/model_marketplace.py and
chap_core/cli_endpoints/marketplace.py at tag v2.4.0.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from chap_disease_forecasting.chapkit_adapter.service import (
    MODEL_VERSION,
    SERVICE_ID,
    DiseaseModelConfig,
    create_app,
)

ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = ROOT / "marketplace"
RESERVED_CONFIG_KEYS = ("prediction_periods", "additional_continuous_covariates")
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _entry(path: Path | None = None) -> dict[str, Any]:
    path = path or MARKETPLACE / "models" / "chap_disease_forecasting.yaml"
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _pin_module() -> Any:
    spec = importlib.util.spec_from_file_location("pin_release", ROOT / "deploy" / "pin_release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stable_pin(entry: dict[str, Any]) -> str:
    stable = entry["channels"].get("stable")
    version = next((item for item in entry["versions"] if item["version"] == stable), None)
    if version is None or version["status"] != "verified":
        raise ValueError("no verified stable version")
    prefix = version["image_tag"][4:] if version["image_tag"].startswith("sha-") else ""
    if len(prefix) < 7 or not version["commit"].startswith(prefix):
        raise ValueError("invalid stable image pin")
    return f"{entry['source']['image']}:{version['image_tag']}"


def test_registry_lists_the_entry() -> None:
    registry = yaml.safe_load((MARKETPLACE / "registry.yaml").read_text(encoding="utf-8"))
    assert registry["schema_version"] == 2
    assert registry["models"] == ["models/chap_disease_forecasting.yaml"]


def test_identity_matches_the_running_service(tmp_path, monkeypatch) -> None:
    entry = _entry()
    assert entry["schema_version"] == 2
    assert entry["kind"] == "model"
    assert re.fullmatch(r"[a-z0-9][a-z0-9_]*", entry["id"])
    assert entry["service_id"] == entry["id"].replace("_", "-") == SERVICE_ID
    assert re.fullmatch(r"[a-z0-9][a-z0-9._/\-]*", entry["source"]["image"])

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'chapkit.db'}")
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as client:
        info = client.get("/api/v1/info").json()
    assert info["id"] == entry["service_id"]
    assert info["display_name"] == entry["display_name"]
    assert info["period_type"] in entry["compatibility"]["period_types"]
    assert info["min_prediction_periods"] == entry["compatibility"]["min_prediction_periods"]
    assert info["max_prediction_periods"] == entry["compatibility"]["max_prediction_periods"]
    assert info["required_covariates"] == entry["covariates"]["required"]
    assert info["allow_free_additional_continuous_covariates"] is False
    assert entry["covariates"]["allow_free_additional"] is False
    # The marketplace copies attribution and status from model_metadata.
    metadata = info["model_metadata"]
    assert metadata["author"] == entry["attribution"]["author"]
    assert metadata["organization"] == entry["attribution"]["organization"]
    assert metadata["author_assessed_status"] == entry["assessed_status"]
    assert metadata["repository_url"].rstrip("/") == entry["source"]["repository"]


def test_each_configuration_is_stored_unchanged_by_the_service() -> None:
    properties = DiseaseModelConfig.model_json_schema()["properties"]
    configurations = _entry()["configurations"]
    engines = {"xgboost": "xgboost", "lightgbm": "lightgbm", "best_of_both": "auto"}
    assert set(configurations) == set(engines)
    for name, configuration in configurations.items():
        # Required by the official marketplace schema, dropped by chap-admin.
        periods = configuration["config"]["prediction_periods"]
        assert isinstance(periods, int) and 1 <= periods <= 3, name
        options = {
            key: value
            for key, value in configuration["config"].items()
            if key not in RESERVED_CONFIG_KEYS
        }
        # Core validates options against the template schema, then refuses a
        # service that stores a different value than the one sent.
        assert set(options) <= set(properties), name
        stored = DiseaseModelConfig(**options).model_dump(mode="json")
        assert {key: stored[key] for key in options} == options, name
        assert options["engine"] == engines[name]


def test_public_names_shown_in_the_modeling_app() -> None:
    # ConfiguredModelDB.display_name in chap_core/database/model_templates_and_config_tables.py
    # (v2.4.0): "<template display_name> [<stub with spaces, .capitalize()>]".
    entry = _entry()
    shown = sorted(
        f"{entry['display_name']} [{stub.replace('_', ' ').capitalize()}]"
        for stub in entry["configurations"]
    )
    assert shown == [
        "Weekly Disease Forecast (Gradient Boosting) [Best of both]",
        "Weekly Disease Forecast (Gradient Boosting) [Lightgbm]",
        "Weekly Disease Forecast (Gradient Boosting) [Xgboost]",
    ]


def test_committed_stable_pin_is_accepted_by_chap_admin() -> None:
    entry = _entry()
    if not entry["versions"]:
        pytest.skip("no release pinned yet; pin_release.py adds the first one")
    image = _stable_pin(entry)
    stable = next(v for v in entry["versions"] if v["version"] == entry["channels"]["stable"])
    assert image == f"{entry['source']['image']}:sha-{stable['commit'][:7]}"
    assert re.fullmatch(r"[0-9a-f]{40}", stable["commit"])
    assert entry["channels"]["latest"] in {v["version"] for v in entry["versions"]}


def _unpinned_copy(tmp_path: Path, pin_release: Any) -> Path:
    target = tmp_path / "entry.yaml"
    shutil.copy(MARKETPLACE / "models" / "chap_disease_forecasting.yaml", target)
    text = pin_release._replace_sections(
        target.read_text(encoding="utf-8"),
        {"channels": "channels: {}\n", "versions": "versions: []\n"},
    )
    target.write_text(text, encoding="utf-8")
    return target


def test_unpinned_entry_is_refused_by_chap_admin(tmp_path) -> None:
    target = _unpinned_copy(tmp_path, _pin_module())
    with pytest.raises(ValueError, match="no verified stable version"):
        _stable_pin(_entry(target))


def test_pin_release_produces_a_pin_chap_admin_accepts(tmp_path) -> None:
    pin_release = _pin_module()
    target = _unpinned_copy(tmp_path, pin_release)
    original = target.read_text(encoding="utf-8")

    pin_release.pin(target, COMMIT, MODEL_VERSION, "first release")
    entry = _entry(target)

    assert _stable_pin(entry).endswith(":sha-0123456")
    release = entry["versions"][0]
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}", release["version"])
    assert re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}", release["image_tag"])
    assert release["chapkit"] == ">=2.4.1,<3"
    # Everything outside channels/versions, comments included, is untouched.
    assert "# Each configuration becomes one configured model" in target.read_text()
    assert entry["configurations"] == yaml.safe_load(original)["configurations"]
    with pytest.raises(ValueError, match="already pinned"):
        pin_release.pin(target, COMMIT, MODEL_VERSION)


def test_service_version_matches_the_package_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        assert tomllib.load(stream)["project"]["version"] == MODEL_VERSION
