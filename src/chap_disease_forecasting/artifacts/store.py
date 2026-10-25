"""Safe local artifact bundles with path validation and SHA-256 manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from chap_disease_forecasting.forecasting import DirectForecaster

_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUNDLE_PATH = re.compile(r"^[A-Za-z0-9_]{1,64}(/[A-Za-z0-9_]{1,64}){0,3}\.(json|txt)$")
BUNDLE_FORMAT_VERSION = 1


class ArtifactIntegrityError(ValueError):
    pass


def pack_forecaster(model: DirectForecaster) -> dict[str, Any]:
    """Serialize a fitted forecaster to an in-memory bundle of native text files.

    XGBoost JSON, LightGBM text and metadata JSON are all text formats, so the
    bundle can travel inside the CHAPKit training artifact without pickling
    any engine object and without depending on a second storage location.
    """

    with tempfile.TemporaryDirectory(prefix="chap-disease-pack-") as temporary:
        directory = Path(temporary)
        model.save(directory)
        files: dict[str, dict[str, str]] = {}
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if not _BUNDLE_PATH.fullmatch(relative):
                raise ArtifactIntegrityError(f"unexpected file in model bundle: {relative}")
            text = path.read_text(encoding="utf-8")
            files[relative] = {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
    return {"format_version": BUNDLE_FORMAT_VERSION, "hash": "sha256", "files": files}


def unpack_forecaster(bundle: Any) -> DirectForecaster:
    """Verify every path and checksum of a bundle before loading native files."""

    if (
        not isinstance(bundle, dict)
        or bundle.get("format_version") != BUNDLE_FORMAT_VERSION
        or bundle.get("hash") != "sha256"
        or not isinstance(bundle.get("files"), dict)
        or not bundle["files"]
    ):
        raise ArtifactIntegrityError("unsupported or empty model bundle")
    with tempfile.TemporaryDirectory(prefix="chap-disease-unpack-") as temporary:
        directory = Path(temporary)
        for relative, entry in bundle["files"].items():
            if not isinstance(relative, str) or not _BUNDLE_PATH.fullmatch(relative):
                raise ArtifactIntegrityError("unsafe file path in model bundle")
            if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
                raise ArtifactIntegrityError(f"invalid bundle entry: {relative}")
            payload = entry["text"].encode("utf-8")
            if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                raise ArtifactIntegrityError(f"checksum mismatch in model bundle: {relative}")
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return DirectForecaster.load(directory)


class ArtifactStore:
    """Store only documented native model formats; never deserialize pickle."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError("artifact_id contains unsafe characters")
        path = (self.root / artifact_id).resolve()
        if path.parent != self.root:
            raise ValueError("artifact path escapes the configured root")
        return path

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _manifest(cls, directory: Path) -> dict[str, object]:
        files = {
            str(path.relative_to(directory)): cls._digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        return {"format_version": 1, "hash": "sha256", "files": files}

    def save(self, artifact_id: str, model: DirectForecaster) -> Path:
        destination = self._path(artifact_id)
        if destination.exists():
            raise FileExistsError(f"artifact already exists: {artifact_id}")
        with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=self.root) as temporary:
            staging = Path(temporary) / artifact_id
            model.save(staging)
            manifest = self._manifest(staging)
            (staging / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )
            os.replace(staging, destination)
        return destination

    def verify(self, artifact_id: str) -> Path:
        directory = self._path(artifact_id)
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.is_file():
            raise ArtifactIntegrityError("artifact or manifest is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("manifest is unreadable") from exc
        if manifest.get("format_version") != 1 or manifest.get("hash") != "sha256":
            raise ArtifactIntegrityError("unsupported artifact manifest")
        expected = manifest.get("files")
        if not isinstance(expected, dict):
            raise ArtifactIntegrityError("invalid artifact manifest file list")
        observed = self._manifest(directory)["files"]
        if observed != expected:
            raise ArtifactIntegrityError("artifact checksum or file list mismatch")
        return directory

    def load(self, artifact_id: str) -> DirectForecaster:
        return DirectForecaster.load(self.verify(artifact_id))
