"""Add a verified release pin to the marketplace entry read by `chap-admin`.

Usage (after CI published ghcr.io/<owner>/chap-disease-forecasting:sha-<commit[:7]>
and the acceptance tests passed):

    uv run python deploy/pin_release.py --commit <40-char sha> [--version 0.2.0]

Only the top-level ``channels`` and ``versions`` sections are rewritten, so the
comments elsewhere in the YAML file are preserved.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "marketplace" / "models" / "chap_disease_forecasting.yaml"
_TOP_LEVEL = re.compile(r"^([a-z_]+):")


def _project() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return dict(tomllib.load(stream)["project"])


def _chapkit_range(project: dict[str, Any]) -> str:
    for requirement in project["dependencies"]:
        if requirement.startswith("chapkit"):
            return str(requirement.removeprefix("chapkit")).strip()
    raise ValueError("chapkit is not a declared dependency")


def _replace_sections(text: str, replacements: dict[str, str]) -> str:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines(keepends=True):
        match = _TOP_LEVEL.match(line)
        if match:
            sections.append((match.group(1), [line]))
        else:
            sections[-1][1].append(line)
    output: list[str] = []
    for key, lines in sections:
        if key in replacements:
            # Keep trailing blank and comment lines: they introduce the next section.
            tail: list[str] = []
            while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")):
                tail.insert(0, lines.pop())
            output.append(replacements[key])
            output.extend(tail)
        else:
            output.extend(lines)
    return "".join(output)


def pin(entry_path: Path, commit: str, version: str, changelog: str | None = None) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a full 40-character lowercase hexadecimal SHA")
    text = entry_path.read_text(encoding="utf-8")
    entry = yaml.safe_load(text)
    versions = list(entry.get("versions") or [])
    if any(item["version"] == version for item in versions):
        raise ValueError(f"version {version} is already pinned; bump the service version")
    release: dict[str, Any] = {
        "version": version,
        "commit": commit,
        "image_tag": f"sha-{commit[:7]}",
        "chapkit": _chapkit_range(_project()),
        "status": "verified",
    }
    if changelog:
        release["changelog"] = changelog
    versions.append(release)
    replacements = {
        "channels": yaml.safe_dump(
            {"channels": {"stable": version, "latest": version}}, sort_keys=False
        ),
        "versions": yaml.safe_dump({"versions": versions}, sort_keys=False),
    }
    entry_path.write_text(_replace_sections(text, replacements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version", default=_project()["version"])
    parser.add_argument("--changelog")
    parser.add_argument("--entry", type=Path, default=ENTRY)
    args = parser.parse_args()
    pin(args.entry, args.commit, args.version, args.changelog)
    print(f"pinned {args.version} -> sha-{args.commit[:7]} in {args.entry}")


if __name__ == "__main__":
    main()
