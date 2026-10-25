"""Dependency-free container health probe for the CHAPKit HTTP endpoint."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def main() -> int:
    url = os.environ.get("DISEASE_HEALTHCHECK_URL", "http://127.0.0.1:8000/health")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        print("health endpoint must use HTTP on the local container", file=sys.stderr)
        return 1
    try:
        request = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310
            if 200 <= response.status < 300:
                return 0
            print(f"health endpoint returned HTTP {response.status}", file=sys.stderr)
    except (OSError, urllib.error.URLError) as exc:
        print(f"health endpoint unavailable: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
