"""Minimal HTTP helper with bounded retries and strict JSON validation."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from mtmbot.data.base import DataError

__all__ = ["get_json"]

USER_AGENT = "MTMbot/1.0 (+https://github.com/MrTalal-AI/MTMbot)"
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 15.0,
    retries: int = 4,
    backoff: float = 0.6,
    headers: dict[str, str] | None = None,
) -> Any:
    """GET ``url`` and decode JSON, retrying transient failures.

    Raises:
        DataError: on a permanent HTTP error, exhausted retries or bad payload.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries!r}")

    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    full_url = f"{url}{query}"
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    last_error: str = "unknown error"
    for attempt in range(retries + 1):
        request = urllib.request.Request(full_url, headers=request_headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code} from {url}"
            if exc.code not in _RETRY_STATUS:
                raise DataError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"network error talking to {url}: {exc}"
        else:
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DataError(f"{url} returned invalid JSON: {exc}") from exc

        if attempt < retries:
            delay = backoff * (2**attempt) * (0.75 + random.random() / 2)
            time.sleep(min(delay, 30.0))

    raise DataError(last_error)
