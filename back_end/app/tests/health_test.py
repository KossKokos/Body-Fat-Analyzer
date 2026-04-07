from __future__ import annotations

import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def print_result(title: str, ok: bool, status_code: int | None, data: Any) -> None:
    icon = "OK" if ok else "FAIL"
    print(f"\n[{icon}] {title}")
    if status_code is not None:
        print(f"Status: {status_code}")
    print("Response:")
    print(data)


def get_json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def main() -> int:
    load_dotenv()

    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
    api_key = require_env("API_KEY")

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    checks = [
        ("App health", f"{base_url}/api/health/"),
        ("DB health", f"{base_url}/api/health/db"),
    ]

    all_ok = True

    for title, url in checks:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            ok = 200 <= response.status_code < 300
            all_ok = all_ok and ok
            print_result(title, ok, response.status_code, get_json_or_text(response))
        except requests.RequestException as exc:
            all_ok = False
            print_result(title, False, None, f"Request failed: {exc}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())