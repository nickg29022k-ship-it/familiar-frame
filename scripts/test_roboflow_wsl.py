#!/usr/bin/env python3
"""Probe WSL -> Roboflow connectivity, then run one face-detection inference.

Usage:
  # Put keys back in .env first, then:
  source .venv_hybrid/bin/activate
  python scripts/test_roboflow_wsl.py path/to/face.jpg
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def check_dns(host: str) -> str:
    addrs = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    ip = addrs[0][4][0]
    print(f"[ok] DNS {host} -> {ip}")
    return ip


def check_tcp(host: str, port: int = 443, timeout: float = 8.0) -> None:
    started = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        elapsed = time.time() - started
        raise TimeoutError(
            f"TCP {host}:{port} failed after {elapsed:.2f}s ({exc}). "
            "DNS works but WSL cannot open outbound HTTPS — same reason enroll stalls."
        ) from exc
    print(f"[ok] TCP {host}:{port} in {time.time() - started:.2f}s")


def check_https(url: str, timeout: float = 10.0) -> None:
    started = time.time()
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        # Any HTTP response means the network path works.
        status = exc.code
    elapsed = time.time() - started
    print(f"[ok] HTTPS reachability {url} -> HTTP {status} in {elapsed:.2f}s")


def run_infer(image: Path, api_key: str, model_id: str, timeout: float = 60.0) -> dict:
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    # inference-sdk doesn't expose a clean timeout kwarg everywhere; bound with socket default.
    socket.setdefaulttimeout(timeout)
    started = time.time()
    result = client.infer(str(image), model_id=model_id)
    elapsed = time.time() - started
    preds = result.get("predictions", []) if isinstance(result, dict) else []
    print(f"[ok] Roboflow infer in {elapsed:.2f}s — {len(preds)} prediction(s)")
    print(json.dumps(result, indent=2)[:2000])
    return result


def main() -> int:
    load_env(ROOT / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    model_id = os.environ.get("ROBOFLOW_MODEL_ID", "").strip()

    print(f"cwd={Path.cwd()}")
    print(f"env_file={ROOT / '.env'}")

    if not api_key or not model_id:
        print(
            "[fail] ROBOFLOW_API_KEY / ROBOFLOW_MODEL_ID empty.\n"
            "       Uncomment them in .env, save, then rerun this script."
        )
        return 1

    print(f"model_id={model_id}")
    print(f"api_key_set=yes (len={len(api_key)})")

    host = "serverless.roboflow.com"
    try:
        check_dns(host)
    except OSError as exc:
        print(f"[fail] DNS for {host}: {exc}")
        print(
            "\nWSL DNS fix to try:\n"
            "  1) sudo sh -c 'echo \"nameserver 8.8.8.8\" > /etc/resolv.conf'\n"
            "  2) Or in Windows PowerShell (Admin):\n"
            "       wsl --shutdown\n"
            "     then reopen WSL and retry.\n"
            "  3) Create %UserProfile%\\.wslconfig with:\n"
            "       [network]\n"
            "       dnsTunneling=true\n"
            "     then wsl --shutdown and retry.\n"
        )
        return 2

    try:
        check_tcp(host, 443, timeout=8.0)
        check_https(f"https://{host}/", timeout=15.0)
    except Exception as exc:
        print(f"[fail] HTTPS/TCP to {host}: {exc}")
        print(
            "\nWSL can resolve DNS but cannot connect. Try this from Windows PowerShell:\n"
            "  curl.exe -I https://serverless.roboflow.com\n"
            "If that works, fix WSL networking:\n"
            "  1) In Windows: wsl --shutdown\n"
            "  2) Create/edit C:\\Users\\<you>\\.wslconfig :\n"
            "       [wsl2]\n"
            "       networkingMode=mirrored\n"
            "       dnsTunneling=true\n"
            "  3) Reopen WSL and rerun this script.\n"
            "Or keep using local MTCNN (empty keys) until WSL networking is fixed.\n"
        )
        return 3

    if len(sys.argv) < 2:
        print("[fail] Pass an image path: python scripts/test_roboflow_wsl.py face.jpg")
        return 4

    image = Path(sys.argv[1]).expanduser().resolve()
    if not image.is_file():
        print(f"[fail] Image not found: {image}")
        return 4

    print(f"image={image}")
    try:
        run_infer(image, api_key=api_key, model_id=model_id, timeout=60.0)
    except Exception as exc:
        print(f"[fail] Roboflow infer: {type(exc).__name__}: {exc}")
        return 5

    print("\nAll good — WSL can reach Roboflow. Restart uvicorn with keys set in .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
