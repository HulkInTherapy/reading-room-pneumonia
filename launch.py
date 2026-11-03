"""Start a local-only server on an available port and open the browser."""

import argparse
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def choose_port(preferred):
    for port in range(preferred, preferred + 100):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError(
        "No free local port found. Close an old demo window and try again."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from scripts.download_models import verify_models

    problems = verify_models()
    if problems:
        raise RuntimeError(
            "Model files are incomplete. Run setup_demo.py to download or repair them."
        )
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.chdir(ROOT)
    import uvicorn

    port = choose_port(args.port)
    url = "http://127.0.0.1:" + str(port)
    if not args.no_browser:

        def open_browser():
            import urllib.request

            for _ in range(90):
                try:
                    urllib.request.urlopen(url + "/api/status", timeout=1)
                    webbrowser.open(url)
                    return
                except OSError:
                    time.sleep(1)

        threading.Thread(target=open_browser, daemon=True).start()
    print(
        "\nReading Room: " + url + "\nKeep this window open. Press Ctrl+C to stop.\n",
        flush=True,
    )
    uvicorn.run("app:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - friendly errors at the CLI boundary
        print("The demo could not start: " + str(exc))
        raise SystemExit(1)
