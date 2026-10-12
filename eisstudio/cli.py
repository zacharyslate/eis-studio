"""``eis-studio`` command: start the app on this computer and open it in the browser.

The server listens on 127.0.0.1 only, so the interface (and any uploaded spectrum) is not
reachable from other machines on the network. Close the terminal window, or press Ctrl+C,
to stop it.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__

APP = Path(__file__).with_name("app.py")


def _free_port(start: int, host: str) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port found between {start} and {start + 49}.")


def _open_when_ready(url: str, host: str, port: int, timeout: float = 60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.3)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eis-studio", description="EIS analysis (ECM + DRT) in your browser.")
    ap.add_argument("--port", type=int, default=8501, help="first port to try (default 8501)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("--network", action="store_true",
                    help="listen on all interfaces so others on your network can connect (off by default)")
    ap.add_argument("--version", action="version", version=f"EIS Studio {__version__}")
    args = ap.parse_args(argv)

    host = "0.0.0.0" if args.network else "127.0.0.1"
    port = _free_port(args.port, "127.0.0.1" if host == "127.0.0.1" else "")
    url = f"http://localhost:{port}"
    print(f"EIS Studio {__version__} is starting at {url}")
    print("Keep this window open while you use it. Press Ctrl+C (or close the window) to quit.")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url, "127.0.0.1", port), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(APP),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",          # we open the browser ourselves; avoids Streamlit's e-mail prompt
        "--server.maxUploadSize", "50",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
        "--client.toolbarMode", "minimal",
        "--theme.base", "light",
        "--theme.primaryColor", "#0072B2",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
