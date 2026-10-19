"""Start the KAL detection console:  python -m app   (or double-click RUN_APP.bat).

Serves the UI on http://127.0.0.1:8050 (this computer only) and opens it in a browser window.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]
os.environ.setdefault("KAL_CONFIG_OVERLAY", "configs/m1.yaml")
os.chdir(ROOT)

URL = "http://127.0.0.1:8050"


def open_window():
    """An app-style window (Microsoft Edge --app) when available, otherwise the default browser."""
    time.sleep(1.5)
    edge = shutil.which("msedge") or next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                                       r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
                                           if Path(p).exists()), None)
    if edge:
        subprocess.Popen([edge, f"--app={URL}", "--window-size=1600,960"])
    else:
        webbrowser.open(URL)


def main():
    from app.export import unpack_if_present
    from app.server import app
    unpack_if_present()
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_window, daemon=True).start()
    print(f"KAL detection console on {URL}  (close this window to stop)")
    app.run(host="127.0.0.1", port=8050, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
