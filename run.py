"""One command to train (if needed) and launch HomePulse.

    python run.py              # train on first run, then serve http://127.0.0.1:8004
    python run.py --retrain    # force a fresh training run
    python run.py --no-browser # do not open a browser tab
"""
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PORT = 8004

if __name__ == "__main__":
    if "--retrain" in sys.argv or not (ROOT / "artifacts" / "graphs.json").exists():
        from backend.train import main as train
        train()

    import uvicorn

    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  HomePulse running at {url}   (API docs: {url}/docs)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("backend.api:app", host="127.0.0.1", port=PORT, log_level="warning")
