"""FastAPI backend for HomePulse. Start with `python run.py`; API docs at http://127.0.0.1:8004/docs"""
import json

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .data import chronological_split, load_days
from .fhmm import FactorialHMM
from .metrics import f1_score

app = FastAPI(title="HomePulse API", version="1.0",
              description="Decode one smart-meter reading into the devices hidden inside it, with models written from scratch.")


def _load(name):
    path = config.ARTIFACTS / name
    if not path.exists():
        raise RuntimeError(f"{path} is missing. Run `python -m backend.train` first.")
    return json.loads(path.read_text())


REPORT = _load("report.json")
GRAPHS = _load("graphs.json")
MODEL = _load("model.json")
FH = FactorialHMM.from_state(MODEL["fhmm"])
HEATER = MODEL["heater_chain"]
DAYS = load_days()
SPLIT = chronological_split(DAYS)
TEST = DAYS.take(SPLIT.test)
DATE_INDEX = {d: i for i, d in enumerate(TEST.dates.tolist())}


def _round(a, n=3):
    return np.round(a, n).tolist()


def _kwh(a):
    return float(np.sum(a) / 60)


@app.get("/api/overview")
def overview():
    R, F = REPORT, REPORT["fhmm"]
    return {
        "data": R["data"], "energy": R["energy"], "decode": R["decode"], "restarts": R["restarts"], "rivals": R["rivals"],
        "budget": R["rival_detail"]["budget"], "limits": R["limits"], "week": R["week"], "routine": R["routine"], "empty_home": R["empty_home"],
        "monthly": R["monthly"], "per_day": {k: R["per_day_f1"][k] for k in ("p10", "median", "p90")},
        "kalman": {k: R["kalman"][k] for k in ("q", "r", "gain_final", "forecast", "decline", "missing_days")},
        "message_passing": {k: R["message_passing"][k] for k in ("max_gamma_diff", "same_viterbi_path", "ll_diff", "naive_breaks_at", "naive_direction")},
        "devices": [{"id": "ABCDE"[m], "kw": F["watts_kW"][m], "run_min": F["run_min"][m], "gap_min": F["gap_min"][m], "duty": F["duty"][m],
                     "kwh": R["energy"]["device_kwh"][m], "is_heater": m == HEATER} for m in range(F["M"])],
        "base_kw": F["base_kW"], "presets": R["presets"], "tariff": config.DEFAULT_TARIFF, "sources": config.SOURCES,
        "split_date": R["split_date"], "n_graphs": len(GRAPHS), "trained_at": R["trained_at"],
    }


@app.get("/api/days")
def day_list():
    """Every unseen day, with how well the decoder did on it."""
    return {"days": REPORT["days_index"], "presets": REPORT["presets"]}


@app.get("/api/decode")
def decode(date: str = Query(..., description="An unseen day, YYYY-MM-DD")):
    """Run the trained decoder on one unseen day. The answer key is returned for display only."""
    if date not in DATE_INDEX:
        raise HTTPException(404, f"{date} is not an unseen day in this dataset")
    i = DATE_INDEX[date]
    x = TEST.P[i]
    smooth, live, path = FH.smooth(x), FH.filtered(x), FH.viterbi(x)
    H, K, L = TEST.H[i], TEST.K[i], TEST.L[i]
    on = H >= config.ON_RULES["H"]
    pred = smooth[:, HEATER] > 0.5
    devices = []
    for m in range(FH.M):
        devices.append({"id": "ABCDE"[m], "kw": float(FH.watts[m]), "is_heater": m == HEATER, "p": _round(smooth[:, m]), "live": _round(live[:, m]),
                        "path": path[:, m].astype(int).tolist(), "kwh": _kwh(smooth[:, m] * FH.watts[m]), "minutes_on": int((smooth[:, m] > 0.5).sum())})
    return {
        "date": date, "total": _round(x), "base_kw": FH.base, "devices": devices,
        "answer": {"heater_kw": _round(H * 60 / 1000), "kitchen_kw": _round(K * 60 / 1000), "laundry_kw": _round(L * 60 / 1000),
                   "heater_kwh": float(H.sum() / 1000), "kitchen_kwh": float(K.sum() / 1000), "laundry_kwh": float(L.sum() / 1000)},
        "total_kwh": _kwh(x), "heater_est_kwh": _kwh(smooth[:, HEATER] * FH.watts[HEATER]),
        "f1": f1_score(pred, on) if (on.any() or pred.any()) else None, "heater_on_minutes": int(on.sum()),
    }


@app.get("/api/graphs")
def graphs():
    return {"graphs": GRAPHS}


@app.get("/api/graphs/{gid}")
def graph(gid: str):
    for g in GRAPHS:
        if g["id"] == gid:
            return g
    raise HTTPException(404, "no such graph")


@app.middleware("http")
async def revalidate_assets(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/assets") or request.url.path in ("/", "/graphs"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/assets", StaticFiles(directory=config.FRONTEND / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND / "index.html")


@app.get("/graphs", include_in_schema=False)
def graphs_page():
    return FileResponse(config.FRONTEND / "graphs.html")
