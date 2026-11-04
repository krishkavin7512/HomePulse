"""The running app: every route, on real trained artifacts."""
import pytest
from fastapi.testclient import TestClient

from backend.api import REPORT, app

client = TestClient(app)


def test_pages_load():
    assert client.get("/").status_code == 200
    assert client.get("/graphs").status_code == 200


def test_overview_reports_devices_and_rivals():
    r = client.get("/api/overview").json()
    assert len(r["devices"]) == REPORT["fhmm"]["M"]
    assert sum(d["is_heater"] for d in r["devices"]) == 1
    assert any(x["labels"] == "none" for x in r["rivals"]) and any(x["labels"] != "none" for x in r["rivals"])
    assert r["n_graphs"] >= 30


def test_decode_a_real_unseen_day():
    date = REPORT["presets"][0]["date"]
    r = client.get("/api/decode", params={"date": date}).json()
    assert r["date"] == date and len(r["total"]) == 1440
    for d in r["devices"]:
        assert len(d["p"]) == 1440 and len(d["live"]) == 1440 and len(d["path"]) == 1440
        assert all(0 <= v <= 1.0001 for v in d["p"])
    assert len(r["answer"]["heater_kw"]) == 1440


def test_decode_refuses_training_and_unknown_days():
    assert client.get("/api/decode", params={"date": "2007-06-01"}).status_code == 404
    assert client.get("/api/decode", params={"date": "nonsense"}).status_code == 404


@pytest.mark.parametrize("preset", REPORT["presets"], ids=lambda p: p["key"])
def test_every_preset_decodes(preset):
    assert client.get("/api/decode", params={"date": preset["date"]}).status_code == 200


def test_every_graph_is_explained():
    graphs = client.get("/api/graphs").json()["graphs"]
    assert len(graphs) == len({g["id"] for g in graphs})
    for g in graphs:
        assert g["figure"]["data"] and g["insight"]
        assert all(g["explain"][k] for k in ("what", "how", "read", "where"))


def test_single_graph_route():
    assert client.get("/api/graphs/raw-day").status_code == 200
    assert client.get("/api/graphs/nope").status_code == 404
