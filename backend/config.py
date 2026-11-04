"""Paths and fixed settings for HomePulse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"
DAYS_FILE = DATA / "days.npz"

MINUTES = 1440
SEED = 7

# Everything before this date can be used for training. Everything after it is unseen.
SPLIT_DATE = "2009-01-01"
N_TRAIN_DAYS = 200      # days the decoder learns from
N_VAL_DAYS = 60         # held-out training-era days, used to choose how many chains to keep

CHAIN_RANGE = (1, 2, 3, 4, 5)
EM_ITERS = 40
ELBOW_RULE = 0.20       # stop adding chains when the next one adds < 20% of the previous gain

# Answer key: what counts as "on" for each sub-meter (Wh per minute).
ON_RULES = {"H": 5, "K": 5, "L": 10}
METER_NAMES = {"H": "Water heater + AC", "K": "Kitchen", "L": "Laundry room"}
HEATER_NOTE = "sub-meter 3: electric water heater and air conditioner"

DEFAULT_TARIFF = 8.0    # Rs per kWh; illustrative, the app lets the viewer change it

SOURCES = {
    "uci": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
    "india_meters": "https://www.tndindia.com/indias-smart-meter-population-at-7-24-crore-parliament/amp/",
    "japan_services": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11175292/",
    "greveler": "https://www.nds.ruhr-uni-bochum.de/research/publications/SmartmeterContentIdent",
    "molina": "https://rts.cs.umass.edu/papers/pdf/buildsys10.pdf",
}
