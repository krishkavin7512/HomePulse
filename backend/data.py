"""Load the household meter data as whole days, split by time.

data/days.npz holds every complete day from the UCI "Individual Household Electric
Power Consumption" dataset: total power in watts plus three sub-meters in Wh/min.
Models only ever see the total. The sub-meters are the answer key used for scoring.

`build_days` recreates that file from the original 127 MB text download:
    python -m backend.data path/to/household_power_consumption.txt
"""
import sys
from dataclasses import dataclass

import numpy as np

from . import config


def build_days(txt_path, out_path=config.DAYS_FILE):
    import pandas as pd
    cols = ["Date", "Time", "Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = pd.read_csv(txt_path, sep=";", na_values="?", low_memory=False, usecols=cols)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    day = ts.dt.normalize()
    counts = df["Global_active_power"].notna().groupby(day).sum()
    full = counts[counts == config.MINUTES].index
    d = df[day.isin(full)]
    P = np.round(d["Global_active_power"].to_numpy(float) * 1000).astype(np.uint16).reshape(-1, config.MINUTES)
    K, L, H = (d[c].to_numpy(float).round().astype(np.uint8).reshape(-1, config.MINUTES)
               for c in ("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"))
    config.DATA.mkdir(exist_ok=True)
    np.savez_compressed(out_path, P=P, K=K, L=L, H=H, dates=np.array(full.strftime("%Y-%m-%d").tolist(), dtype="U10"),
                        raw_minutes=len(df), raw_missing=int(df["Global_active_power"].isna().sum()))
    return len(full)


@dataclass
class Days:
    P: np.ndarray       # kW, days x 1440: the only thing the models see
    K: np.ndarray       # Wh/min kitchen sub-meter (answer key)
    L: np.ndarray       # Wh/min laundry-room sub-meter (answer key)
    H: np.ndarray       # Wh/min water-heater + AC sub-meter (answer key)
    dates: np.ndarray
    raw_minutes: int
    raw_missing: int

    def __len__(self):
        return len(self.dates)

    def take(self, idx):
        return Days(self.P[idx], self.K[idx], self.L[idx], self.H[idx], self.dates[idx], self.raw_minutes, self.raw_missing)


def load_days():
    with np.load(config.DAYS_FILE) as z:
        return Days(z["P"].astype(np.float64) / 1000, z["K"].astype(np.float32), z["L"].astype(np.float32), z["H"].astype(np.float32),
                    z["dates"], int(z["raw_minutes"]), int(z["raw_missing"]))


@dataclass
class Split:
    train: np.ndarray   # day indices the decoder learns from
    val: np.ndarray     # held-out days from the same era, for choosing chains
    test: np.ndarray    # every day after the split date, never touched in training
    era: np.ndarray     # every training-era day


def chronological_split(days, seed=config.SEED):
    """Train and validation come from before the split date, test from after it."""
    era = np.where(days.dates < config.SPLIT_DATE)[0]
    test = np.where(days.dates >= config.SPLIT_DATE)[0]
    rng = np.random.default_rng(seed)
    pick = rng.permutation(era)
    train = np.sort(pick[:config.N_TRAIN_DAYS])
    val = np.sort(pick[config.N_TRAIN_DAYS:config.N_TRAIN_DAYS + config.N_VAL_DAYS])
    return Split(train, val, test, era)


if __name__ == "__main__":
    print("complete days written:", build_days(sys.argv[1]))
