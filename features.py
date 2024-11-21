import os
import sys
from collections import Counter
from datetime import datetime, date, timedelta

import shutil
import numpy as np
import pandas as pd

types = {
    "number_of_sources": np.uint16,
    "number_of_targets": np.uint16,
    "max_density_in": np.uint16,
    "max_density_out": np.uint16,
    "number_of_transactions": np.uint16,
    "number_of_currencies": np.uint8,
    "ts_range": np.uint32,
    "ts_std": np.uint32,
    "btc": np.float32,
    "ils": np.uint64,
    "eur": np.uint64,
    "jpy": np.uint64,
    "usd": np.uint64,
    "cad": np.uint64,
    "mxn": np.uint64,
    "cny": np.uint64,
    "gbp": np.uint64,
    "rub": np.uint64,
    "chf": np.uint64,
    "inr": np.uint64,
    "brl": np.uint64,
    "sar": np.uint64,
    "aud": np.uint64,
}
all_features_columns = set(types.keys())


def generate_features(df):
    max_density = lambda x: Counter(x).most_common()[0][1]
    amount_curr = lambda x: x.to_dict()["amount"]
    other_features = df.groupby("id").agg(
        number_of_sources=("source", "nunique"),
        number_of_targets=("target", "nunique"),
        max_density_in=("source", max_density),
        max_density_out=("target", max_density),
        number_of_transactions=("timestamp", "count"),
        number_of_currencies=("currency", "nunique"),
        ts_min=("timestamp", "min"),
        ts_max=("timestamp", "max"),
        ts_std=("timestamp", "std"),
    )
    amount_features = df.groupby(["id", "currency"]).agg(
        {"amount": "sum"}
    ).reset_index().set_index("currency").groupby("id")[["id", "amount"]].apply(amount_curr)
    amount_features = pd.DataFrame(amount_features.tolist(), index=amount_features.index).fillna(0)
    
    all_features = other_features.join(amount_features, how="left").fillna(0)
    all_features.loc[:, "ts_range"] = all_features.loc[:, "ts_max"] - all_features.loc[:, "ts_min"]
    del all_features["ts_min"]
    del all_features["ts_max"]
    available_columns = set(all_features.columns)
    missing_features = available_columns.symmetric_difference(all_features_columns)
    for missing in missing_features:
        all_features.loc[:, missing] = 0
    all_features = all_features.astype(types)
    return all_features


if __name__ == "__main__":
    part_file = sys.argv[1].strip()
    location_features = sys.argv[2].strip()
    date_first = sys.argv[3].strip().split("-")
    date_last = sys.argv[4].strip().split("-")
    date_first = date(*[int(x) for x in date_first])
    date_last = date(*[int(x) for x in date_last])
    dates = [x.date() for x in pd.date_range(date_first, date_last)]
    max_timestamp = int((max(dates) + timedelta(days=1)).strftime("%s")) * 1e3

    df_part = pd.read_parquet(part_file)
    part = os.path.basename(part_file).split(".")[0]

    window_sizes = [2, 3, 4, 8, 12, 15, 22, 29, 37, 72]
    max_date = datetime.combine(max(dates), datetime.min.time())
    for window_size in window_sizes:
        window_data = []
        for window in dates:
            window_start = datetime.combine(window, datetime.min.time())
            window_end = window_start + timedelta(days=window_size)
            window_start_ts = window_start.timestamp()
            window_end_ts = window_end.timestamp()
            if window_end > max_date:
                continue
            df_window = df_part.loc[
                (df_part["timestamp"] >= window_start_ts) & (df_part["timestamp"] < window_end_ts), :
            ].copy(deep=True)
            features = generate_features(df_window)
            features.loc[:, "window_size"] = np.uint8(window_size)
            features.loc[:, "day_number"] = np.uint8((window - date_first).days + 1)
            window_data.append(features.reset_index().astype({"id": np.uint32}))
        
        features_folder = f"{location_features}/window-{window_size}/"
        try:
            os.mkdir(features_folder)
        except FileExistsError:
            pass
        pd.concat(window_data, ignore_index=True).to_parquet(f"{features_folder}{part}.parquet")
    print(f"Done -> {part}")
