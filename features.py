import os
import sys

import numpy as np
import pandas as pd
import igraph as ig


currency_rates = {
    "jpy": np.float32(0.009487665410827868),
    "cny": np.float32(0.14930721887033868),
    "cad": np.float32(0.7579775434031815),
    "sar": np.float32(0.2665884611958837),
    "aud": np.float32(0.7078143121927827),
    "ils": np.float32(0.29612081311363503),
    "chf": np.float32(1.0928961554056371),
    "usd": np.float32(1.0),
    "eur": np.float32(1.171783425225877),
    "rub": np.float32(0.012852809604990688),
    "gbp": np.float32(1.2916554735187644),
    "btc": np.float32(11879.132698717296),
    "inr": np.float32(0.013615817231245796),
    "mxn": np.float32(0.047296753463246695),
    "brl": np.float32(0.1771008654705292),
}

types = {
    "key": str,
    "num_source_or_target": np.uint16,
    "num_source_and_target": np.uint16,
    "num_source_only": np.uint16,
    "num_target_only": np.uint16,
    "num_transactions": np.uint16,
    "num_currencies": np.uint16,
    "num_source_or_target_bank": np.uint16,
    "num_source_and_target_bank": np.uint16,
    "num_source_only_bank": np.uint16,
    "num_target_only_bank": np.uint16,
    "turnover": np.uint64,
    "ts_range": np.uint32,
    "ts_std": np.float64,
    "assortativity_degree": np.float64,
    "max_degree": np.uint16,
    "max_degree_in": np.uint16,
    "max_degree_out": np.uint16,
    "diameter": np.uint8,
    "assortativity_degree_bank": np.float64,
    "max_degree_bank": np.uint16,
    "max_degree_in_bank": np.uint16,
    "max_degree_out_bank": np.uint16,
    "diameter_bank": np.uint8,
    "usd": np.float32,
    "btc": np.float32,
    "chf": np.float32,
    "gbp": np.float32,
    "inr": np.float32,
    "jpy": np.float32,
    "rub": np.float32,
    "aud": np.float32,
    "mxn": np.float32,
    "ils": np.float32,
    "cad": np.float32,
    "brl": np.float32,
    "sar": np.float32,
    "cny": np.float32,
    "eur": np.float32,
}


def get_segments(source_column, target_column, data_in):
    sources = set(data_in[source_column].unique())
    targets = set(data_in[target_column].unique())
    source_or_target = sources.union(targets)
    source_and_target = sources.intersection(targets)
    source_only = sources.difference(targets)
    target_only = targets.difference(sources)
    return source_or_target, source_and_target, source_only, target_only


def generate_features(df, group_id):
    source_or_target, source_and_target, source_only, target_only = get_segments(
        "source", "target", df
    )
    features_row = {
        "key": group_id,
        "num_source_or_target": len(source_or_target),
        "num_source_and_target": len(source_and_target),
        "num_source_only": len(source_only),
        "num_target_only": len(target_only),
        "num_transactions": df.shape[0],
        "num_currencies": df["currency"].nunique(),
    }
    source_or_target, source_and_target, source_only, target_only = get_segments(
        "source_bank", "target_bank", df
    )
    features_row["num_source_or_target_bank"] = len(source_or_target)
    features_row["num_source_and_target_bank"] = len(source_and_target)
    features_row["num_source_only_bank"] = len(source_only)
    features_row["num_target_only_bank"] = len(target_only)

    left = (
        df.loc[:, ["target", "currency", "amount"]]
        .rename(columns={"target": "source"})
        .groupby(["source", "currency"])
        .agg({"amount": "sum"})
    )
    right = df.groupby(["source", "currency"]).agg({"amount": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_left"] - result["amount"]
    turnover_currency = result[result["delta"] > 0].reset_index(drop=True)
    turnover_currency = (
        turnover_currency.groupby("currency").agg({"delta": "sum"}).to_dict()["delta"]
    )

    left = (
        df.loc[:, ["target", "amount_usd"]]
        .rename(columns={"target": "source"})
        .groupby("source")
        .agg({"amount_usd": "sum"})
    )
    right = df.groupby("source").agg({"amount_usd": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_usd_left"] - result["amount_usd"]
    turnover = float(result[result["delta"] > 0]["delta"].sum())
    turnover_currency_norm = {}
    for key, value in turnover_currency.items():
        turnover_currency_norm[key] = float((currency_rates[key] * value) / turnover)

    features_row["turnover"] = turnover
    features_row.update(turnover_currency_norm)

    features_row["ts_range"] = df["timestamp"].max() - df["timestamp"].min()
    features_row["ts_std"] = df["timestamp"].std()

    graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
    features_row["assortativity_degree"] = graph.assortativity_degree(directed=True)
    features_row["max_degree"] = max(graph.degree(mode="all"))
    features_row["max_degree_in"] = max(graph.degree(mode="in"))
    features_row["max_degree_out"] = max(graph.degree(mode="out"))
    features_row["diameter"] = graph.diameter(directed=True, unconn=True)

    graph = ig.Graph.DataFrame(
        df[["source_bank", "target_bank"]], use_vids=False, directed=True
    )
    features_row["assortativity_degree_bank"] = graph.assortativity_degree(
        directed=True
    )
    features_row["max_degree_bank"] = max(graph.degree(mode="all"))
    features_row["max_degree_in_bank"] = max(graph.degree(mode="in"))
    features_row["max_degree_out_bank"] = max(graph.degree(mode="out"))
    features_row["diameter_bank"] = graph.diameter(directed=True, unconn=True)

    return features_row


if __name__ == "__main__":
    part_file = sys.argv[1].strip()
    part = os.path.basename(part_file).split(".")[0]
    location_features = sys.argv[2].strip()

    df_part = pd.read_parquet(part_file)
    features_all = []
    for key_, group in df_part.groupby("id"):
        features_all.append(generate_features(group, key_))

    pd.DataFrame(features_all).astype(types).to_parquet(
        f"{location_features}{os.sep}{part}.parquet"
    )
