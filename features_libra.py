import os
import sys

import numpy as np
import pandas as pd
import igraph as ig


types = {
    "key": str,
    "num_source_or_target": np.uint16,
    "num_source_and_target": np.uint16,
    "num_source_only": np.uint16,
    "num_target_only": np.uint16,
    "num_edges": np.uint32,
    "num_transactions": np.uint32,
    "turnover": np.uint64,
    "assortativity_degree": np.float64,
    "max_degree": np.uint16,
    "max_degree_in": np.uint16,
    "max_degree_out": np.uint16,
    "diameter": np.uint8,
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
        "num_edges": df.shape[0],
        "num_transactions": df["transactions_count"].sum(),
    }

    left = (
        df.loc[:, ["target", "amount"]]
        .rename(columns={"target": "source"})
        .groupby("source")
        .agg({"amount": "sum"})
    )
    right = df.groupby("source").agg({"amount": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_left"] - result["amount"]
    turnover = int(result[result["delta"] > 0]["delta"].sum())

    features_row["turnover"] = turnover

    graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
    features_row["assortativity_degree"] = graph.assortativity_degree(directed=True)
    features_row["max_degree"] = max(graph.degree(mode="all"))
    features_row["max_degree_in"] = max(graph.degree(mode="in"))
    features_row["max_degree_out"] = max(graph.degree(mode="out"))
    features_row["diameter"] = graph.diameter(directed=True, unconn=True)

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
