import pandas as pd
import networkx as nx
import time
import random
import glob
import os

# -------------------------------
# 1–2. Loop over all sparse graphs, compute reciprocal-distance betweenness, and save
# -------------------------------
parquet_files = sorted(glob.glob("dfz_s_*.parquet"))
eps = 1e-12
for pq in parquet_files:
    print(f"\nProcessing {pq} …")
    # Load adjacency matrix
    df = pd.read_parquet(pq)
    df.index = df.columns
    print(f"Loaded DataFrame of shape {df.shape}")

    # Build directed graph
    G_directed = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
    print(f"Graph has {G_directed.number_of_nodes()} nodes & {G_directed.number_of_edges()} edges")

    # Convert flow to distance
    for u, v, data in G_directed.edges(data=True):
        flow = data.get("weight", 0)
        data["distance"] = 1.0 / (flow + eps)
    print("Assigned reciprocal-distance attribute to edges")

    # Compute betweenness
    start_time = time.time()
    bc = nx.betweenness_centrality(
        G_directed,
        normalized=True,
        weight="distance",
        endpoints=False
    )
    elapsed = time.time() - start_time
    print(f"Computed betweenness in {elapsed:.2f} seconds")

    # Save results as compressed Parquet
    bc_series = pd.Series(bc, name="betweenness")
    out_file = pq.replace(".parquet", "_bc.parquet")
    bc_series.to_frame().to_parquet(out_file, compression="gzip")
    print(f"Saved centrality to {out_file}")

    # Uncomment to read back:
    # df_bc = pd.read_parquet(out_file)
    # bc_series = df_bc["betweenness"]
