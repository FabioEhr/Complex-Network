import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from World_map import plot_world_map

def main():
    # 1) LOAD BASELINE FILES (2010–2020) FOR Clustering_Coefficient
    baseline_files = [f"clus_{year}.parquet" for year in range(2010, 2021)]
    baseline_dfs = {}

    for bf in baseline_files:
        try:
            df = pd.read_parquet(bf)
            # We expect columns: 'Node', 'Clustering_Coefficient'
            if 'Clustering_Coefficient' in df.columns and 'Node' in df.columns:
                # Convert scores to numeric, drop NaNs
                score_series = pd.to_numeric(df['Clustering_Coefficient'], errors='coerce')
                # Align index to 'Node'
                score_series.index = df['Node']

                # Report NaN and zero counts
                nan_score = score_series.isna().sum()
                zero_score = score_series.eq(0).sum()
                print(f"{bf} (Clustering_Coefficient): {nan_score} NaN, {zero_score} zeros")

                baseline_dfs[bf] = score_series
            else:
                print(f"Skipping {bf}: missing Clustering_Coefficient or Node columns.")
        except Exception as e:
            print(f"Error reading {bf}: {e}")

    # Ensure we have baseline data
    if not baseline_dfs:
        print("No baseline data loaded for Clustering_Coefficient.")
        return

    # Combine into DataFrame: columns are filenames, index are node labels
    baseline_df = pd.DataFrame(baseline_dfs)

    # 2) FILTER OUT NODES WITH ANY ZERO OR NaN IN baseline 2010–2020
    zero_or_nan_counts = (baseline_df == 0).sum(axis=1) + baseline_df.isna().sum(axis=1)
    discarded_nodes = set(zero_or_nan_counts[zero_or_nan_counts > 0].index)
    print(f"Discarding {len(discarded_nodes)} nodes due to zero or NaN in baseline.")

    # Drop those nodes
    baseline_df = baseline_df.drop(index=discarded_nodes, errors='ignore')

    # 3) COMPUTE σ (STD DEV) OVER 2010–2020 FOR EACH NODE
    sigma = baseline_df.std(axis=1)

    # Discard any nodes where σ == 0
    zero_sigma_nodes = sigma[sigma == 0].index
    print(f"{len(zero_sigma_nodes)} nodes with zero σ will be discarded.")

    # Final set of discarded nodes
    discarded_nodes = discarded_nodes.union(set(zero_sigma_nodes))
    print(f"Total discarded nodes after σ filter: {len(discarded_nodes)}")

    # Drop them from baseline table and σ
    baseline_df = baseline_df.drop(index=zero_sigma_nodes, errors='ignore')
    sigma = sigma.drop(index=zero_sigma_nodes, errors='ignore')

    # 4) EXTRACT 2020 BASELINE
    baseline_2020 = baseline_df.get("clus_2020.parquet")
    if baseline_2020 is None:
        print("2020 baseline data not found after filtering.")
        return

    # 5) LOAD POLICY FILES: bc, eu, gl
    policy_files = {
        "bc": "clus_bc.parquet",
        "eu": "clus_eu.parquet",
        "gl": "clus_gl.parquet"
    }

    results = {}
    central_policy = {}

    for policy, pf in policy_files.items():
        try:
            dfp = pd.read_parquet(pf)
            # Extract series and drop discarded nodes
            score_series_p = pd.to_numeric(dfp['Clustering_Coefficient'], errors='coerce')
            score_series_p.index = dfp['Node']

            # Drop discarded nodes
            score_series_p = score_series_p.drop(index=discarded_nodes, errors='ignore')

            # Report NaNs and zeros
            nan_score_p = score_series_p.isna().sum()
            zero_score_p = score_series_p.eq(0).sum()
            print(f"{pf} (Clustering_Coefficient): {nan_score_p} NaN, {zero_score_p} zeros")

            # Compute z-scores
            z_score = (score_series_p - baseline_2020) / sigma

            results[policy] = z_score
            central_policy[policy] = score_series_p

        except Exception as e:
            print(f"Error reading {pf}: {e}")

    # 6) COMPUTE NET INFLOWS PER NODE (USE dfz_2018.parquet AS BEFORE)
    df_net = pd.read_parquet('./dfz_2018.parquet')

    inflows = df_net.sum(axis=0)
    outflows = df_net.sum(axis=1) 
    net = -inflows + outflows 
    net_trillions = net / 1e9
    print(f"Net inflows (trillions) computed for {len(net_trillions)} nodes.")
    print(net_trillions.head())

    # 7) LISTE DI PAESI DI INTERESSE (COME PRIMA)
    countries = [
        "CHN","USA","RoW","JPN","DEU","FRA","IND","GBR","ITA","KOR","BRA","RUS",
        "AUS","CAN","ESP","TUR","MEX","IDN","NLD","CHE","POL","BEL","SWE","AUT",
        "CZE","NOR","FIN","DNK","ROU","IRL","PRT","GRC","HUN","SVK","LUX","BGR",
        "HRV","SVN","LTU","LVA","EST","CYP","MLT"
    ]

    # 8) CALCOLO WEIGHTED AVERAGES PER POLITICA
    weighted_avgs = {policy: {} for policy in results}

    for policy in results:
        for country in countries:
            num = 0.0
            den = 0.0
            for node in results[policy].index:
                if node.startswith(f"{country}_"):
                    weight = net_trillions.get(node, 0.0)
                    num += weight * results[policy].get(node, 0.0)
                    den += weight
            wavg = num / den if den != 0.0 else np.nan
            weighted_avgs[policy][country] = wavg

    # 9) PLOT MAPPE CON plot_world_map
    # Per ogni policy, plot Clustering_Coefficient z-scores
    for policy in weighted_avgs:
        df_plot = pd.DataFrame(list(weighted_avgs[policy].items()), columns=['ISO_A3', 'value'])
        plot_world_map(df_plot, '', '', f"Weighted Avg Clustering Coefficient Z-score - Policy '{policy}'")

if __name__ == "__main__":
    main()