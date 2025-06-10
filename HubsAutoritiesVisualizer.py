import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from World_map import plot_world_map

def main():
    # 1) LOAD BASELINE FILES (2010–2020) FOR hub_score AND authority_score
    baseline_files = [f"hub_aut_{year}.parquet" for year in range(2010, 2021)]
    baseline_hub_dfs = {}
    baseline_auth_dfs = {}

    for bf in baseline_files:
        try:
            df = pd.read_parquet(bf)
            # We expect columns: 'node', 'hub_score', 'authority_score'
            if 'hub_score' in df.columns and 'authority_score' in df.columns:
                # Convert scores to numeric, drop NaNs
                hub_series = pd.to_numeric(df['hub_score'], errors='coerce')
                auth_series = pd.to_numeric(df['authority_score'], errors='coerce')
                # Align index to 'node'
                hub_series.index = df['node']
                auth_series.index = df['node']

                # Report NaN and zero counts
                nan_hub = hub_series.isna().sum()
                zero_hub = hub_series.eq(0).sum()
                print(f"{bf} (hub_score): {nan_hub} NaN, {zero_hub} zeros")

                nan_auth = auth_series.isna().sum()
                zero_auth = auth_series.eq(0).sum()
                print(f"{bf} (authority_score): {nan_auth} NaN, {zero_auth} zeros")

                baseline_hub_dfs[bf] = hub_series
                baseline_auth_dfs[bf] = auth_series
            else:
                print(f"Skipping {bf}: missing hub_score or authority_score columns.")
        except Exception as e:
            print(f"Error reading {bf}: {e}")

    # Ensure we have baseline data
    if not baseline_hub_dfs or not baseline_auth_dfs:
        print("No baseline data loaded for hub or authority scores.")
        return

    # Combine into DataFrames: columns are filenames, index are node labels
    baseline_hub_df = pd.DataFrame(baseline_hub_dfs)
    baseline_auth_df = pd.DataFrame(baseline_auth_dfs)

    # 2) FILTER OUT NODES WITH ANY ZERO OR NaN IN baseline 2010–2020
    # Count zeros or NaNs per row for each score table
    hub_zero_counts = (baseline_hub_df == 0).sum(axis=1) + baseline_hub_df.isna().sum(axis=1)
    auth_zero_counts = (baseline_auth_df == 0).sum(axis=1) + baseline_auth_df.isna().sum(axis=1)

    # Identify nodes to discard if they have ANY zero or NaN across baseline years
    discarded_hub_nodes = set(hub_zero_counts[hub_zero_counts > 0].index)
    discarded_auth_nodes = set(auth_zero_counts[auth_zero_counts > 0].index)
    # Union of both to discard if either score is invalid
    discarded_nodes = discarded_hub_nodes.union(discarded_auth_nodes)
    print(f"Discarding {len(discarded_nodes)} nodes due to zero or NaN in baseline.")

    # Drop those nodes
    baseline_hub_df = baseline_hub_df.drop(index=discarded_nodes, errors='ignore')
    baseline_auth_df = baseline_auth_df.drop(index=discarded_nodes, errors='ignore')

    # 3) COMPUTE σ (STD DEV) OVER 2010–2020 FOR EACH NODE, FOR BOTH SCORES
    sigma_hub = baseline_hub_df.std(axis=1)
    sigma_auth = baseline_auth_df.std(axis=1)

    # Discard any nodes where σ == 0
    zero_sigma_hub = sigma_hub[sigma_hub == 0].index
    zero_sigma_auth = sigma_auth[sigma_auth == 0].index
    print(f"{len(zero_sigma_hub)} hub nodes with zero σ will be discarded.")
    print(f"{len(zero_sigma_auth)} authority nodes with zero σ will be discarded.")
    zero_sigma_nodes = set(zero_sigma_hub).union(set(zero_sigma_auth))

    # Final set of discarded nodes
    discarded_nodes = discarded_nodes.union(zero_sigma_nodes)
    print(f"Total discarded nodes after σ filter: {len(discarded_nodes)}")

    # Drop them from baseline tables and σ
    baseline_hub_df = baseline_hub_df.drop(index=zero_sigma_nodes, errors='ignore')
    baseline_auth_df = baseline_auth_df.drop(index=zero_sigma_nodes, errors='ignore')
    sigma_hub = sigma_hub.drop(index=zero_sigma_nodes, errors='ignore')
    sigma_auth = sigma_auth.drop(index=zero_sigma_nodes, errors='ignore')

    # 4) EXTRACT 2020 BASELINE FOR BOTH SCORES
    hub_2020 = baseline_hub_df.get("hub_aut_2020.parquet")
    auth_2020 = baseline_auth_df.get("hub_aut_2020.parquet")
    if hub_2020 is None or auth_2020 is None:
        print("2020 baseline data not found after filtering.")
        return

    # 5) LOAD POLICY FILES: bc, eu, gl
    policy_files = {
        "bc": "hub_aut_bc.parquet",
        "eu": "hub_aut_eu.parquet",
        "gl": "hub_aut_gl.parquet"
    }

    results_hub = {}
    results_auth = {}
    central_policy_hub = {}
    central_policy_auth = {}

    for policy, pf in policy_files.items():
        try:
            dfp = pd.read_parquet(pf)
            # Extract series and drop discarded nodes
            hub_series_p = pd.to_numeric(dfp['hub_score'], errors='coerce')
            auth_series_p = pd.to_numeric(dfp['authority_score'], errors='coerce')
            hub_series_p.index = dfp['node']
            auth_series_p.index = dfp['node']

            # Drop discarded nodes
            hub_series_p = hub_series_p.drop(index=discarded_nodes, errors='ignore')
            auth_series_p = auth_series_p.drop(index=discarded_nodes, errors='ignore')

            # Report NaNs and zeros
            nan_hub_p = hub_series_p.isna().sum()
            zero_hub_p = hub_series_p.eq(0).sum()
            print(f"{pf} (hub_score): {nan_hub_p} NaN, {zero_hub_p} zeros")

            nan_auth_p = auth_series_p.isna().sum()
            zero_auth_p = auth_series_p.eq(0).sum()
            print(f"{pf} (authority_score): {nan_auth_p} NaN, {zero_auth_p} zeros")

            # Compute z-scores for both scores
            z_hub = (hub_series_p - hub_2020) / sigma_hub
            z_auth = (auth_series_p - auth_2020) / sigma_auth

            results_hub[policy] = z_hub
            results_auth[policy] = z_auth
            central_policy_hub[policy] = hub_series_p
            central_policy_auth[policy] = auth_series_p

        except Exception as e:
            print(f"Error reading {pf}: {e}")

    # 6) COMPUTE NET INFLOWS PER NODE (USE dfz_2018.parquet AS BEFORE)
    df_net = pd.read_parquet('./dfz_2018.parquet')
    inflows = df_net.sum(axis=0) - pd.Series(df_net.values.diagonal(), index=df_net.index)
    outflows = df_net.sum(axis=1)
    net =outflows- inflows 
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

    # 8) CALCOLO WEIGHTED AVERAGES PER POLITICA E PER TIPO DI SCORE
    weighted_avgs_hub = {policy: {} for policy in results_hub}
    weighted_avgs_auth = {policy: {} for policy in results_auth}

    for policy in results_hub:
        # hub
        for country in countries:
            num_h = 0.0
            den_h = 0.0
            for node in results_hub[policy].index:
                if node.startswith(f"{country}_"):
                    weight = net_trillions.get(node, 0.0)
                    #weight=1
                    num_h += weight * results_hub[policy].get(node, 0.0)
                    den_h += weight
            wavg_h = num_h / den_h if den_h != 0.0 else np.nan
            weighted_avgs_hub[policy][country] = wavg_h

        # authority
        for country in countries:
            num_a = 0.0
            den_a = 0.0
            for node in results_auth[policy].index:
                if node.startswith(f"{country}_"):
                    weight = net_trillions.get(node, 0.0)
                    num_a += weight * results_auth[policy].get(node, 0.0)
                    den_a += weight
            wavg_a = num_a / den_a if den_a != 0.0 else np.nan
            weighted_avgs_auth[policy][country] = wavg_a

    # 9) PLOT MAPPE CON plot_world_map
    # Per ogni policy, plot hub_score z-scores e authority_score z-scores
    for policy in weighted_avgs_hub:
        df_plot_hub = pd.DataFrame(list(weighted_avgs_hub[policy].items()), columns=['ISO_A3', 'value'])
        plot_world_map(df_plot_hub, '', '', f"Weighted Avg Hub Z-score - Policy '{policy}'")

        df_plot_auth = pd.DataFrame(list(weighted_avgs_auth[policy].items()), columns=['ISO_A3', 'value'])
        plot_world_map(df_plot_auth, '', '', f"Weighted Avg Authority Z-score - Policy '{policy}'")

if __name__ == "__main__":
    main()