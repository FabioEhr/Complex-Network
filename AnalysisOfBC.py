import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from World_map import plot_world_map

def main():
    # Load baseline years from 2010 to 2020
    baseline_files = [f"dfz_s_{year}_bc.parquet" for year in range(2010, 2021)]
    baseline_dfs = {}
    for bf in baseline_files:
        try:
            df = pd.read_parquet(bf)
            if 'betweenness' in df.columns:
                series = pd.to_numeric(df['betweenness'], errors='coerce')
                nan_count = series.isna().sum()
                print(f"{bf}: {nan_count} NaN values in baseline centrality series first")
                zero_count = series.eq(0).sum()
                print(f"{bf}: {zero_count} zero values in baseline centrality series")
            else:
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                if len(numeric_cols) == 1:
                    series = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
                    nan_count = series.isna().sum()
                    print(f"{bf}: {nan_count} NaN values in baseline centrality series second")
                    zero_count = series.eq(0).sum()
                    print(f"{bf}: {zero_count} zero values in baseline centrality series")
                else:
                    print(f"Skipping baseline file {bf}: cannot identify centrality column.")
                    continue
            series.name = bf
            baseline_dfs[bf] = series
        except Exception as e:
            print(f"Error reading {bf}: {e}")
    if not baseline_dfs:
        print("No baseline data loaded.")
        return

    baseline_df = pd.DataFrame(baseline_dfs)
    # Discard country sectors with at least 5 zero centrality values between 2010-2020
    zero_counts = (baseline_df == 0).sum(axis=1)
    baseline_df = baseline_df[zero_counts < 1]

    sigma = baseline_df.std(axis=1)
    # Discard country sectors with sigma almost zero
    zero_sigma_count = (sigma == 0).sum()
    print(f"{zero_sigma_count} country sectors have sigma = 0 and will be discarded")
    nonzero_sigma = sigma != 0
    baseline_df = baseline_df[nonzero_sigma]
    sigma = sigma[nonzero_sigma]

    # Keep track of all discarded country sectors
    discarded_sectors = set(zero_counts[zero_counts >= 1].index).union(set(nonzero_sigma[~nonzero_sigma].index))
    print(f"Total discarded sectors due to zero values or zero sigma: {len(discarded_sectors)}")

    # Remove discarded sectors from baseline_df to ensure bc_2020 is also clean
    baseline_df = baseline_df.drop(index=discarded_sectors, errors='ignore')

    bc_2020 = baseline_df.get("dfz_s_2020_bc.parquet")
    if bc_2020 is None:
        print("2020 baseline data not found after filtering.")
        return

    policy_files = {
        "gl": "dfz_s_gl_bc.parquet",
        "eu": "dfz_s_eu_bc.parquet",
        "cbam": "dfz_s_bc_bc.parquet"
    }
    results = {}
    central_policy = {}
    for policy, pf in policy_files.items():
        try:
            dfp = pd.read_parquet(pf)
            if 'betweenness' in dfp.columns:
                series_p = pd.to_numeric(dfp['betweenness'], errors='coerce')
                series_p = series_p.drop(index=discarded_sectors, errors='ignore')
                nan_count_p = series_p.isna().sum()
                print(f"{pf}: {nan_count_p} NaN values in policy '{policy}' centrality series")
                zero_count_p = series_p.eq(0).sum()
                print(f"{pf}: {zero_count_p} zero values in policy '{policy}' centrality series")
                central_policy[policy] = series_p
            else:
                numeric_cols = dfp.select_dtypes(include='number').columns.tolist()
                if len(numeric_cols) == 1:
                    series_p = pd.to_numeric(dfp[numeric_cols[0]], errors='coerce')
                    series_p = series_p.drop(index=discarded_sectors, errors='ignore')
                    nan_count_p = series_p.isna().sum()
                    print(f"{pf}: {nan_count_p} NaN values in policy '{policy}' centrality series")
                    zero_count_p = series_p.eq(0).sum()
                    print(f"{pf}: {zero_count_p} zero values in policy '{policy}' centrality series")
                    central_policy[policy] = series_p
                else:
                    print(f"Skipping policy file {pf}: cannot identify centrality column.")
                    continue
            z = (series_p - bc_2020) / sigma
            results[policy] = z
        except Exception as e:
            print(f"Error reading {pf}: {e}")

    # --- Compute net inflows per country from the 2017 matrix ---
    df = pd.read_parquet('./dfz_2018.parquet')

    # Compute total inflows per country (sum of each column)
    inflows = df.sum(axis=0)

    # Compute total outflows per country (sum of each row, excluding the diagonal)
    # i.e., subtract the self-flow df.loc[c, c] from each row sum
    outflows = df.sum(axis=1) - pd.Series(df.values.diagonal(), index=df.index)

    # Net inflow = inflows − outflows
    net = inflows - outflows

    # Convert to trillions of dollars
    net_trillions = net / 1e9
    print(f"Net inflows (in trillions) computed for {len(net_trillions)} sectors.")
    print(net_trillions.head())
    # List of countries of interest
    countries = [
        "CHN","USA","RoW","JPN","DEU","FRA","IND","GBR","ITA","KOR","BRA","RUS",
        "AUS","CAN","ESP","TUR","MEX","IDN","NLD","CHE","POL","BEL","SWE","AUT",
        "CZE","NOR","FIN","DNK","ROU","IRL","PRT","GRC","HUN","SVK","LUX","BGR",
        "HRV","SVN","LTU","LVA","EST","CYP","MLT"
    ]

    # Compute weighted average of raw centrality differences for each policy and country
    # Raw difference: central_policy[p] - bc_2020
    weighted_avgs = {policy: {} for policy in central_policy}
    for policy, series_p in central_policy.items():
        for country in countries:
            # Identify non-discarded sectors for this country
            num = 0.0
            den = 0.0
            for s in series_p.index:
                if s.startswith(f"{country}_"):
                    num += net_trillions.get(s, 0.0) * results[policy].get(s, 0.0)
                    den += net_trillions.get(s, 0.0)
                    #num += 1 * (series_p[s] - bc_2020[s])
                    #den += 1
            wavg = num / den if den != 0.0 else np.nan
            weighted_avgs[policy][country] = wavg
    
    for policy in weighted_avgs:
        print(weighted_avgs[policy])
        # Converti il dict in DataFrame con colonne ['ISO_A3', 'value']
        dict_data = weighted_avgs[policy]
        df_plot = pd.DataFrame(list(dict_data.items()), columns=['ISO_A3', 'value'])
        # Chiama la funzione usando il DataFrame appena creato
        plot_world_map(df_plot, '', '', f"Weighted Avg. Centrality Diff. - Policy '{policy}'")


    if results:
        result_df = pd.DataFrame(results)
        print("Z-scores for policies (columns: gl, eu, cbam):")
        print(result_df)
        # For each policy scenario, print top 20 increases and decreases
        for policy in result_df.columns:
            na_discarded = result_df[policy].isna().sum()
            print(f"{na_discarded} NaN values discarded for policy '{policy}'")
            total = result_df.shape[0]
            remaining = total - na_discarded
            print(f"{remaining} sectors remaining after discarding NaN for policy '{policy}' (out of {total})")

            # Compute average centrality among discarded sectors
            discarded_idx = result_df.index[result_df[policy].isna()]
            central = central_policy[policy]
            avg_disc = central.reindex(discarded_idx).dropna().mean()
            print(f"Average centrality in policy '{policy}' of discarded sectors: {avg_disc:.6f}")

            # Identify discarded sector with highest and lowest centrality
            central_disc = central.reindex(discarded_idx).dropna()
            if not central_disc.empty:
                max_idx = central_disc.idxmax()
                max_val = central_disc.max()
                min_idx = central_disc.idxmin()
                min_val = central_disc.min()
                sigma_val = sigma.loc[max_idx]
                bc2020_val = bc_2020.loc[max_idx]
                print(f"Discarded sector with highest centrality in '{policy}': {max_idx}: centrality={max_val:.6f}, sigma={sigma_val:.6f}, central_2020={bc2020_val:.6f}")
                print(f"Discarded sector with lowest centrality in '{policy}': {min_idx}: {min_val:.10f}")

            sorted_series = result_df[policy].dropna().sort_values(ascending=False)
            top20 = sorted_series.head(20)
            bottom20 = sorted_series.tail(20)
            print(f"\nPolicy {policy} - Top 20 increases:")
            for node, val in top20.items():
                print(f"  {node}: {val:.6f}")
            print(f"\nPolicy {policy} - Top 20 decreases:")
            for node, val in bottom20.items():
                print(f"  {node}: {val:.6f}")
        result_df.to_csv("centrality_policy_zscores.csv")
        print("Results saved to centrality_policy_zscores.csv")
    else:
        print("No policy results computed.")

if __name__ == "__main__":
    main()