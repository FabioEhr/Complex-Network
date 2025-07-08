import os
import sys
# Add project root to module search path so Code_and_dataset can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ensure results directory exists
os.makedirs('results', exist_ok=True)

SIMPLE_AVG = True

# Set of EU member country codes
EU_MEMBERS = {
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
    "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU",
    "SVK","SVN","ESP","SWE"
}

def compute_simple_average(series, countries):
    simple_avgs = {}
    for country in countries:
        nodes = [node for node in series.index if node.startswith(f"{country}_")]
        if nodes:
            values = np.array([series.get(node, np.nan) for node in nodes], dtype=float)
            # Replace NaNs with zeros
            values = np.nan_to_num(values, nan=0.0)
            simple_avgs[country] = np.sum(values)
        else:
            simple_avgs[country] = np.nan
    return simple_avgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Code_and_dataset.World_map import plot_world_maps_grid

def compute_weighted_average(series, net_trillions, countries):
    weighted_avgs = {}
    for country in countries:
        nodes = [node for node in series.index if node.startswith(f"{country}_")]
        weights = np.array([net_trillions.get(node, 0.0) for node in nodes], dtype=float)
        values = np.array([series.get(node, np.nan) for node in nodes], dtype=float)
        total_weight = np.nansum(weights)
        if total_weight != 0:
            weighted_avgs[country] = np.nansum(weights * values) / total_weight
        else:
            weighted_avgs[country] = np.nan
    return weighted_avgs

def load_net_trillions(filepath):
    df_net = pd.read_parquet(filepath)
    inflows = df_net.sum(axis=0) - pd.Series(df_net.values.diagonal(), index=df_net.index)
    outflows = df_net.sum(axis=1)
    net = outflows - inflows
    return net / 1e9


def create_and_save_map_plot(metric_name, baseline_vals, bc_vals, eu_vals, gl_vals, output_filename):
    """Create a 2×2 world‑map figure: baseline metric (upper‑left) and rank‑change maps for BC, EU, GL."""
    import pandas as pd
    import numpy as np

    # Convert dictionaries to Series for easy ranking
    s_baseline = pd.Series(baseline_vals, dtype=float)
    s_bc       = pd.Series(bc_vals,       dtype=float)
    s_eu       = pd.Series(eu_vals,       dtype=float)
    s_gl       = pd.Series(gl_vals,       dtype=float)

    # --- panel 1: actual metric values in 2020 (baseline) ---
    df_baseline = s_baseline.reset_index()
    df_baseline.columns = ["ISO_A3", "value"]

    # Helper: compute rank‑change (baseline rank – new rank)
    def _rank_delta(series_policy):
        baseline_rank = (-s_baseline).rank(method="min")  # descending
        policy_rank   = (-series_policy ).rank(method="min")
        delta = baseline_rank - policy_rank
        return delta.reset_index().rename(columns={"index": "ISO_A3", 0: "value"})

    df_delta_bc = _rank_delta(s_bc)
    df_delta_eu = _rank_delta(s_eu)
    df_delta_gl = _rank_delta(s_gl)

    # Assemble dataframes and labels for the 2×2 grid
    dfs    = [df_baseline, df_delta_bc, df_delta_eu, df_delta_gl]
    titles = [
        f"{metric_name} – Baseline 2020",
        "Δ Rank: EU tax + CBAM",
        "Δ Rank: EU‑only tax",
        "Δ Rank: Global tax"
    ]

    # Blank axis labels – they are removed inside plot_world_maps_grid
    x_lbls = y_lbls = ["", "", "", ""]

    plot_world_maps_grid(dfs, x_lbls, y_lbls, titles, nrows=2, ncols=2, figsize=(16, 8))

    # Optionally save the figure created by plot_world_maps_grid (the helper already calls plt.show()).
    # A simple workaround is to call plt.savefig after the grid is drawn.
    import matplotlib.pyplot as _plt
    _plt.savefig(output_filename, bbox_inches="tight")
    _plt.close()

# New function for percent change map plot
def create_and_save_percent_change_map_plot(metric_name, baseline_vals, bc_vals, eu_vals, gl_vals, output_filename):
    import pandas as pd
    import numpy as np
    # Compute percent changes relative to baseline
    pct_bc = {
        country: ((bc_vals[country] - baseline_vals[country]) / baseline_vals[country] * 100)
        if baseline_vals[country] not in (0, None, np.nan) else np.nan
        for country in baseline_vals
    }
    pct_eu = {
        country: ((eu_vals[country] - baseline_vals[country]) / baseline_vals[country] * 100)
        if baseline_vals[country] not in (0, None, np.nan) else np.nan
        for country in baseline_vals
    }
    pct_gl = {
        country: ((gl_vals[country] - baseline_vals[country]) / baseline_vals[country] * 100)
        if baseline_vals[country] not in (0, None, np.nan) else np.nan
        for country in baseline_vals
    }

    # Convert to Series and DataFrames
    s_bc = pd.Series(pct_bc, dtype=float)
    s_eu = pd.Series(pct_eu, dtype=float)
    s_gl = pd.Series(pct_gl, dtype=float)

    dfs = []
    for series_pct in (s_bc, s_eu, s_gl):
        df = series_pct.reset_index()
        df.columns = ["ISO_A3", "value"]
        dfs.append(df)

    titles = [
        f"% Change: {metric_name} – EU tax + CBAM",
        f"% Change: {metric_name} – EU‑only tax",
        f"% Change: {metric_name} – Global tax"
    ]
    x_lbls = y_lbls = [""] * 3

    # Plot a 1×3 grid of percent-change maps
    plot_world_maps_grid(dfs, x_lbls, y_lbls, titles, nrows=1, ncols=3, figsize=(18, 6))

    import matplotlib.pyplot as _plt
    _plt.savefig(output_filename, bbox_inches="tight")
    _plt.close()

def process_clustering(countries, net_trillions):
    # 1) Load baseline files (2010-2020) for Clustering Coefficient
    baseline_dfs = {}
    for year in range(2010, 2021):
        bf = f"Dataset/clus_{year}.parquet"
        df = pd.read_parquet(bf)
        if {'Node', 'Clustering_Coefficient'}.issubset(df.columns):
            series = pd.to_numeric(df['Clustering_Coefficient'], errors='coerce')
            series.index = df['Node']
            baseline_dfs[year] = series
    baseline_df = pd.DataFrame(baseline_dfs).dropna(how='any', axis=0)
    baseline_2020 = baseline_df[2020]

    # 2) Load policy files
    policy_series = {}
    for policy, pf in [('bc', 'Dataset/clus_bc.parquet'), ('eu', 'Dataset/clus_eu.parquet'), ('gl', 'Dataset/clus_gl.parquet')]:
        dfp = pd.read_parquet(pf)
        series = pd.to_numeric(dfp['Clustering_Coefficient'], errors='coerce')
        series.index = dfp['Node']
        policy_series[policy] = series

    # 3) Compute averages per country (weighted or simple)
    if SIMPLE_AVG:
        weighted_baseline = compute_simple_average(baseline_2020, countries)
        weighted_bc = compute_simple_average(policy_series['bc'], countries)
        weighted_eu = compute_simple_average(policy_series['eu'], countries)
        weighted_gl = compute_simple_average(policy_series['gl'], countries)
    else:
        weighted_baseline = compute_weighted_average(baseline_2020, net_trillions, countries)
        weighted_bc = compute_weighted_average(policy_series['bc'], net_trillions, countries)
        weighted_eu = compute_weighted_average(policy_series['eu'], net_trillions, countries)
        weighted_gl = compute_weighted_average(policy_series['gl'], net_trillions, countries)

    # Check for negative weighted averages in clustering
    for label, weighted in [
        ("Baseline", weighted_baseline),
        ("BC", weighted_bc),
        ("EU", weighted_eu),
        ("GL", weighted_gl),
    ]:
        negatives = [country for country, value in weighted.items() if value is not None and value < 0]
        if negatives:
            print(f"Warning: Negative clustering averages in {label} for countries: {negatives}")

    create_and_save_percent_change_map_plot('Clustering Coefficient', weighted_baseline, weighted_bc, weighted_eu, weighted_gl, 'results/clustering_percent_change_maps.png')

def process_betweenness(countries, net_trillions):
    # 1) Load baseline files (2010-2020) for Betweenness Centrality
    baseline_dfs = {}
    for year in range(2010, 2021):
        bf = f"Dataset/dfz_s_{year}_bc.parquet"
        df = pd.read_parquet(bf)
        # Identify column
        if 'betweenness' in df.columns:
            series = pd.to_numeric(df['betweenness'], errors='coerce')
        else:
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) == 1:
                series = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
            else:
                continue
        series.index = df.index if 'betweenness' in df.columns else df[ df.index.name ]
        baseline_dfs[year] = series
    baseline_df = pd.DataFrame(baseline_dfs).dropna(how='any', axis=0)
    baseline_2020 = baseline_df[2020]

    # 2) Load policy files
    policy_series = {}
    for policy, pf in [('bc', 'Dataset/dfz_s_bc_bc.parquet'), ('eu', 'Dataset/dfz_s_eu_bc.parquet'), ('gl', 'Dataset/dfz_s_gl_bc.parquet')]:
        dfp = pd.read_parquet(pf)
        if 'betweenness' in dfp.columns:
            series = pd.to_numeric(dfp['betweenness'], errors='coerce')
        else:
            numeric_cols = dfp.select_dtypes(include='number').columns
            if len(numeric_cols) == 1:
                series = pd.to_numeric(dfp[numeric_cols[0]], errors='coerce')
            else:
                continue
        series.index = dfp.index if 'betweenness' in dfp.columns else dfp[ dfp.index.name ]
        policy_series[policy] = series

    # 3) Compute averages per country (weighted or simple)
    if SIMPLE_AVG:
        weighted_baseline = compute_simple_average(baseline_2020, countries)
        weighted_bc = compute_simple_average(policy_series['bc'], countries)
        weighted_eu = compute_simple_average(policy_series['eu'], countries)
        weighted_gl = compute_simple_average(policy_series['gl'], countries)
    else:
        weighted_baseline = compute_weighted_average(baseline_2020, net_trillions, countries)
        weighted_bc = compute_weighted_average(policy_series['bc'], net_trillions, countries)
        weighted_eu = compute_weighted_average(policy_series['eu'], net_trillions, countries)
        weighted_gl = compute_weighted_average(policy_series['gl'], net_trillions, countries)

    # Check for negative weighted averages in betweenness
    for label, weighted in [
        ("Baseline", weighted_baseline),
        ("BC", weighted_bc),
        ("EU", weighted_eu),
        ("GL", weighted_gl),
    ]:
        negatives = [country for country, value in weighted.items() if value is not None and value < 0]
        if negatives:
            print(f"Warning: Negative betweenness averages in {label} for countries: {negatives}")

    create_and_save_percent_change_map_plot('Betweenness Centrality', weighted_baseline, weighted_bc, weighted_eu, weighted_gl, 'results/betweenness_percent_change_maps.png')

def process_hub_authority(countries, net_trillions):
    # 1) Load baseline files (2010-2020) for Hub and Authority scores
    hub_baseline_dfs = {}
    auth_baseline_dfs = {}
    for year in range(2010, 2021):
        bf = f"Dataset/hub_aut_{year}.parquet"
        df = pd.read_parquet(bf)
        if {'node', 'hub_score', 'authority_score'}.issubset(df.columns):
            hub_series = pd.to_numeric(df['hub_score'], errors='coerce')
            auth_series = pd.to_numeric(df['authority_score'], errors='coerce')
            hub_series.index = df['node']
            auth_series.index = df['node']
            hub_baseline_dfs[year] = hub_series
            auth_baseline_dfs[year] = auth_series

    hub_baseline_df = pd.DataFrame(hub_baseline_dfs).dropna(how='any', axis=0)
    auth_baseline_df = pd.DataFrame(auth_baseline_dfs).dropna(how='any', axis=0)
    hub_2020 = hub_baseline_df[2020]
    auth_2020 = auth_baseline_df[2020]

    # 2) Load policy files
    policy_hub = {}
    policy_auth = {}
    for policy, pf in [('bc', 'Dataset/hub_aut_bc.parquet'), ('eu', 'Dataset/hub_aut_eu.parquet'), ('gl', 'Dataset/hub_aut_gl.parquet')]:
        dfp = pd.read_parquet(pf)
        hub_series = pd.to_numeric(dfp['hub_score'], errors='coerce')
        auth_series = pd.to_numeric(dfp['authority_score'], errors='coerce')
        hub_series.index = dfp['node']
        auth_series.index = dfp['node']
        policy_hub[policy] = hub_series
        policy_auth[policy] = auth_series

    # 3) Compute hub averages per country (weighted or simple)
    if SIMPLE_AVG:
        weighted_hub_baseline = compute_simple_average(hub_2020, countries)
        weighted_hub_bc = compute_simple_average(policy_hub['bc'], countries)
        weighted_hub_eu = compute_simple_average(policy_hub['eu'], countries)
        weighted_hub_gl = compute_simple_average(policy_hub['gl'], countries)
    else:
        weighted_hub_baseline = compute_weighted_average(hub_2020, net_trillions, countries)
        weighted_hub_bc = compute_weighted_average(policy_hub['bc'], net_trillions, countries)
        weighted_hub_eu = compute_weighted_average(policy_hub['eu'], net_trillions, countries)
        weighted_hub_gl = compute_weighted_average(policy_hub['gl'], net_trillions, countries)

    # Check for negative weighted averages in hub scores
    for label, weighted in [
        ("Baseline Hub", weighted_hub_baseline),
        ("BC Hub", weighted_hub_bc),
        ("EU Hub", weighted_hub_eu),
        ("GL Hub", weighted_hub_gl),
    ]:
        negatives = [country for country, value in weighted.items() if value is not None and value < 0]
        if negatives:
            print(f"Warning: Negative hub averages in {label} for countries: {negatives}")

    create_and_save_percent_change_map_plot('Hub Score', weighted_hub_baseline, weighted_hub_bc, weighted_hub_eu, weighted_hub_gl, 'results/hub_percent_change_maps.png')

    # 4) Compute authority averages per country (weighted or simple)
    if SIMPLE_AVG:
        weighted_auth_baseline = compute_simple_average(auth_2020, countries)
        weighted_auth_bc = compute_simple_average(policy_auth['bc'], countries)
        weighted_auth_eu = compute_simple_average(policy_auth['eu'], countries)
        weighted_auth_gl = compute_simple_average(policy_auth['gl'], countries)
    else:
        weighted_auth_baseline = compute_weighted_average(auth_2020, net_trillions, countries)
        weighted_auth_bc = compute_weighted_average(policy_auth['bc'], net_trillions, countries)
        weighted_auth_eu = compute_weighted_average(policy_auth['eu'], net_trillions, countries)
        weighted_auth_gl = compute_weighted_average(policy_auth['gl'], net_trillions, countries)

    # Check for negative weighted averages in authority scores
    for label, weighted in [
        ("Baseline Auth", weighted_auth_baseline),
        ("BC Auth", weighted_auth_bc),
        ("EU Auth", weighted_auth_eu),
        ("GL Auth", weighted_auth_gl),
    ]:
        negatives = [country for country, value in weighted.items() if value is not None and value < 0]
        if negatives:
            print(f"Warning: Negative authority averages in {label} for countries: {negatives}")

    create_and_save_percent_change_map_plot('Authority Score', weighted_auth_baseline, weighted_auth_bc, weighted_auth_eu, weighted_auth_gl, 'results/authority_percent_change_maps.png')

def main():
    # List of countries of interest
    countries = [
        "CHN","USA","RoW","JPN","DEU","FRA","IND","GBR","ITA","KOR","BRA","RUS",
        "AUS","CAN","ESP","TUR","MEX","IDN","NLD","CHE","POL","BEL","SWE","AUT",
        "CZE","NOR","FIN","DNK","ROU","IRL","PRT","GRC","HUN","SVK","LUX","BGR",
        "HRV","SVN","LTU","LVA","EST","CYP","MLT"
    ]

    # Compute net inflows per node
    net_trillions = load_net_trillions('Dataset/dfz_2019.parquet')
    # Check for negative net inflows
    neg_net = net_trillions[net_trillions < 0]
    if not neg_net.empty:
        print("Warning: Negative net inflows detected for nodes:", neg_net.index.tolist())

    # Process each metric
    process_clustering(countries, net_trillions)
    process_betweenness(countries, net_trillions)
    process_hub_authority(countries, net_trillions)

if __name__ == "__main__":
    main()