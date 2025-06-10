
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

def create_and_save_table_plot(metric_name, weighted_baseline, weighted_bc, weighted_eu, weighted_gl, output_filename):
    # Convert to DataFrames and sort
    df_baseline = pd.DataFrame.from_dict(weighted_baseline, orient='index', columns=['Value']).sort_values('Value', ascending=False)

    # Compute baseline rank positions
    baseline_ranks = {country: rank for rank, country in enumerate(df_baseline.index, start=1)}

    df_bc = pd.DataFrame.from_dict(weighted_bc, orient='index', columns=['Value']).sort_values('Value', ascending=False)
    df_eu = pd.DataFrame.from_dict(weighted_eu, orient='index', columns=['Value']).sort_values('Value', ascending=False)
    df_gl = pd.DataFrame.from_dict(weighted_gl, orient='index', columns=['Value']).sort_values('Value', ascending=False)

    fig, axes = plt.subplots(1, 4, figsize=(20, 18))
    # Place the metric name once at the top
    fig.suptitle(metric_name, y=0.98, fontsize=25, fontweight='bold')
    # Add labels above each table (Baseline, BC, EU, GL)
    label_positions = [0.125, 0.375, 0.625, 0.875]
    label_names = ['Baseline 2020', 'BC', 'EU', 'GL']
    for xpos, label in zip(label_positions, label_names):
        fig.text(xpos, 0.94, label, ha='center', fontsize=18)
    tables = [(df_baseline, 'Baseline 2020'), (df_bc, 'BC'), (df_eu, 'EU'), (df_gl, 'GL')]

    for ax, (df_table, title) in zip(axes.flatten(), tables):
        ax.axis('off')
        # Prepare table data with values rounded to two significant digits
        table_rows = []
        for idx, (country, value) in enumerate(df_table['Value'].items(), start=1):
            # Format the value
            if pd.isna(value):
                formatted = ''
            else:
                formatted = f"{value:.2g}"
            # For Baseline 2020, only show country and value
            if title == 'Baseline 2020':
                table_rows.append([country, formatted])
            else:
                # For policy tables, compute ranking change only if country in EU_MEMBERS
                curr_rank = idx
                baseline_rank = baseline_ranks.get(country)
                if baseline_rank is not None and country in EU_MEMBERS:
                    change = baseline_rank - curr_rank
                    if change > 0:
                        change_str = f"+{change}"
                    elif change < 0:
                        change_str = f"{change}"
                    else:
                        change_str = "0"
                else:
                    change_str = ""
                table_rows.append([country, formatted, change_str])
        # Create table without column labels
        tbl = ax.table(cellText=table_rows, colLabels=None, loc='center')
        if title != 'Baseline 2020':
            # Set text color based on positive (green) or negative (red) change in third column
            for row_idx, row in enumerate(table_rows):
                change_text = row[2]
                if change_text.startswith('+') and change_text != "+0":
                    tbl[row_idx, 2].get_text().set_color('green')
                elif change_text.startswith('-'):
                    tbl[row_idx, 2].get_text().set_color('red')
        # Turn off automatic font sizing and increase text size
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        # Increase row height for more spacing and make columns narrower
        tbl.scale(0.8, 2)
        # ax.set_title(title, pad=10, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_filename)
    plt.close(fig)

def process_clustering(countries, net_trillions):
    # 1) Load baseline files (2010-2020) for Clustering Coefficient
    baseline_dfs = {}
    for year in range(2010, 2021):
        bf = f"clus_{year}.parquet"
        df = pd.read_parquet(bf)
        if {'Node', 'Clustering_Coefficient'}.issubset(df.columns):
            series = pd.to_numeric(df['Clustering_Coefficient'], errors='coerce')
            series.index = df['Node']
            baseline_dfs[year] = series
    baseline_df = pd.DataFrame(baseline_dfs).dropna(how='any', axis=0)
    baseline_2020 = baseline_df[2020]

    # 2) Load policy files
    policy_series = {}
    for policy, pf in [('bc', 'clus_bc.parquet'), ('eu', 'clus_eu.parquet'), ('gl', 'clus_gl.parquet')]:
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

    create_and_save_table_plot('Clustering Coefficient', weighted_baseline, weighted_bc, weighted_eu, weighted_gl, 'clustering_tables.png')

def process_betweenness(countries, net_trillions):
    # 1) Load baseline files (2010-2020) for Betweenness Centrality
    baseline_dfs = {}
    for year in range(2010, 2021):
        bf = f"dfz_s_{year}_bc.parquet"
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
    for policy, pf in [('bc', 'dfz_s_bc_bc.parquet'), ('eu', 'dfz_s_eu_bc.parquet'), ('gl', 'dfz_s_gl_bc.parquet')]:
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

    create_and_save_table_plot('Betweenness Centrality', weighted_baseline, weighted_bc, weighted_eu, weighted_gl, 'betweenness_tables.png')

def process_hub_authority(countries, net_trillions):
    # 1) Load baseline files (2010-2020) for Hub and Authority scores
    hub_baseline_dfs = {}
    auth_baseline_dfs = {}
    for year in range(2010, 2021):
        bf = f"hub_aut_{year}.parquet"
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
    for policy, pf in [('bc', 'hub_aut_bc.parquet'), ('eu', 'hub_aut_eu.parquet'), ('gl', 'hub_aut_gl.parquet')]:
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

    create_and_save_table_plot('Hub Score', weighted_hub_baseline, weighted_hub_bc, weighted_hub_eu, weighted_hub_gl, 'hub_tables.png')

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

    create_and_save_table_plot('Authority Score', weighted_auth_baseline, weighted_auth_bc, weighted_auth_eu, weighted_auth_gl, 'authority_tables.png')

def main():
    # List of countries of interest
    countries = [
        "CHN","USA","RoW","JPN","DEU","FRA","IND","GBR","ITA","KOR","BRA","RUS",
        "AUS","CAN","ESP","TUR","MEX","IDN","NLD","CHE","POL","BEL","SWE","AUT",
        "CZE","NOR","FIN","DNK","ROU","IRL","PRT","GRC","HUN","SVK","LUX","BGR",
        "HRV","SVN","LTU","LVA","EST","CYP","MLT"
    ]

    # Compute net inflows per node
    net_trillions = load_net_trillions('./dfz_2019.parquet')
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