import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# years and the four pairs we’ll track
years = list(range(2010, 2021))

# define the measures whose year-over-year ranking stability we want
measures = [
    "Clustering_Coefficient",
    "hub_score",
    "authority_score",
    "betweenness",
]

# read all dataframes into dictionaries keyed by year
dfs = {}
for year in years:
    df_clus = pd.read_parquet(f"./clus_{year}.parquet").set_index("Node")
    df_hub = (
        pd.read_parquet(f"./hub_aut_{year}.parquet")
         .rename(columns=str.lower)
         .set_index("node")
         .rename_axis("Node")
    )
    df_bet = (
        pd.read_parquet(f"./dfz_s_{year}_bc.parquet")
         .rename_axis("Node")
    )
    # join all three into one per-year table
    dfs[year] = df_clus.join(
        df_hub[["hub_score","authority_score"]]
    ).join(
        df_bet[["betweenness"]]
    )

# prepare a DataFrame to hold Spearman SRCC between consecutive years
srcc = pd.DataFrame(
    index=years[1:], 
    columns=measures, 
    dtype=float
)

# compute SRCC for each measure between t-1 and t
for t in years[1:]:
    df_prev = dfs[t-1]
    df_curr = dfs[t]
    # find common nodes
    common = df_prev.index.intersection(df_curr.index)
    for m in measures:
        print(f"Computing SRCC for year {t} and measure '{m}'")
        a = df_prev.loc[common, m]
        b = df_curr.loc[common, m]
        # Sort by values and compute position changes
        a_sorted = a.sort_values()
        b_sorted = b.sort_values()
        a_pos = {node: i for i, node in enumerate(a_sorted.index)}
        b_pos = {node: i for i, node in enumerate(b_sorted.index)}
        print(f"Position changes for measure '{m}' between {t-1} and {t}:")
        if m == "clustering_coefficient":
            # For clustering coefficient, we can also print the delta in values
            print("  Node: ΔClustering Coefficient")
            for node in common[220:250]:
                delta = b_pos[node] - a_pos[node]
                print(f"  {node}: Δpos = {delta}")
        #print(a.head(), b.head())
        #print(f"Year {t}, measure '{m}': common={len(a)}, prev={len(df_prev.loc[:, m])}, curr={len(df_curr.loc[:, m])}")
        rho, _ = spearmanr(a, b)
        print(f"Spearman SRCC for {m} between {t-1} and {t}: ρ={rho:.3f}")
        #print(f"Spearman SRCC for {m} between {t-1} and {t}: ρ={rho:.3f}, p-value={pippo:.3f}")
        srcc.loc[t, m] = rho

# (optional) print the time-average SRCC for each measure
print("Time-averaged Spearman SRCC:")
print(srcc.mean())

# --- compute SRCC between baseline (2020) and each policy scenario ---
scenario_keys = ["bc", "eu", "gl"]
scenario_labels = {"bc": "CBAM", "eu": "EU carbon tax", "gl": "Global carbon tax"}

# load scenario results for year 2020
scenario_dfs = {}
for sc in scenario_keys:
    df_c = pd.read_parquet(f"./clus_{sc}.parquet").set_index("Node")
    df_h = (
        pd.read_parquet(f"./hub_aut_{sc}.parquet")
         .rename(columns=str.lower)
         .set_index("node")
         .rename_axis("Node")
    )
    df_b = pd.read_parquet(f"./dfz_s_{sc}_bc.parquet").rename_axis("Node")
    scenario_dfs[sc] = df_c.join(
        df_h[["hub_score", "authority_score"]]
    ).join(
        df_b[["betweenness"]]
    )

# compute SRCC vs baseline 2020
scenario_srcc = pd.DataFrame(index=scenario_keys, columns=measures, dtype=float)
baseline = dfs[2020]
for sc in scenario_keys:
    df_sc = scenario_dfs[sc]
    common = baseline.index.intersection(df_sc.index)
    print(len(common), "common nodes between baseline and", sc)
    for m in measures:
        a = baseline.loc[common, m]
        b = df_sc.loc[common, m]
        rho, _ = spearmanr(a, b)
        scenario_srcc.loc[sc, m] = rho

print("Spearman SRCC vs baseline 2020 for each scenario:")
print(scenario_srcc)

# plot the SRCC time series for each measure in a 2×2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, m in zip(axes, measures):
    ax.plot(srcc.index, srcc[m], marker="o", linestyle="-")
    # overlay scenario SRCCs as square markers at x=2021
    for sc in scenario_keys:
        ax.plot(2021, scenario_srcc.loc[sc, m], marker="s", linestyle="", label=sc.upper())
    # adjust x-axis to include 2021
    ax.set_xlim(srcc.index.min(), 2021.5)
    ax.set_title(f"SRCC of {m}: year-to-year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Spearman ρ")
    ax.grid(True)

# place a global legend mapping SCENARIO markers
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(scenario_keys), title="Policy Scenario")

plt.tight_layout()
plt.show()