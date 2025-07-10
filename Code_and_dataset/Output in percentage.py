import os
import sys
# Add project root to module search path so Code_and_dataset can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ensure results directory exists
os.makedirs('results', exist_ok=True)
os.makedirs(os.path.join('results', 'country specific network measurements'), exist_ok=True)
import pandas as pd
import numpy as np
from Code_and_dataset.World_map import plot_world_map, plot_world_maps_grid
import matplotlib.pyplot as plt
# Disable interactive showing from helper functions
_original_show = plt.show
plt.show = lambda *args, **kwargs: None
from scipy.stats import pearsonr

def main():
    # 1) LOAD BASELINE COUNTRY-LEVEL DATA FOR YEARS 2010–2020
    baseline_files = [
        (2010, "Code_and_dataset/dfz_2010_c.parquet"),
        (2011, "Code_and_dataset/dfz_2011_c.parquet"),
        (2012, "Code_and_dataset/dfz_2012_c.parquet"),
        (2013, "Code_and_dataset/dfz_2013_c.parquet"),
        (2014, "Code_and_dataset/dfz_2014_c.parquet"),
        (2015, "Code_and_dataset/dfz_2015_c.parquet"),
        (2016, "Code_and_dataset/dfz_2016_c.parquet"),
        (2017, "Code_and_dataset/dfz_2017_c.parquet"),
        (2018, "Code_and_dataset/dfz_2018_c.parquet"),
        (2019, "Code_and_dataset/dfz_2019_c.parquet"),
        (2020, "Code_and_dataset/dfz_2020_c.parquet")  # Baseline 2020 data
    ]

    baseline_net = {}

    for year, filename in baseline_files:
        try:
            df = pd.read_parquet(filename)
            # Compute inflows and outflows
            inflows = df.sum(axis=0) -pd.Series(df.values.diagonal(), index=df.index)
            outflows = df.sum(axis=1)
            net = -inflows + outflows
            net_trillions = net / 1e9
            print(f"{filename} net inflows (trillions): {len(net_trillions)} countries.")
            baseline_net[year] = net_trillions
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Combine into DataFrame: index are country ISO codes, columns are years
    baseline_df = pd.DataFrame(baseline_net)

    # 2) COMPUTE STANDARD DEVIATION OVER 2010–2020 FOR EACH COUNTRY
    std_dev = baseline_df.std(axis=1)
    zero_std_countries = std_dev[std_dev == 0].index
    if len(zero_std_countries) > 0:
        print(f"{len(zero_std_countries)} countries have zero standard deviation; their z-scores will be NaN.")

    # 3) EXTRACT BASELINE 2020 NET INFLOWS
    if 2020 not in baseline_df:
        print("Baseline 2020 data not found. Exiting.")
        return
    baseline_2020 = baseline_df[2020]

    # 4) LOAD POLICY SCENARIO FILES AND COMPUTE PERCENTAGE CHANGES
    policy_files = {
        "BC": "Code_and_dataset/dfz_bc_c.parquet",  # EU-wide carbon tax with CBAM
        "EU": "Code_and_dataset/dfz_eu_c.parquet",  # EU-only carbon tax
        "GL": "Code_and_dataset/dfz_gl_c.parquet"   # Global carbon tax
    }

    policy_changes = {}

    for policy, filename in policy_files.items():
        try:
            dfp = pd.read_parquet(filename)
            inflows_p = dfp.sum(axis=0)-pd.Series(dfp.values.diagonal(), index=df.index)
            outflows_p = dfp.sum(axis=1)
            net_p = -inflows_p + outflows_p
            net_p_trillions = net_p / 1e9

            # Align with baseline countries
            net_p_aligned = net_p_trillions.reindex(std_dev.index)

            # Difference from baseline 2020
            diff = net_p_aligned - baseline_2020

            # Compute percentage change relative to baseline 2020
            perc_change = diff / baseline_2020 * 100
            policy_changes[policy] = perc_change
            print(f"{filename} policy net inflows computed and percentage change generated.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    # Compute year-over-year relative changes and then average them
    # rel_changes[t] = (baseline_df[t] - baseline_df[t-1]) / baseline_df[t-1]
    rel_changes = baseline_df.pct_change(axis=1)
    # Keep only years 2011–2020
    rel_changes = rel_changes.loc[:, list(range(2011, 2021))]
    print("Year-over-year relative changes:", rel_changes.head(30))
    mean_rel_change = rel_changes.mean(axis=1) * 100

    # 5) PLOT 2x2 WORLD MAPS: average relative variation + policy scenario maps
    # Prepare DataFrames for maps
    dfs = []
    titles = []
    # Average relative variation map
    df_avg = pd.DataFrame({
        'ISO_A3': mean_rel_change.index,
        'value': mean_rel_change.values
    })
    dfs.append(df_avg)
    titles.append('Average Net Flow Variation (2010-2020)')
    # Policy scenario maps
    policy_title_map = {
        'BC': 'Carbon tax in the EU + CBAM',
        'EU': 'carbon tax only in the EU',
        'GL': 'global carbon tax'
    }
    for p in ['BC', 'EU', 'GL']:
        series = policy_changes[p]
        df_p = pd.DataFrame({
            'ISO_A3': series.index,
            'value': series.values
        })
        dfs.append(df_p)
        titles.append(f"{policy_title_map[p]}: % Variation from Baseline 2020")

    # Plot all four maps in a 2x2 grid
    plot_world_maps_grid(
        dfs,
        x_labels=[''] * 4,
        y_labels=[''] * 4,
        titles=titles,
        nrows=2,
        ncols=2,
        figsize=(15, 12)
    )
    # Save the 2×2 world map grid figure
    plt.savefig(os.path.join('results', 'country specific network measurements', 'NetFlow_Variation_2x2_Maps.png'), bbox_inches='tight')
    plt.close()

    # 6) PLOT CORRELATION BETWEEN GLOBAL TAX NET FLOW CHANGE AND CARBON INTENSITY
    try:
        # Load carbon intensity data for 2020
        ci_df = pd.read_csv(
            "Code_and_dataset/API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv",
            skiprows=4
        )
        # Extract ISO code and 2020 values
        ci_2020 = (
            ci_df[["Country Code", "2020"]]
            .rename(columns={"Country Code": "ISO_A3", "2020": "ci_2020"})
            .set_index("ISO_A3")["ci_2020"]
        )
        # Align with global policy percentage change
        gl_change = policy_changes["GL"]
        combined = (
            pd.DataFrame({"perc_change_gl": gl_change, "ci_2020": ci_2020})
            .dropna()
        )
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(combined["ci_2020"], combined["perc_change_gl"])
        # Annotate each point with its country code
        for iso, row in combined.iterrows():
            plt.annotate(
                iso,
                xy=(row["ci_2020"], row["perc_change_gl"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        plt.xlabel(
            "Carbon Intensity of GDP in 2020 (kg CO2e per 2021 PPP $)"
        )
        plt.ylabel(
            "Net Flow % Change under Global Carbon Tax"
        )
        plt.title(
            "Countries with Higher Carbon Intensity Tend to Experience Greater Net Trade Losses under a Global Carbon Tax"
        )
        # Compute Pearson correlation and p-value
        r_value, p_value = pearsonr(combined["ci_2020"], combined["perc_change_gl"])
        plt.annotate(
            f"r = {r_value:.2f}, p = {p_value:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top"
        )
        plt.tight_layout()
        # Save scatter plot of global tax net flow change vs. carbon intensity
        plt.savefig(os.path.join('results', 'country specific network measurements', 'GlobalTax_vs_CarbonIntensity_Scatter.png'), bbox_inches='tight')
        plt.close()
        # Compute GDP per CO2 (inverse of carbon intensity) for countries present in other maps
        gdp_per_co2 = (1 / ci_2020).rename('value')
        # Filter to countries present in earlier maps
        iso_list = pd.concat(dfs)['ISO_A3'].unique()
        df_gdp_co2 = gdp_per_co2.reset_index()  # columns: ISO_A3, value
        df_gdp_co2 = df_gdp_co2[df_gdp_co2['ISO_A3'].isin(iso_list)]
        plot_world_map(
            df_gdp_co2,
            '',
            '',
            'Carbon Efficiency of GDP (2020): PPP-adjusted $ Output per kg CO2e'
        )
        # Save the carbon efficiency map
        plt.savefig(os.path.join('results', 'country specific network measurements', 'CarbonEfficiency_Map.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting correlation: {e}")

if __name__ == "__main__":
    main()