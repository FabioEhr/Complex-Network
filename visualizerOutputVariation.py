import pandas as pd
import numpy as np
from World_map import plot_world_map

def main():
    # 1) LOAD BASELINE COUNTRY-LEVEL DATA FOR YEARS 2010–2020
    baseline_files = [
        (2010, "dfz_2010_c.parquet"),
        (2011, "dfz_2011_c.parquet"),
        (2012, "dfz_2012_c.parquet"),
        (2013, "dfz_2013_c.parquet"),
        (2014, "dfz_2014_c.parquet"),
        (2015, "dfz_2015_c.parquet"),
        (2016, "dfz_2016_c.parquet"),
        (2017, "dfz_2017_c.parquet"),
        (2018, "dfz_2018_c.parquet"),
        (2019, "dfz_2019_c.parquet"),
        (2020, "dfz_2020_c.parquet")  # Baseline 2020 data
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

    # 4) LOAD CARBON INTENSITY DATA (2020)
    try:
        carb_df = pd.read_csv(
            "API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv",
            skiprows=4, header=0
        )
        carbon_intensity = carb_df.set_index("Country Code")["2020"].astype(float)
        print("Loaded carbon intensity data for 2020.")
    except Exception as e:
        print(f"Error loading carbon intensity data: {e}")
        return

    # 4) LOAD POLICY SCENARIO FILES AND COMPUTE Z-SCORES
    policy_files = {
        "BC": "dfz_bc_c.parquet",  # EU-wide carbon tax with CBAM
        "EU": "dfz_eu_c.parquet",  # EU-only carbon tax
        "GL": "dfz_gl_c.parquet"   # Global carbon tax
    }

    policy_zscores = {}

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

            # Compute correlation with carbon intensity
            ci_aligned = carbon_intensity.reindex(diff.index)
            corr = diff.corr(ci_aligned)
            print(f"Correlation between change in net inflow and carbon intensity for policy {policy}: {corr:.3f}")

            # Compute z-scores
            #z_score = diff / std_dev
            z_score = diff / baseline_2020
            policy_zscores[policy] = z_score
            print(f"{filename} policy net inflows computed and z-scores generated.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 5) PLOT WORLD MAPS FOR EACH SCENARIO
    for policy, z in policy_zscores.items():
        df_plot = pd.DataFrame(list(z.items()), columns=['ISO_A3', 'value'])
        title = f"Net Inflow Variation Z-score - Policy {policy}"
        plot_world_map(df_plot, '', '', title)

    # 6) CREATE 2x2 BAR CHARTS: VALUE PER COUNTRY FOR EACH SCENARIO
    import matplotlib.pyplot as plt

    # Compute average relative variation for each country over 2010–2020
    rel_changes = (baseline_df.subtract(baseline_df[2010], axis=0)).div(baseline_df[2010], axis=0)
    mean_rel_change = rel_changes.loc[:, list(range(2011, 2021))].mean(axis=1)

    # Compute baseline correlation with carbon intensity
    ci_baseline = carbon_intensity.reindex(mean_rel_change.index)
    corr_baseline = mean_rel_change.corr(ci_baseline)
    print(f"Baseline correlation between mean relative change and carbon intensity: {corr_baseline:.3f}")

    # Define EU country ISO codes
    eu_codes = {
        "AUT", "BEL", "BGR", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
    }

    # Prepare figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    scenarios = list(policy_zscores.keys()) + ["Baseline"]

    for ax, scenario in zip(axes.flat, scenarios):
        if scenario != "Baseline":
            data = policy_zscores[scenario].dropna()
            title = f"Policy {scenario}: Z-score per Country"
        else:
            data = mean_rel_change.dropna()
            title = "Baseline: Mean Rel. Variation 2010–2020"

        # Prepare lists for plotting
        codes = list(data.index)
        values = data.values

        # Assign colors: EU=blue, Non-EU=grey, RoW=red
        colors = []
        for code in codes:
            if code == "RoW":
                colors.append('red')
            elif code in eu_codes:
                colors.append('blue')
            else:
                colors.append('grey')

        # Plot bar chart
        x_positions = np.arange(len(codes))
        ax.bar(x_positions, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(codes, rotation=90, fontsize=6)
        # Add legend patches
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='EU')
        grey_patch = mpatches.Patch(color='grey', label='Non-EU')
        red_patch = mpatches.Patch(color='red', label='RoW')
        ax.legend(handles=[blue_patch, grey_patch, red_patch], loc='upper right')

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()