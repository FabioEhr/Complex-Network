import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

# --- Constants (from your provided code, not all directly used by this specific analysis) ---
countries = ['AUS', 'AUT', 'BEL', 'BGR', 'BRA', 'CAN', 'CHE', 'CHN', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 
                     'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IDN', 'IND', 'IRL', 'ITA', 'JPN', 'KOR', 
                     'LTU', 'LUX', 'LVA', 'MEX', 'MLT', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SVK', 'SVN', 
                     'SWE', 'TUR', 'USA','RoW']

sector_codes_dict = { # Renamed from sector_codes to avoid conflict if used elsewhere
    "ENRcoa": "ENRcoa", "ENRele": "ENRele", "ENRgas": "ENRele", "ENRoil": "ENRoil", "EXT+": "EXT+",
    "TRDret": "Retail trade services, except of motor vehicles and motorcycles", "TRA+": "Transportation and storage",
    "TRAinl": "Land transport and transport via pipelines", "TRAwat": "Water transport", "TRAair": "Air transport",
    "TRAwar": "Warehousing and support activities for transportation", "TRApos": "Postal and courier activities",
    "FD+": "Accommodation and food service activities", "COM+": "Information and communication",
    "COMpub": "Publishing activities", "COMvid": "Motion picture, video and television production, sound recording, broadcast-ing",
    "COMtel": "Telecommunications", "COMcom": "Computer programming, consultancy; Information service activities",
    "FIN+": "Financial and insurance activities", "FINser": "Financial services, except insurance and pension funding",
    "FINins": "Insurance, reinsurance and pension funding services, except compulsory social security",
    "FINaux": "Activities auxiliary to financial services and insurance services", "RES+": "Real estate activities",
    "PRO+": "Professional, scientific and technical activities",
    "PROleg": "Legal and accounting services; Activities of head offices; management consultancy activities",
    "PROeng": "Architectural and engineering activities; technical testing and analysis",
    "PROsci": "Scientific research and development", "PROadv": "Advertising and market research",
    "PROoth": "Other professional, scientific and technical activities; Veterinary activities",
    "ADM+": "Administrative and support service activities", "PUB+": "Public administration and defence; compulsory social security",
    "EDU+": "Education", "HEA+": "Human health and social work activities", "ART+": "Arts, entertainment and recreation",
    "HOU+": "Activities of households as employers", "AGR+": "Agriculture, forestry and fishing",
    "AGRagr": "Crop and animal production, hunting and related service activities", "AGRfor": "Forestry and logging",
    "AGRfis": "Fishing and aquaculture", "MIN+": "Mining and quarrying",
    "MINfos": "Mining and extraction of energy producing products",
    "MINoth": "Mining and quarrying of non-energy producing products", "MINsup": "Mining support service activities",
    "MAN+": "Manufacturing", "MANfoo": "Food, beverages and tobacco products",
    "MANtex": "Textiles, wearing apparel, leather and related products",
    "MANwoo": "Wood and products of wood and cork, except furniture", "MANpap": "Paper and paper products",
    "MANpri": "Printing and reproduction of recorded media", "MANref": "Coke and refined petroleum products",
    "MANche": "Chemicals and chemical products", "MANpha": "Basic pharmaceutical products and pharmaceutical preparations",
    "MANpla": "Rubber and plastic products", "MANmin": "Other non-metallic mineral products",
    "MANmet": "Basic metals", "MANfmp": "Fabricated metal products, except machinery and equipment",
    "MANcom": "Computer, electronic and optical products", "MANele": "Electrical equipment",
    "MANmac": "Machinery and equipment n.e.c.", "MANmot": "Motor vehicles, trailers and semi-trailers",
    "MANtra": "Other transport equipment", "MANfur": "Furniture and other manufactured goods",
    "MANrep": "Repair and installation services of machinery and equipment", "PWR+": "Electricity, gas, steam and air conditioning",
    "WAT+": "Water supply; sewerage; waste management and remediation",
    "WATwat": "Natural water; water treatment and supply services",
    "WATwst": "Sewerage services; sewage sludge; waste collection, treatment and disposal services",
    "CNS+": "Constructions and construction works", "TRD+": "Wholesale and retail trade; repair of motor vehicles and motorcycles",
    "TRDmot": "Wholesale and retail trade and repair services of motor vehicles and motorcycles",
    "TRDwho": "Wholesale trade, except of motor vehicles and motorcycles"
}

# --- Helper function to compute sum of measure per country (unweighted) ---
def compute_country_sum_measure(series, countries):
    """Computes the sum of the measure for each country from node-level data."""
    country_sums = {}
    for country in countries:
        # Filter nodes belonging to the current country
        nodes = [node for node in series.index if node.startswith(f"{country}_")]
        if nodes:
            # Ensure nodes are actually in the series' index before attempting to access
            valid_nodes = [node for node in nodes if node in series.index]
            if valid_nodes:
                values = np.array([series.get(node, np.nan) for node in valid_nodes], dtype=float)
                values = np.nan_to_num(values, nan=0.0) # Replace NaNs with zeros for summation
                country_sums[country] = np.sum(values)
            else:
                country_sums[country] = 0.0 # No valid nodes for this country in the series
        else:
            country_sums[country] = 0.0 # No nodes starting with this country prefix
    return pd.Series(country_sums)

# --- Core Logic Functions ---

def get_all_sectors_from_series(series_list):
    """Extracts all unique sector codes from a list of pandas Series with 'COUNTRY_SECTOR' index."""
    all_sectors = set()
    for series in series_list:
        if series is None or series.empty:
            continue
        for node_index in series.index:
            try:
                parts = node_index.split('_', 1)
                if len(parts) > 1:
                    all_sectors.add(parts[1])
            except AttributeError: # Handle cases where index might not be strings
                # print(f"Warning: Index item '{node_index}' is not a string. Skipping for sector extraction.")
                pass # Or handle more robustly if necessary
    return sorted(list(all_sectors))

def calculate_average_sector_change_across_countries(base_series, policy_series, all_sector_codes, all_countries):
    """
    Calculates the average change of each sector's network measure across all countries.
    Change = policy_value - baseline_value for each country-sector node.
    Average is taken over all countries for that sector where the sector is present in baseline or policy.
    """
    avg_sector_changes = {}
    for sector_code in all_sector_codes:
        sector_node_changes = []
        for country_code in all_countries:
            node = f"{country_code}_{sector_code}"
            
            # Consider a change if the node exists in either baseline or policy
            if node in base_series.index or node in policy_series.index:
                base_val = base_series.get(node, 0.0) # Default to 0 if node not in baseline
                policy_val = policy_series.get(node, 0.0) # Default to 0 if node not in policy
                sector_node_changes.append(policy_val - base_val)

        if sector_node_changes: # If the sector was found and had changes in at least one country
            avg_sector_changes[sector_code] = np.mean(sector_node_changes)
        else:
            # If sector code never appears or has no changes, average change is 0
            avg_sector_changes[sector_code] = 0.0
    return pd.Series(avg_sector_changes)


def calculate_predicted_country_change(country_code, base_node_series, avg_sector_changes_map, country_sectors_in_baseline):
    """
    Calculates the "predicted" change for a single country.
    Predicted_change = sum over sectors s (share_of_s_in_country_baseline * avg_change_of_s_globally)
    """
    predicted_change_value = 0.0

    # 1. Calculate shares of each sector in the country's baseline total for the measure
    # Get all nodes for this country that are in the base_node_series
    country_baseline_nodes_present = [
        f"{country_code}_{s_code}" for s_code in country_sectors_in_baseline 
        if f"{country_code}_{s_code}" in base_node_series.index
    ]
    
    if not country_baseline_nodes_present: # Country has no baseline data for any of its identified sectors
        return 0.0

    # Sum of baseline values for the specific country's sectors that are present
    country_total_baseline_measure = base_node_series.reindex(country_baseline_nodes_present).sum()

    if pd.isna(country_total_baseline_measure) or country_total_baseline_measure == 0:
        # If total baseline is zero or NaN, shares are undefined or all zero.
        # Predicted change will be 0.
        return 0.0

    sector_shares_in_country = {}
    for sector_code in country_sectors_in_baseline:
        node = f"{country_code}_{sector_code}"
        if node in base_node_series.index: # Ensure sector node exists for the country
            sector_baseline_value = base_node_series.get(node, 0.0)
            sector_shares_in_country[sector_code] = sector_baseline_value / country_total_baseline_measure
        else:
            sector_shares_in_country[sector_code] = 0.0 # Sector not in this country's baseline
            
    # 2. Calculate predicted change by summing (share * average_sector_change)
    for sector_code in country_sectors_in_baseline: # Iterate over sectors identified for this country
        share = sector_shares_in_country.get(sector_code, 0.0)
        avg_sector_change = avg_sector_changes_map.get(sector_code, 0.0) # Get pre-calculated global avg change for this sector
        predicted_change_value += share * avg_sector_change
        
    return predicted_change_value

# --- Functions to load data for each measure ---

def load_clustering_data():
    df_base_clus = pd.read_parquet("clus_2020.parquet")
    if not ({'Node', 'Clustering_Coefficient'}.issubset(df_base_clus.columns)):
        raise ValueError("Clustering baseline data ('clus_2020.parquet') has unexpected columns.")
    base_series = pd.to_numeric(df_base_clus['Clustering_Coefficient'], errors='coerce')
    base_series.index = df_base_clus['Node']

    policy_series_map = {}
    for policy_code, policy_file in [('BC', 'clus_bc.parquet'), ('EU', 'clus_eu.parquet'), ('GL', 'clus_gl.parquet')]:
        dfp = pd.read_parquet(policy_file)
        if not ({'Node', 'Clustering_Coefficient'}.issubset(dfp.columns)):
             raise ValueError(f"Clustering policy data {policy_file} has unexpected columns.")
        series = pd.to_numeric(dfp['Clustering_Coefficient'], errors='coerce')
        series.index = dfp['Node']
        policy_series_map[policy_code] = series
    return base_series, policy_series_map

def load_betweenness_data():
    df_base = pd.read_parquet("dfz_s_2020_bc.parquet")
    if 'betweenness' in df_base.columns:
        base_series = pd.to_numeric(df_base['betweenness'], errors='coerce')
    else: 
        numeric_cols = df_base.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 1:
            base_series = pd.to_numeric(df_base[numeric_cols[0]], errors='coerce')
        else:
            raise ValueError("Betweenness baseline data ('dfz_s_2020_bc.parquet') has ambiguous numeric columns.")
    base_series.index = df_base.index # Assuming index is node names
            
    policy_series_map = {}
    for policy_code, policy_file in [('BC', 'dfz_s_bc_bc.parquet'), ('EU', 'dfz_s_eu_bc.parquet'), ('GL', 'dfz_s_gl_bc.parquet')]:
        dfp = pd.read_parquet(policy_file)
        if 'betweenness' in dfp.columns:
            series = pd.to_numeric(dfp['betweenness'], errors='coerce')
        else:
            numeric_cols = dfp.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 1:
                series = pd.to_numeric(dfp[numeric_cols[0]], errors='coerce')
            else:
                raise ValueError(f"Betweenness policy data {policy_file} has ambiguous numeric columns.")
        series.index = dfp.index # Assuming index is node names
        policy_series_map[policy_code] = series
    return base_series, policy_series_map

def load_hub_authority_data(score_type='hub_score'): # score_type can be 'hub_score' or 'authority_score'
    df_base = pd.read_parquet("hub_aut_2020.parquet")
    if not ({'node', score_type}.issubset(df_base.columns)):
        raise ValueError(f"Hub/Authority baseline data ('hub_aut_2020.parquet') missing 'node' or '{score_type}'.")
    base_series = pd.to_numeric(df_base[score_type], errors='coerce')
    base_series.index = df_base['node']

    policy_series_map = {}
    for policy_code, policy_file in [('BC', 'hub_aut_bc.parquet'), ('EU', 'hub_aut_eu.parquet'), ('GL', 'hub_aut_gl.parquet')]:
        dfp = pd.read_parquet(policy_file)
        if not ({'node', score_type}.issubset(dfp.columns)):
            raise ValueError(f"Hub/Authority policy data {policy_file} missing 'node' or '{score_type}'.")
        series = pd.to_numeric(dfp[score_type], errors='coerce')
        series.index = dfp['node']
        policy_series_map[policy_code] = series
    return base_series, policy_series_map


# --- Main processing function ---
def analyze_sectoral_contribution_to_change(countries_list):
    correlation_analysis_results = {}
    policy_scenarios = ['BC', 'EU', 'GL']

    measure_loaders_map = {
        "Clustering": load_clustering_data,
        "Betweenness": load_betweenness_data,
        "HubScore": lambda: load_hub_authority_data(score_type='hub_score'),
        "AuthorityScore": lambda: load_hub_authority_data(score_type='authority_score')
    }

    for measure_name, loader_function in measure_loaders_map.items():
        print(f"\nProcessing Measure: {measure_name}")
        correlation_analysis_results[measure_name] = {}
        try:
            base_node_series, policy_node_series_map = loader_function()
        except Exception as e:
            print(f"  Error loading data for {measure_name}: {e}")
            correlation_analysis_results[measure_name] = f"Error loading data: {e}"
            continue
        
        # Replace NaNs with 0 for calculations (measure values are typically non-negative)
        base_node_series = base_node_series.fillna(0.0)
        for pol_code in policy_scenarios:
            if pol_code in policy_node_series_map and policy_node_series_map[pol_code] is not None:
                 policy_node_series_map[pol_code] = policy_node_series_map[pol_code].fillna(0.0)

        # Get all unique sector codes present across all loaded series for this measure
        all_series_for_sector_scan = [base_node_series] + [policy_node_series_map.get(p) for p in policy_scenarios]
        all_unique_sector_codes = get_all_sectors_from_series(all_series_for_sector_scan)
        
        if not all_unique_sector_codes:
            print(f"  No sector codes found for {measure_name} from its data files. Skipping.")
            correlation_analysis_results[measure_name] = "No sector codes found in data."
            continue

        # Calculate sum of measure for each country in baseline (unweighted sum)
        baseline_country_sums = compute_country_sum_measure(base_node_series, countries_list)

        for policy_code in policy_scenarios:
            if policy_code not in policy_node_series_map or policy_node_series_map[policy_code] is None:
                print(f"  Policy {policy_code} data not found for {measure_name}. Skipping this policy.")
                correlation_analysis_results[measure_name][policy_code] = "Policy data not found."
                continue
            
            current_policy_node_series = policy_node_series_map[policy_code]
            
            # 1. Calculate Actual Country-Level Change
            policy_country_sums = compute_country_sum_measure(current_policy_node_series, countries_list)
            actual_country_changes = policy_country_sums - baseline_country_sums # This is a pd.Series
            
            # 2. Calculate "Predicted" Country-Level Change
            # 2a. Average change of each sector across all countries for this policy
            avg_sector_changes_globally = calculate_average_sector_change_across_countries(
                base_node_series, current_policy_node_series, all_unique_sector_codes, countries_list
            )
            
            predicted_country_changes_dict = {}
            for country_code_iter in countries_list:
                # Identify sectors genuinely associated with this country in the baseline data
                country_specific_sectors_in_baseline = [
                    s_code for s_code in all_unique_sector_codes 
                    if f"{country_code_iter}_{s_code}" in base_node_series.index
                ]
                
                pred_change = calculate_predicted_country_change(
                    country_code_iter, base_node_series, avg_sector_changes_globally, country_specific_sectors_in_baseline
                )
                predicted_country_changes_dict[country_code_iter] = pred_change
            
            predicted_country_changes = pd.Series(predicted_country_changes_dict) # Convert to Series
            
            # Align data for correlation (important if some countries are missing from one series)
            common_countries = actual_country_changes.index.intersection(predicted_country_changes.index)
            actual_vals_for_corr = actual_country_changes.reindex(common_countries).fillna(0.0) # Fill NaN for safety
            predicted_vals_for_corr = predicted_country_changes.reindex(common_countries).fillna(0.0)

            # 3. Calculate Correlation
            correlation_val = np.nan
            p_value_val = np.nan
            num_correlated_countries = 0

            if len(actual_vals_for_corr) > 1 and len(predicted_vals_for_corr) > 1:
                num_correlated_countries = len(actual_vals_for_corr)
                # Check for zero variance to avoid warnings/errors from pearsonr
                if np.std(actual_vals_for_corr) > 1e-9 and np.std(predicted_vals_for_corr) > 1e-9:
                    correlation_val, p_value_val = pearsonr(actual_vals_for_corr, predicted_vals_for_corr)
                else:
                    print(f"    Skipping correlation for {measure_name} - {policy_code}: Zero variance in actual or predicted changes for {num_correlated_countries} countries.")
            else:
                 print(f"    Skipping correlation for {measure_name} - {policy_code}: Not enough common data points ({len(actual_vals_for_corr)}).")

            correlation_analysis_results[measure_name][policy_code] = {
                "correlation": correlation_val,
                "p_value": p_value_val,
                "num_countries_correlated": num_correlated_countries
            }
            print(f"  Policy {policy_code}: Correlation = {correlation_val:.4f} (p={p_value_val:.4f}, N={num_correlated_countries})")
            

    return correlation_analysis_results


# --- Main script entry point ---
def main():
    # Use the predefined EU_MEMBERS set as the list of countries
    countries_list = sorted(countries)
    # Run the core analysis
    results = analyze_sectoral_contribution_to_change(countries_list)
    # Print a summary of correlation results per measure and policy
    for measure, policies in results.items():
        print(f"\n=== {measure} ===")
        if isinstance(policies, dict):
            for policy_code, stats in policies.items():
                if isinstance(stats, dict):
                    corr = stats.get("correlation", float('nan'))
                    pval = stats.get("p_value", float('nan'))
                    n = stats.get("num_countries_correlated", 0)
                    print(f"{policy_code}: correlation={corr:.4f}, p_value={pval:.4f}, N={n}")
                else:
                    print(f"{policy_code}: {stats}")
        else:
            print(policies)

    # --- Visualization: Grouped bar chart of correlation coefficients ---
    import matplotlib.pyplot as plt
    # Prepare data for grouped bar chart
    measure_names = []
    policy_codes = ['BC', 'EU', 'GL']
    bar_heights = {policy: [] for policy in policy_codes}
    for measure, policies in results.items():
        measure_names.append(measure)
        for policy in policy_codes:
            val = np.nan
            if isinstance(policies, dict) and policy in policies and isinstance(policies[policy], dict):
                val = policies[policy].get("correlation", np.nan)
            bar_heights[policy].append(val)
    x = np.arange(len(measure_names))
    width = 0.2
    fig, ax = plt.subplots()
    for i, policy in enumerate(policy_codes):
        ax.bar(x + i * width, bar_heights[policy], width=width, label=policy)
    ax.set_xticks(x + width)
    ax.set_xticklabels(measure_names)
    ax.set_ylabel("Correlation coefficient (actual vs predicted)")
    ax.set_title("Correlation between actual and predicted country-level changes\nfor each measure and policy")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Scatter plots: predicted vs. actual changes for each measure and policy
    policies = ['BC', 'EU', 'GL']
    # Define loader mapping for each network measure
    measure_loaders_map = {
        'Clustering': load_clustering_data,
        'Betweenness': load_betweenness_data,
        'HubScore': lambda: load_hub_authority_data(score_type='hub_score'),
        'AuthorityScore': lambda: load_hub_authority_data(score_type='authority_score')
    }
    # Create 2x2 grid of scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (measure_name, loader_fn) in zip(axes.flatten(), measure_loaders_map.items()):
        # Load baseline and policy series
        base_series, policy_series_map = loader_fn()
        base_series = base_series.fillna(0.0)
        # Compute baseline sums per country
        baseline_sums = compute_country_sum_measure(base_series, countries_list)
        # For each policy, compute and plot actual vs predicted
        for policy in policies:
            series_p = policy_series_map.get(policy)
            if series_p is None:
                continue
            series_p = series_p.fillna(0.0)
            # Actual change
            policy_sums = compute_country_sum_measure(series_p, countries_list)
            actual = policy_sums - baseline_sums
            # Predicted change
            all_sectors = get_all_sectors_from_series([base_series, series_p])
            avg_changes = calculate_average_sector_change_across_countries(
                base_series, series_p, all_sectors, countries_list
            )
            predicted_map = {}
            for country in countries_list:
                # sectors present in baseline for this country
                country_sectors = [s for s in all_sectors if f"{country}_{s}" in base_series.index]
                predicted_map[country] = calculate_predicted_country_change(
                    country, base_series, avg_changes, country_sectors
                )
            predicted = pd.Series(predicted_map)
            # Align indices and plot scatter
            common = actual.index.intersection(predicted.index)
            ax.scatter(predicted.reindex(common), actual.reindex(common), label=policy)
        ax.set_xlabel('Predicted change')
        ax.set_ylabel('Actual change')
        ax.set_title(f'{measure_name}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot correlations
    measures = list(results.keys())
    policies = ['BC', 'EU', 'GL']
    corr_data = {
        policy: [
            results[measure][policy]['correlation']
            if isinstance(results[measure], dict) and policy in results[measure] and isinstance(results[measure][policy], dict)
            else float('nan')
            for measure in measures
        ]
        for policy in policies
    }

    x = np.arange(len(measures))
    width = 0.2
    fig, ax = plt.subplots()
    for i, policy in enumerate(policies):
        ax.bar(x + i*width, corr_data[policy], width, label=policy)
    ax.set_xticks(x + width)
    ax.set_xticklabels(measures, rotation=45, ha='right')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation between actual and predicted changes by measure and policy')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
