# Overview

This repository contains scripts to compute and visualize network metrics (hub/authority, clustering coefficient, betweenness centrality) on a global scale, using baseline data (2010–2020) and policy scenarios (bc, eu, gl, cbam). The folder `Code_and_dataset` contains the datasets we worked on and the files used for the analysis. The `results` folder contains the outputs of the analyses (in parquet files) and the final plots.

## `Code_and_dataset`

This folder contains:

### The scripts:

*   **`Filter.ipynb`**:
    Filters the initial dataset (shared separately), creating adjacency matrices including the Rest of the World (RoW) node (`dfz_{label}`) (this node is created from countries in the world that have simulated data in the GLORIA database). It also creates matrices aggregated at the country level by summing over sectors (`dfz_{label}_c`).

*   **`Z_sparser.ipynb`**:
    Makes the adjacency matrices sparse using the function `sparsify_by_sector_inflow()` and saves them (`dfz_s_{label}`). Computes the clustering coefficient for these matrices and saves the results (`clus_{label}`).

*   **`Base measures.ipynb`**:
    Provides an initial analysis for weights and strengths distributions (computes mean and standard deviation, displays the distributions) (saved in `results/base_measures`), displays aggregated hub and authority measures by country, and saves those measures (`hub_aut_{label}`).

*   **`Stat_analysis_final.ipynb`**:
    Contains the code to compute, display, and save the results (and plots where possible) of: four moments of the metric distributions (saved in `results\four_momenta`), conditional probabilities and comparison histogram 2010–2020 (saved in `results\avg_condit_probab`), M_values (saved in `results\M_values`), correlation coefficients (computed from both complete and sparsified matrices for weight-related measures, computed from sparsified matrices for degree-related measures) (saved in `results\correlations`), relative variations (between policy scenarios and 2020) of the measures for various countries (saved in `results\rel_diff_over_avg_rel_diff` and `results\rel_diff_over_std`), SRCC.

*   **`BetweennessCentralityMay.py`**:
    Computes the Betweenness Centrality on the sparsified matrices.

*   **`ChartOfCountries.py`**:
    Loads network-analysis data for 2010–2020 and three policy scenarios, computes per-country sum of four metrics (clustering, betweenness, hub and authority) over all sectors, and outputs side-by-side comparison tables as PNG images.

*   **`CountrySectorAnalysis.py`**:
    Aggregates network-analysis metrics across 2010–2020 baselines and three policy scenarios and generates comparative tables and detailed text reports to highlight sectoral drivers and support economic interpretation of trade-network impacts.

*   **`Output in percentage.py`**:
    Computes country-level net trade flow changes from 2010–2020 baselines and three policy scenarios, then visualizes average trends and percentage variations on a 2×2 world-map grid and correlates global policy impacts with carbon intensity via a scatterplot.

*   **`Visualising Chart of Countries.py`**:
    Aggregates node-level network metrics (clustering coefficient, betweenness centrality, hub and authority scores) into country-level averages—either simple or weighted by net trade flows—and generates 2×2 world-map grids comparing baseline 2020 values with three policy scenarios, saving each metric’s visualization as a PNG.

*   **`World_map.py`**:
    Defines two functions to create choropleth maps from GeoPandas—`plot_world_map` for a single map and `plot_world_maps_grid` for a 2×2 grid—by merging a Natural Earth shapefile with a DataFrame of ISO-A3 country codes and numeric values. The script’s main block also extracts sovereign country codes to a CSV and demonstrates the mapping functions using sample data.

### The datasets:

*   **Adjacency matrices W**: `dfz_{label}` (total), `dfz_{label}_c` (aggregated by country), `dfz_s_{label}` (sparse)
*   **Clustering values**: `clus_{label}`
*   **Betweenness values**: `dfz_s_{label}_bc`
*   **Hub and authority**: `hub_aut_{label}`
*   **`API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv`**: This is the data with the carbon intensities that are sourced from the World Development Indicators and were last updated on July 1, 2025.

## `results`

This folder contains subfolders with the various results and plots used, organized by type of analysis.