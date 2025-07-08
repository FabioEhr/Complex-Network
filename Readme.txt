
BetweennessCentralityMay.py computes the Betweenness Centrality on the sparsified matrices

ChartOfCountries.py loads network‐analysis data for 2010–2020 and three policy scenarios, computes per‐country averages (simple or weighted) of four metrics (clustering, betweenness, hub and authority), and outputs side-by-side comparison tables as PNG images.

CountrySectorAnalysis.py Aggregates network‐analysis metrics across 2010–2020 baselines and three policy scenarios and generates comparative tables and detailed text reports to highlight sectoral drivers and support economic interpretation of trade‐network impacts.

Output in percentage.py Computes country-level net trade flow changes from 2010–2020 baselines and three policy scenarios, then visualizes average trends and percentage variations on a 2×2 world‐map grid and correlates global policy impacts with carbon intensity via a scatterplot.

<<<<<<< Updated upstream
Aggregates node-level network metrics (clustering coefficient, betweenness centrality, hub and authority scores) into country-level averages—either simple or weighted by net trade flows—and generates 2×2 world-map grids comparing baseline 2020 values with three policy scenarios, saving each metric’s visualization as a PNG.
## Elenco dei 
=======
Visualising Chart of Countries.py Aggregates node-level network metrics (clustering coefficient, betweenness centrality, hub and authority scores) into country-level averages—either simple or weighted by net trade flows—and generates 2×2 world-map grids comparing baseline 2020 values with three policy scenarios, saving each metric’s visualization as a PNG.

World_map.py Defines two functions to create choropleth maps from GeoPandas—plot_world_map for a single map and plot_world_maps_grid for a 2×2 grid—by merging a Natural Earth shapefile with a DataFrame of ISO-A3 country codes and numeric values. The script’s main block also extracts sovereign country codes to a CSV and demonstrates the mapping functions using sample data.


- API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv is the data with the carbon intensities that are sourced from the World Development Indicators and were last updated on July 1, 2025.

- authority_tables.png betweenness_tables.png clustering_tables.png hub_tables are the results of the measurements aggregated at a country level for the gl, eu and bc scenario

# Panoramica del progetto

Questo repository contiene script per calcolare e visualizzare metriche di rete (hub/authority, clustering coefficient, betweenness centrality) su scala mondiale, usando dati di base (2010–2020) e scenari di policy (bc, eu, gl, cbam).

## Elenco dei principali script
>>>>>>> Stashed changes

Base_measures.jpynb
analizza le misure principali sui network non sparsificati 

Z_analysis.ipynb
Nella fine del codice c'è un'analisi interessante



- **AnalysisOfBC.py**  
  Analizza la _betweenness centrality_ (baseline 2010–2020) e tre scenari di policy (gl, eu, cbam).  
  - Filtra settori con zero/NaN o σ = 0.  
  - Calcola z-score tra scenario e baseline 2020.  
  - Pondera le differenze con i net inflows 2018 per paese e produce mappe.  
  - Stampa statistiche aggiuntive (top/bottom 20, NaN, medie) e salva un CSV di z-score.



<<<<<<< Updated upstream
- API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv is the data with the carbon intensities that are sourced from the World Development Indicators and were last updated on July 1, 2025.

- authority_tables.png betweenness_tables.png clustering_tables.png hub_tables are the results of the measurements aggregated at a country level for the gl, eu and bc scenario


######################## Panoramica ############################

Questo repository contiene script per calcolare e visualizzare metriche di rete (hub/authority, clustering coefficient, betweenness centrality) su scala mondiale, usando dati di base (2010–2020) e scenari di policy (bc, eu, gl, cbam). La cartella "Code_and_dataset" contiene i dataset su cui abbiamo lavorato e i file utilizzati per l'analisi. La cartella "results" contiene i risultati delle analisi raccolti (in file parquet) e i grafici finali. In particolare:

>> *** Code_and_dataset *** contiene:
   > i codici:
      - **Filter.jpynb** :
        Filtra il dataset iniziale (condiviso separatamente), creando le matrici d'adiacenza con il nodo Rest of the World (RoW) (dfz_{label}) (quest'ultimo creato dalle nazioni del mondo che hanno dati simulati          nel database GLORIA).Infine crea le matrici aggregate a livello delle nazioni, sommando sui settori (dfz_{label}_c).

      - ** Z_sparser.jpynb**
      Rende sparse le matrici d'adiacenza attraverso la funzione sparsify_by_sector_inflow() e le salva (dfz_s_{label}). Calcola il clustering coefficient per tali matrici e lo salva (clus_{label}).

      - **Base measures.jpynb**
      Fornisce una prima analisi per weights and strengths distributions (calcola media e deviazione standard, mostra le distribuzioni) (salvati in results/base_measures), mostra hub e authorities aggregate per         nazioni e salva tali misure (hub_aut_{label}).

      - **Stat_analysis_final.jpynb**
      Contiene il codice per calcolare, mostrare e salvare i risultati (e i grafici ove possibile) di: quattro momenti delle distribuzioni delle misure(salvati in results\four_momenta), probabilità condizionate e       istogramma di comparazione 2010-2020 (salvati in results\avg_condit_probab), M_values (salvati in results\M_values), coefficienti di correlazione (calcolati sia da matrici complete che sparsificate per            misure riguardanti i pesi, calcolati da matrici sparsificate per misure riguardanti il degree) (salvati in results\correlations) , variazioni relative (tra policy e 2020) delle misure per le varie nazioni         (salvati in results\rel_diff_over_avg_rel_diff e esults\rel_diff_over_std) , SRCC. 
   
   > i datasets:
      - matrici di adiacenza W: dfz_{label}(totali), dfz_{label}_c(aggregate per nazione) , dfz_s_{label} (sparse),
      - valori di clustering: clus_{label},
      - valori della betweenees: dfz_s_{label}_bc
      -hub e authority: hub_aut{label}

>> *** results *** contiene le cartelle con i vari risultati e grafici utilizzati, raccolti per tipologia di analisi.
=======
>>>>>>> Stashed changes
