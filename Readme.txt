Base_measures.jpynb
analizza le misure principali sui network non sparsificati 

Z_analysis.ipynb
Nella fine del codice c'è un'analisi interessante

BetweennessCentralityMay.py computes the Betweenness Centrality on the sparsified matrices

ChartOfCountries.py loads network‐analysis data for 2010–2020 and three policy scenarios, computes per‐country averages (simple or weighted) of four metrics (clustering, betweenness, hub and authority), and outputs side-by-side comparison tables as PNG images.

CountrySectorAnalysis.py Aggregates network‐analysis metrics across 2010–2020 baselines and three policy scenarios and generates comparative tables and detailed text reports to highlight sectoral drivers and support economic interpretation of trade‐network impacts.

Output in percentage.py Computes country-level net trade flow changes from 2010–2020 baselines and three policy scenarios, then visualizes average trends and percentage variations on a 2×2 world‐map grid and correlates global policy impacts with carbon intensity via a scatterplot.

Aggregates node-level network metrics (clustering coefficient, betweenness centrality, hub and authority scores) into country-level averages—either simple or weighted by net trade flows—and generates 2×2 world-map grids comparing baseline 2020 values with three policy scenarios, saving each metric’s visualization as a PNG.

######################## Panoramica

Questo repository contiene script per calcolare e visualizzare metriche di rete (hub/authority, clustering coefficient, betweenness centrality) su scala mondiale, usando dati di base (2010–2020) e scenari di policy (bc, eu, gl, cbam). La cartella 

## Elenco dei 

- **World_map.py**  
  Funzione generica per tracciare un _choropleth_ mondiale dati codici ISO A3 e valori numerici.  
  Ha inoltre un blocco di esempio per ispezionare lo shapefile e produrre un CSV di codici country.

- **HubsAutoritiesVisualizer.py**  
  Calcola z-score di _hub_score_ e _authority_score_ (baseline 2010–2020) per ciascuna policy (bc, eu, gl), li pondera con i net inflows del 2018 e genera mappe mondiali separate per hub e authority.

- **ClusVisualizer.py**  
  Simile a HubsAutoritiesVisualizer, ma per il _Clustering_Coefficient_.  
  - Carica i file `clus_{anno}.parquet` (2010–2020), filtra nodi con zeri/NaN o σ = 0.  
  - Calcola z-score per ciascuna policy (bc, eu, gl).  
  - Pondera ogni z-score con i net inflows 2018 per paese e disegna la mappa.

- **AnalysisOfBC.py**  
  Analizza la _betweenness centrality_ (baseline 2010–2020) e tre scenari di policy (gl, eu, cbam).  
  - Filtra settori con zero/NaN o σ = 0.  
  - Calcola z-score tra scenario e baseline 2020.  
  - Pondera le differenze con i net inflows 2018 per paese e produce mappe.  
  - Stampa statistiche aggiuntive (top/bottom 20, NaN, medie) e salva un CSV di z-score.

- **Filter.jpynb**
   Filtra il dataset iniziale, creando le matrici d'adiacenza con il nodo Rest of the World (RoW) (quest'ultimo creato dalle nazioni del mondo che hano dati simulati nel database GLORIA). Infine crea le matrici      aggregate a livello delle nazioni, sommando sui settori.

-** Z_sparser.jpynb**
    Rende sparse le matrici d'adiacenza attraverso la funzione sparsify_by_sector_inflow() e le salva. Calcola il clustering coefficient per tali matrici e lo salva.

-**Base measures.jpynb**
   Fornisce una prima analisi per weights and strengths distributions (calcola media e deviazione standard, mostra le distribuzioni), mostra hub e authorities aggregate per nazioni e salva tale misure.

-**Stat_analysis_final.jpynb**
  Contiene il codice per calcolare, mostrare e salvare i risultati (e i grafici ove possibile) di: quattro momenti delle distribuzioni delle misure, probabilità condizionate e istogramma di comparazione 2010-       2020, M_values, coefficienti di correlazione (calcolati sia da matrici complete che sparsificate per misure riguardanti i pesi, calcolati da matrici sparsificate per misure riguardanti il degree), variazioni 
  relative (tra policy e 2020) delle misure per le varie nazioni, SRCC. 

- API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv is the data with the carbon intensities that are sourced from the World Development Indicators and were last updated on July 1, 2025.

- authority_tables.png betweenness_tables.png clustering_tables.png hub_tables are the results of the measurements aggregated at a country level for the gl, eu and bc scenario
