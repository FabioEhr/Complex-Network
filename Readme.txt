Base_measures.jpynb
analizza le misure principali sui network non sparsificati 

Z_analysis.ipynb
Nella fine del codice c'è un'analisi interessante

BetweennessCentralityMay.py
file che calcola la Betweenness Centrality

AnalysisOfBC.py
File che dà una prima lettura dei risultati della Betweenness Centrality


# Panoramica del progetto

Questo repository contiene script per calcolare e visualizzare metriche di rete (hub/authority, clustering coefficient, betweenness centrality) su scala mondiale, usando dati di base (2010–2020) e scenari di policy (bc, eu, gl, cbam).

## Elenco dei principali script

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

---
