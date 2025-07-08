#non mi ricordo a cosa serva questo file, per me si può cestinare


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# List of Parquet files for baseline years only (2010–2020)
year_files = [
    "dfz_s_2010.parquet",
    "dfz_s_2011.parquet",
    "dfz_s_2012.parquet",
    "dfz_s_2013.parquet",
    "dfz_s_2014.parquet",
    "dfz_s_2015.parquet",
    "dfz_s_2016.parquet",
    "dfz_s_2017.parquet",
    "dfz_s_2018.parquet",
    "dfz_s_2019.parquet",
    "dfz_s_2020.parquet",
]

# Directory where Parquet files are stored
DATA_DIR = "./"  # modifica se necessario

# Dictionary per memorizzare le medie di grado per anno
avg_in_per_year = {}
avg_out_per_year = {}

# Creiamo una cartella per salvare i singoli grafici se non esiste
output_dir = "degree_plots_by_year"
os.makedirs(output_dir, exist_ok=True)

for fname in year_files:
    year = fname.split("_")[2].split(".")[0]  # ad esempio "2010" da "dfz_s_2010.parquet"
    path = os.path.join(DATA_DIR, fname)
    
    # Legge la matrice di adiacenza
    df = pd.read_parquet(path)
    # Assicurati che le righe abbiano lo stesso nome delle colonne
    df.index = df.columns
    # df.shape dovrebbe essere (N, N) con nodi elencati sia in righe che in colonne
    
    # Costruisci la versione binaria: 1 se peso ≠ 0, 0 altrimenti
    binary = (df.values != 0).astype(int)
    
    # Calcola out-degree (somma righe) e in-degree (somma colonne)
    out_deg = binary.sum(axis=1)
    in_deg = binary.sum(axis=0)
    
    # Memorizza le medie
    avg_out_per_year[year] = np.mean(out_deg)
    avg_in_per_year[year] = np.mean(in_deg)
    
    # Stampa nodi con out-degree > 1000 nel 2010, ordinati in modo decrescente
    if year == "2010":
        # Usa sempre i nomi delle colonne come etichette (che corrispondono ai nodi)
        out_series = pd.Series(out_deg, index=df.columns)
        filtered = out_series[out_series > 1000].sort_values(ascending=False)
        print("Nodi con out-degree > 1000 nel 2010 (in ordine decrescente):")
        for node, deg in filtered.items():
            print(f"{node}: {deg}")
    
    # Crea figura con due sottotrame (in-degree a sinistra, out-degree a destra)
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    # Istogramma in-degree
    ax_in.hist(in_deg, bins=50, color="C0", edgecolor="black")
    ax_in.set_title(f"{year} — In-Degree Distribution")
    ax_in.set_xlabel("In-Degree (numero di archi in ingresso)")
    ax_in.set_ylabel("Conteggio di nodi")
    
    # Istogramma out-degree
    ax_out.hist(out_deg, bins=50, color="C1", edgecolor="black")
    ax_out.set_title(f"{year} — Out-Degree Distribution")
    ax_out.set_xlabel("Out-Degree (numero di archi in uscita)")
    ax_out.set_ylabel("Conteggio di nodi")
    
    # Sovra-titolo
    fig.suptitle(f"Distribuzioni di grado per l'anno {year}", fontsize=14, y=1.02)
    
    # Salva la figura in PNG
    output_path = os.path.join(output_dir, f"degree_dist_{year}.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

# Stampa le medie di grado a console
print("Medie per anno (In-Degree / Out-Degree):")
for year in sorted(avg_in_per_year.keys()):
    print(f"  {year}: avg_in = {avg_in_per_year[year]:.2f}, avg_out = {avg_out_per_year[year]:.2f}")

# (Opzionale) Se vuoi un unico grafico che mostri l’evoluzione temporale delle medie:
years = sorted(avg_in_per_year.keys())
avg_in_vals = [avg_in_per_year[y] for y in years]
avg_out_vals = [avg_out_per_year[y] for y in years]

plt.figure(figsize=(8, 4))
plt.plot(years, avg_in_vals, marker="o", label="Avg In-Degree")
plt.plot(years, avg_out_vals, marker="o", label="Avg Out-Degree")
plt.title("Evoluzione delle medie di grado (2010–2020)")
plt.xlabel("Anno")
plt.ylabel("Grado medio")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("avg_degree_trend_2010_2020.png", dpi=200)
plt.close()


print("\nGrafico dell’evoluzione delle medie salvato come 'avg_degree_trend_2010_2020.png'")

# ---------------------------------------------
# Plot istogramma dei pesi in uscita per settori selezionati nel 2010
# ---------------------------------------------

# Leggi il file 2010
df_2010 = pd.read_parquet('./dfz_s_2010.parquet')
# Assicurati che le righe abbiano lo stesso nome delle colonne
df_2010.index = df_2010.columns

# Elenco dei settori di interesse
sectors = [
    "RoW_AGRfor",
    "DEU_MANmac",
    "DEU_MANele",
    "USA_COMpub",
    "DEU_MANmot",
    "RoW_TRAair",
    "DEU_MANfur",
    "DEU_MANpha",
    "DEU_TRAwat"
]

# Prepara una figura 3x3 per gli istogrammi
fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
print(df_2010)
for ax, sector in zip(axes.flatten(), sectors):
    print(f"Plotting sector: {sector}")
    if sector in df_2010.index:
        # Estrai i pesi in uscita per il settore
        row_values = df_2010.loc[sector].values.astype(float)
        # Normalizza in modo che la somma sia 1
        total = row_values.sum()
        if total != 0:
            normalized = row_values / total
        else:
            normalized = row_values  # rimane tutto zero
        # Calcola log(peso normalizzato), escludi zeri per evitare -inf
        positive_mask = normalized > 0
        log_weights = np.log(normalized[positive_mask])
        # Plot istogramma del log dei pesi normalizzati
        ax.hist(log_weights, bins=50, color='C2', edgecolor='black')
        ax.set_title(sector)
        ax.set_xlabel("Log(peso normalizzato)")
        ax.set_ylabel("Conteggio")
        # Calcola soglia in log che separa il 50% della somma totale
        sorted_weights = np.sort(normalized)[::-1]
        cumsum = np.cumsum(sorted_weights)
        # Trova peso alla prima posizione dove cumsum >= 0.5
        idx50 = np.argmax(cumsum >= 0.5)
        threshold_weight = sorted_weights[idx50]
        # Evita di prendere log(0)
        if threshold_weight > 0:
            threshold_log = np.log(threshold_weight)
            ax.axvline(x=threshold_log, color='red', linestyle='--')
    else:
        ax.text(0.5, 0.5, f"{sector} non trovato", ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Distribuzione dei pesi in uscita (normalizzati) per i settori selezionati (2010)", fontsize=16, y=1.02)
plt.show()