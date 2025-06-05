

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def plot_world_map(df, x_label, y_label, title):
    """
    Plot a world map choropleth based on values in df.
    df: Pandas DataFrame where the first column is ISO_A3 country codes (3-letter), and the second column is numeric values.
    x_label: label for x-axis
    y_label: label for y-axis
    title: title of the map
    """
    # Path to the Natural Earth shapefile (assumes a 'Map' folder in the same directory)
    shapefile_path = './Map/ne_110m_admin_0_countries.shp'
    # Carica il GeoDataFrame del mondo
    world = gpd.read_file(shapefile_path)

    # Assume first column of df is ISO codes, second is il valore da plottare
    iso_col = df.columns[0]
    value_col = df.columns[1]

    data = df.copy()
    # Rinomina solo la colonna dei valori su 'value'
    data = data.rename(columns={value_col: 'value'})

    # Unisci shapefile e dati usando ADM0_A3 da 'world' e la prima colonna di 'data'
    merged = world.merge(data, left_on='ADM0_A3', right_on=iso_col, how='left')

    # Crea la mappa
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(column='value', ax=ax, legend=True, missing_kwds={'color': 'lightgrey', 'label': 'No Data'})

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    shapefile_path = './Map/ne_110m_admin_0_countries.shp'
    output_csv = 'world_iso_codes.csv'
    try:
        world = gpd.read_file(shapefile_path)
        print("All columns in the shapefile:")
        for col in world.columns:
            print(col)
        # Write all SOV_A3 codes to CSV
        sov_a3_codes = world[['SOVEREIGNT']]
        sov_a3_codes.to_csv(output_csv, index=False)
        print(f"SOV_A3 codes written to {output_csv}") 
    except Exception as e:
        print(f"Errore: {e}")
    # Esempio di utilizzo
    sample_data = {
        'ISO_3': ['USA', 'CAN', 'MEX', 'BRA', 'RUS', 'CHN', 'AUS', 'IND', 'ZAF', 'FRA', 'NOR'],
        'value': [10, 20, 15, 25, 30, 35, 5, 40, 12, 40, 7]
    }
    df_sample = pd.DataFrame(sample_data)

    plot_world_map(df_sample, 'Longitude', 'Latitude', 'Sample World Map')