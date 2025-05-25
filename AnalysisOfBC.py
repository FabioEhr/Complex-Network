import glob
import pandas as pd

def main():
    # Find all betweenness output files
    parquet_files = sorted(glob.glob("*_bc.parquet"))
    if not parquet_files:
        print("No *_bc.parquet files found.")
        return

    for pq in parquet_files:
        # Read saved centrality DataFrame
        df_bc = pd.read_parquet(pq)

        # Identify numeric columns
        numeric_cols = df_bc.select_dtypes(include='number').columns.tolist()
        if 'betweenness' in numeric_cols:
            central_col = 'betweenness'
            print(f"Using 'betweenness' column as centrality measure for {pq}.")
        elif len(numeric_cols) == 1:
            central_col = numeric_cols[0]
            print(f"Using single numeric column '{central_col}' as centrality measure.")
        else:
            # Skip files that are not centrality outputs
            print(f"Skipping {pq}: multiple numeric columns found ({len(numeric_cols)}), not a centrality file.")
            continue

        # Extract the centrality series and ensure numeric dtype
        bc_series = pd.to_numeric(df_bc[central_col], errors='coerce').dropna()
        
        # Get top 5 and bottom 5 nodes
        top5 = bc_series.nlargest(5)
        low5 = bc_series.nsmallest(5)
        
        # Print results
        print(f"\nResults for {pq}:")
        print(" Top 5 nodes by betweenness:")
        for node, val in top5.items():
            print(f"   {node}: {val:.6f}")
        print(" Bottom 5 nodes by betweenness:")
        for node, val in low5.items():
            print(f"   {node}: {val:.6f}")

if __name__ == "__main__":
    main()