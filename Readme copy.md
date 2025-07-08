
# Complex Network Analysis

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

**Analysis and Visualization of Global Trade Network Metrics (2010–2020) under Policy Scenarios**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [Scripts](#scripts)
- [Directory Structure](#directory-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This project computes and visualizes network metrics—**clustering coefficient**, **betweenness centrality**, **hub/authority scores**—on global trade data from 2010 to 2020. It also compares baseline metrics with three policy scenarios: **Global Liberalization (gl)**, **EU-centric (eu)**, and **CBAM (cbam)**.

![Workflow Diagram](docs/workflow.png)

## Features
- Compute node-level network metrics on original and sparsified adjacency matrices.
- Aggregate metrics at the country level (simple and flow-weighted averages).
- Generate 2×2 world-map grids and choropleth maps for scenario comparisons.
- Perform z-score analysis and sectoral impact assessments.
- Correlate policy impacts with carbon intensity.

## Prerequisites
- Python 3.8+
- GeoPandas, NetworkX, Matplotlib, Pandas
- Download and place the World Development Indicators CSV in `data/`

## Installation
```bash
git clone https://github.com/yourusername/Complex-Network.git
cd Complex-Network
pip install -r requirements.txt
```

## Data
- `API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv` — Carbon intensity data (last updated July 1, 2025)
- Adjacency matrices and processed datasets in `Code_and_dataset/`

## Usage

### Scripts

| Script | Description |
| ------ | ----------- |
| `BetweennessCentralityMay.py` | Compute betweenness centrality on sparsified matrices. |
| `ChartOfCountries.py` | Load data (2010–2020 & scenarios), compute country averages, and export comparison tables as PNG. |
| `CountrySectorAnalysis.py` | Aggregate metrics across years/scenarios; generate comparative tables and detailed text reports. |
| `Output in percentage.py` | Visualize net trade flow changes on 2×2 world-map grid; scatterplot with carbon intensity. |
| `World_map.py` | Functions to plot choropleth maps using GeoPandas and Natural Earth shapefile. |
| `AnalysisOfBC.py` | Z-score analysis of betweenness centrality; filter sectors; generate CSV and maps. |
| `Base_measures.ipynb` | Exploratory analysis of weights and strength distributions. |
| `Z_analysis.ipynb` | Supplemental analysis with additional insights. |

## Directory Structure
```
Complex-Network/
├── Code_and_dataset/
│   ├── dfz_* (adjacency matrices)
│   ├── clus_* (clustering data)
│   └── hub_aut_* (hub/authority data)
├── results/
│   ├── base_measures/
│   ├── four_momenta/
│   ├── avg_condit_probab/
│   ├── M_values/
│   ├── correlations/
│   └── ... (other result folders)
├── docs/
│   └── workflow.png
├── requirements.txt
└── README.md
```

## Results
All generated tables (PNG and CSV) are stored in `results/`. Example outputs:

![Example Map](results/example_map.png)
![Comparison Table](results/comparison_table.png)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or feedback, contact [your.email@example.com].