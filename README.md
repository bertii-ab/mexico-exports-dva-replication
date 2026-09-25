# Replication Package: Asymmetric Effects of Real Exchange Rate Movements on Domestic Value Added in Mexican Exports

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Thesis: CIDE Digital Repository](https://img.shields.io/badge/Thesis-CIDE%20Handle%2011651%2F6699-crimson.svg)](https://repositorio-digital.cide.edu/handle/11651/6699)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://zenodo.org)

**Author:** Bertin Yair Acosta Bautista  
**Advisor:** Dr. Rodrigo Aliphat Rodríguez  
**Institution:** Centro de Investigación y Docencia Económicas, A.C. (CIDE)  
**Degree:** Master's Thesis (*Tesis de Maestría en Economía*)  
**Year:** 2026  
**Repository URI:** [http://hdl.handle.net/11651/6699](https://repositorio-digital.cide.edu/handle/11651/6699)  
**PDF Full Text:** [CIDE DSpace PDF](http://repositorio-digital.cide.edu/bitstream/11651/6699/1/TESIS_BYAB.pdf)

---

## 📌 Overview & Abstract

This replication package contains the full dataset, machine learning disaggregation pipelines, econometric models, and visualization scripts supporting the thesis:

> **Abstract:** This thesis examines the asymmetric transmission of real effective exchange rate (REER) movements to domestic value added (DVA) embodied in Mexican manufacturing exports during 2012–2022. To circumvent the limitation of the annual frequency of OECD Trade in Value Added (TiVA) tables, an **Artificial Neural Network Mixed-Data Sampling (ANN-MIDAS)** model is formulated to temporally disaggregate the DVA series to quarterly frequency. The quarterly series is subsequently employed in a **Non-linear Autoregressive Distributed Lag (NARDL)** framework to assess short- and long-run asymmetric dynamics. The empirical findings reject the classical symmetry hypothesis: the long-run appreciation multiplier is nearly triple the depreciation multiplier. This asymmetry demonstrates that the structure of imported intermediate inputs shared in Global Value Chains (GVCs) and dominant currency pricing (DCP) generates cost-push pressures that offset traditional expenditure-switching gains, implying that exchange rate depreciations cannot substitute for targeted industrial policies.

---

## 📊 Key Empirical Findings

1. **ANN-MIDAS Temporal Disaggregation:** Successfully bridges low-frequency OECD TiVA tables with high-frequency monthly manufacturing and trade indicators, outperforming standard linear and Denton benchmark disaggregations while preserving accounting identities.
2. **Asymmetric Pass-Through (NARDL):** Currency depreciations fail to stimulate domestic value added proportionally due to the high import content of manufacturing exports. Appreciations, conversely, have a pronounced impact, yielding a long-run multiplier ratio of nearly 3:1.
3. **Causality Dynamics:** Toda-Yamamoto Granger causality tests confirm robust non-linear directional transmission from exchange rate shocks to domestic value added.

---

## 📁 Repository Structure

```text
├── CITATION.cff                      # Standard citation metadata for GitHub
├── LICENSE                           # MIT License
├── README.md                         # Project documentation and guide
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # Master replication script
├── data/                             # Curated datasets
│   ├── Interpolated_TiVA_DVA_Final.csv
│   ├── Interpolated_TiVA_DVA_Circularity.csv
│   ├── global_supply_chain_pressure_index.csv
│   ├── master_annual.csv
│   ├── master_monthly.csv
│   ├── master_quarterly.csv
│   └── mexico_monthly_eer_bis_official.csv
├── scripts/                          # Analysis & estimation pipeline
│   ├── stage1_data_acquisition/     # API extraction (OECD SDMX, BIS)
│   │   └── data_extraction_mt.py
│   ├── stage2_ann_midas/             # Neural network temporal disaggregation
│   │   ├── ann_midas.py
│   │   └── ann_midas_circularity.py
│   └── stage3_econometrics/          # NARDL & Toda-Yamamoto models
│       ├── nardl.py
│       ├── nardl_circularity.py
│       ├── toda_yamamoto.py
│       ├── toda_yamamoto_asymmetric.py
│       ├── compute_irf.py
│       └── bootstrap_irf.py
└── results/                          # Output figures and estimates
    ├── circularity_comparison.png
    ├── plot_ann_vs_annual.png
    ├── plot_ann_vs_linear.png
    ├── plot_dva_reer_timeseries.png
    ├── plot_irf_dynamic_multipliers.csv
    ├── plot_partial_sums.png
    ├── plot_structural_break.png
    └── plot_validation_yearly.png
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.9 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/bertii-ab/mexico-exports-dva-replication.git
cd mexico-exports-dva-replication
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Reproduction Workflow

### Quick Run
To run the full end-to-end replication pipeline:
```bash
python run_pipeline.py
```

### Modular Execution

1. **Data Acquisition (OECD & BIS):**
   ```bash
   python scripts/stage1_data_acquisition/data_extraction_mt.py
   ```

2. **Temporal Disaggregation (ANN-MIDAS):**
   ```bash
   python scripts/stage2_ann_midas/ann_midas.py
   ```

3. **Econometric Estimation (NARDL & Asymmetric Multipliers):**
   ```bash
   python scripts/stage3_econometrics/nardl.py
   ```

4. **Toda-Yamamoto Causality Tests:**
   ```bash
   python scripts/stage3_econometrics/toda_yamamoto.py
   ```

All figures and CSV multiplier outputs will be written to the `results/` directory.

---

## 📖 Citation

If you use this replication package, data, or methodology in your research, please cite:

### BibTeX
```bibtex
@mastersthesis{acosta2026asymmetric,
  author       = {Acosta Bautista, Bertin Yair},
  title        = {Asymmetric effects of real exchange rate movements on domestic value added in Mexican exports},
  school       = {Centro de Investigaci{\'o}n y Docencia Econ{\'o}micas (CIDE)},
  year         = {2026},
  address      = {Mexico City, Mexico},
  url          = {https://repositorio-digital.cide.edu/handle/11651/6699},
  note         = {Replication package: \url{https://github.com/bertii-ab/mexico-exports-dva-replication}}
}
```

### APA
> Acosta Bautista, B. Y. (2026). *Asymmetric effects of real exchange rate movements on domestic value added in Mexican exports* [Master's thesis, Centro de Investigación y Docencia Económicas (CIDE)]. CIDE Digital Repository. https://repositorio-digital.cide.edu/handle/11651/6699

---

## 🏷️ Zenodo DOI Minting Instructions

To generate a permanent DOI for this repository via Zenodo:
1. Log in to [Zenodo](https://zenodo.org) using your GitHub account (`bertii-ab`).
2. Go to **GitHub Settings** in Zenodo (`https://zenodo.org/account/settings/github/`).
3. Toggle the switch for `bertii-ab/mexico-exports-dva-replication` to **ON**.
4. In GitHub, create a new Release:
   - Tag version: `v1.0.0`
   - Release title: `v1.0.0 - Thesis Replication Release`
5. Zenodo will automatically archive the repository snapshot and assign a permanent citable **DOI**.
6. Replace the placeholder badge in this `README.md` with your generated Zenodo badge markdown.

---

## 📄 License
This codebase is licensed under the [MIT License](LICENSE). The curated datasets and figures are licensed under Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
