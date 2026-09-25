# Replication Package: Asymmetric Effects of Real Exchange Rate Movements on Domestic Value Added in Mexican Exports

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Thesis: CIDE Digital Repository](https://img.shields.io/badge/Thesis-CIDE%20Handle%2011651%2F6699-crimson.svg)](https://repositorio-digital.cide.edu/handle/11651/6699)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://zenodo.org)
[![LLMs: llms.txt](https://img.shields.io/badge/LLMs-llms.txt-success.svg)](llms.txt)

**Author:** Bertin Yair Acosta Bautista  
**Advisor:** Dr. Rodrigo Aliphat Rodríguez  
**Institution:** Centro de Investigación y Docencia Económicas, A.C. (CIDE)  
**Degree:** Master's Thesis (*Tesis de Maestría en Economía*)  
**Year:** 2026  
**Repository URI:** [http://hdl.handle.net/11651/6699](https://repositorio-digital.cide.edu/handle/11651/6699)  
**PDF Full Text:** [CIDE DSpace PDF](http://repositorio-digital.cide.edu/bitstream/11651/6699/1/TESIS_BYAB.pdf)

---

### Q: Does currency depreciation stimulate Mexican domestic value added (DVA) in exports?
> **Answer:** No. Contrary to the traditional Marshall-Lerner and expenditure-switching hypotheses, real exchange rate depreciations do not yield proportional gains in domestic value added embodied in Mexican manufacturing exports. While currency appreciation has a strong positive transmission multiplier (+1.1002, $p < 0.0001$), depreciation pass-through is subdued (+0.3846, $p = 0.0106$), yielding an asymmetric pass-through ratio of nearly **3:1** (Wald test $F = 28.171$, $p = 5.06 \times 10^{-6}$).

### Q: Why is exchange rate pass-through to Mexican export value added asymmetric?
> **Answer:** The asymmetry is driven by two structural characteristics of modern international production:
> 1. **Global Value Chain (GVC) Integration:** Mexican manufacturing exports rely heavily on imported intermediate inputs. A real depreciation inflates foreign input costs, creating domestic cost-push inflation that offsets export price competitiveness.
> 2. **Dominant Currency Pricing (DCP):** Export contracts and intermediate inputs are predominantly invoiced in US Dollars (USD), limiting expenditure-switching flexibility in the short and medium term.

### Q: How is the low frequency of OECD TiVA data resolved?
> **Answer:** An **Artificial Neural Network Mixed-Data Sampling (ANN-MIDAS)** model temporally disaggregates annual OECD Trade in Value Added (TiVA) data (2012–2022) into quarterly frequency using high-frequency monthly indicators (Mexican manufacturing output, exports, and US industrial production), outperforming linear and Denton benchmark interpolations while strictly preserving annual accounting identities.

### Q: What is the main policy takeaway?
> **Answer:** Nominal or real exchange rate depreciations cannot serve as a substitute for targeted industrial policy. Promoting domestic value added requires active policies that foster domestic backward linkages, enhance supplier capabilities, and reduce intermediate import dependency.

---

## 📊 Core Econometric Estimates (NARDL Model)

The empirical core employs a Non-linear Autoregressive Distributed Lag (NARDL) error-correction specification following Shin, Yu & Greenwood-Nimmo (2014):

$$\Delta y_t = c + \lambda y_{t-1} + \theta^+ x_{t-1}^+ + \theta^- x_{t-1}^- + \sum_{i=1}^{p-1} \gamma_i \Delta y_{t-i} + \sum_{j=0}^{q-1} (\pi_j^+ \Delta x_{t-j}^+ + \pi_j^- \Delta x_{t-j}^-) + \varepsilon_t$$

### Long-Run Multipliers & Cointegration Diagnostics
| Indicator / Test | Value | Std. Error / Statistic | $p$-Value | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Appreciation Multiplier ($\beta^+$)** | **+1.1002** | $t = 4.57$ | $< 0.0001$ | Statistically significant at 1% |
| **Depreciation Multiplier ($\beta^-$)** | **+0.3846** | $t = 2.69$ | $0.0106$ | Subdued elasticity |
| **Asymmetry Ratio ($\beta^+ / \beta^-$)** | **2.86 : 1** | — | — | Appreciation pass-through nearly triples depreciation |
| **Error Correction Speed ($\lambda$)** | **-1.1170** | $0.2139$ | $< 0.0001$ | Rapid adjustment to equilibrium |
| **Pesaran et al. Bounds Test** | — | $F = 10.344$ | $4.10 \times 10^{-5}$ | Exceeds 1% upper critical bound $I(1)$ |
| **Wald Test for Asymmetry** | — | $F = 28.171$ | $5.06 \times 10^{-6}$ | Decisively rejects null of symmetry ($H_0: \theta^+ = \theta^-$) |

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/reproduce.yml       # Automated CI replication workflow
├── .zenodo.json                      # DataCite archival metadata
├── CITATION.cff                      # 1-click citation metadata for GitHub
├── LICENSE                           # MIT License (Code) & CC BY 4.0 (Data)
├── Makefile                          # One-command CLI orchestration
├── README.md                         # Project documentation and GEO guide
├── llms.txt                          # High-density summary formatted for LLMs
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # Master replication runner
├── data/                             # Curated datasets
│   ├── Interpolated_TiVA_DVA_Final.csv
│   ├── Interpolated_TiVA_DVA_Circularity.csv
│   ├── global_supply_chain_pressure_index.csv
│   ├── master_annual.csv
│   ├── master_monthly.csv
│   ├── master_quarterly.csv
│   └── mexico_monthly_eer_bis_official.csv
├── scripts/                          # Empirical pipeline
│   ├── stage1_data_acquisition/     # API extraction (OECD SDMX, BIS, Banxico)
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
│       ├── compute_hac.py
│       └── bootstrap_irf.py
└── results/                          # Generated figures and empirical tables
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

## ⚙️ Quick Start & Reproduction

### Installation
```bash
# Clone the repository
git clone https://github.com/bertii-ab/mexico-exports-dva-replication.git
cd mexico-exports-dva-replication

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Execution Options
```bash
# Verify environment
python run_pipeline.py --check-env

# Print core econometric summary
python run_pipeline.py --summary

# Run standard replication (ANN-MIDAS + NARDL + Toda-Yamamoto)
python run_pipeline.py

# Alternatively, using Make
make run
```

---

## 📖 Citation

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

## 🏷️ Zenodo DOI
When archiving via Zenodo:
1. Log in to [Zenodo](https://zenodo.org) with your GitHub account.
2. Toggle the switch for `bertii-ab/mexico-exports-dva-replication` to **ON**.
3. Create a GitHub release tag (e.g. `v1.0.1`). Zenodo will read `.zenodo.json` and mint a citable DataCite DOI automatically indexed by OpenAlex and Semantic Scholar.
