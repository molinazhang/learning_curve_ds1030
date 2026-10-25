# Predicting AP Participation in U.S. Public High Schools

Team Learning Curve — DATA1030: Hands-on Data Science, Brown University

**Team members:** Seungyeon Jung, Nfamara Dabo, Jiajin Zhang, and Xueyi Huang.

## Project overview

Research Question: How well can school socioeconomic, geographic, and organizational characteristics predict Advanced Placement (AP) participation in U.S. public high schools, and which characteristics are most predictive?

Method Overview: We combine school-level enrollment and directory data to study differences in AP participation. Predictors include free or reduced-price lunch eligibility as a socioeconomic proxy, school size, student–teacher ratio, location, and charter, magnet, and Title I status. The project aims to identify patterns associated with low participation and supports further investigation of barriers to advanced coursework. 

## Data and targets

We collect 2021 data through the [Urban Institute Education Data Portal API](https://educationdata.urban.org/):

- **Civil Rights Data Collection (CRDC):** school-level AP enrollment
- **Common Core of Data (CCD) school directory:** school characteristics, location, enrollment, staffing, and lunch eligibility
- **CCD enrollment by grade:** enrollment in grades 9–12, aggregated to construct the participation-rate denominator
  
The cleaned dataset contains **19,416 schools and 20 columns**. The sample includes regular public schools offering grade 12 in the 50 states, Washington, DC, and Puerto Rico; fully virtual schools are excluded.

We define two targets:

1. **Any AP enrollment:** whether a school's AP enrollment count is greater than zero.
2. **AP participation rate:** AP enrollment divided by enrollment in grades 9–12, modeled among schools with positive AP enrollment.

Run `src/01_data_collection.ipynb` to download the raw files, then `src/02_data_cleaning.ipynb` to create `data/ap_clean_2021.csv`. API access requires an internet connection, and a future download may reflect revisions to the source data.

## Analysis workflow and current progress

1. **Data collection — implemented:** `src/01_data_collection.ipynb` downloads the three datasets as raw CSV files.
2. **Cleaning — implemented:** `src/02_data_cleaning.ipynb` aggregates grade enrollment, filters the sample, merges school records, constructs the participation rate, handles selected missing-value codes, and engineers features.
3. **Exploratory analysis — implemented:** `src/03_exploratory_data_analysis.ipynb` examines the target distribution, numeric and categorical relationships, geography, missingness, and correlations. Generated plots are saved in `figures/`.
4. **Modeling and evaluation — planned:** first classify whether a school has any AP enrollment, then predict the participation rate among schools with positive enrollment. We plan to compare logistic regression and ordinary least squares baselines with random forest and gradient boosting models. Hyperparameters will be tuned using cross-validation within the training set, with a held-out test set reserved for final evaluation. Preprocessing will be fitted within training data, and AP enrollment counts and school identifiers will be excluded from predictors.

Modeling notebooks, fitted models, predictions, and evaluation metrics are not yet included in the repository. The environment below covers the current collection, cleaning, and EDA notebooks.

## Python and package versions
| Component | Version |
| --- | --- |
| Python | 3.14.6 |
| pandas | 3.0.3 |
| NumPy | 2.4.6 |
| requests | 2.34.2 |
| Matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| JupyterLab | 4.6.3 |
| ipykernel | 7.3.0 |

## Run locally

Install Conda, clone this repository, and run:

```bash
git clone https://github.com/molinazhang/learning_curve_ds1030.git
cd learning_curve_ds1030
conda env create -f environment.yml
conda activate learning-curve-ds1030
python -m ipykernel install --user --name learning-curve-ds1030 --display-name "Python (learning-curve-ds1030)"
mkdir -p data figures results report
cd src
python -m jupyterlab
```

Select the **Python (learning-curve-ds1030)** kernel in each notebook. Run all cells in this order:

1. `01_data_collection.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_exploratory_data_analysis.ipynb`

Use `src/` as the working directory because the notebooks refer to `../data/` and `../figures/`.

Expected outputs are the three raw CSVs and `ap_clean_2021.csv` in `data/`, and the EDA PNG figures in `figures/`. The notebooks currently do not write model results to `results/`.

## Repository organization

```text
learning_curve_ds1030/
├── data/                 # Raw and cleaned CSVs generated locally; ignored by Git
├── figures/              # Generated EDA figures
├── results/              # Model metrics, predictions, and fitted models when available
├── report/               # Midterm report PDF and final presentation slides when added
├── src/                  # Data collection, cleaning, and analysis notebooks
├── .gitignore
├── environment.yml
├── LICENSE
└── README.md
```

See `results/README.md` for the current status of model outputs. The midterm report PDF and final slides should be placed in `report/` when ready.

## License

Project code is available under the [MIT License](LICENSE).
