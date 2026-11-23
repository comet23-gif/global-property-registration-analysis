# Global Property Registration Time Analysis

An exploratory data analysis and predictive modeling project examining how economic development and administrative procedures relate to the time required to register property across countries from 2004 to 2014.

## Highlights

- Analyzed 2,176 complete country-year observations using GDP per capita, business startup time, and the number of property registration procedures.
- Business startup time had the strongest individual Pearson correlation with property registration time among the three predictors (r = 0.246).
- An ordinary least squares model explained 9.46% of the variation (R² = 0.0946). A Breusch–Pagan test indicated heteroskedasticity, so the analysis also reports HC3 robust standard errors.
- Of the three ensemble models evaluated on a held-out test set, the random forest had the lowest reported RMSE, 55.64 days. Its feature importance ranked business startup time highest.

These are associations, not evidence of causation. The low linear-model R² and the random country-year split limit interpretation: observations from the same country can appear in both training and test sets, so performance may not generalize to unseen countries or future years.

## Explore the analysis

Open [the notebook](notebooks/property_analysis.ipynb) on GitHub to view the saved analysis, charts, model diagnostics, and conclusions. The workflow covers data filtering and reshaping, exploratory visualization, correlation, OLS regression, variance inflation factors, heteroskedasticity testing, ensemble model comparison, feature importance, and partial dependence.

## Data and reproducibility

The notebook uses a file named `worlddata.csv`, with `CountryName`, `CountryCode`, `Year`, `IndicatorName`, and `Value` columns. The source file is not included in this repository, and its original download location and license are not documented in the notebook. The notebook reads it from a personal Google Drive path and writes a cleaned file to Drive; those paths must be changed to run it elsewhere. The saved outputs can still be inspected without the dataset.

The notebook was written for Google Colab. It imports pandas, NumPy, matplotlib, seaborn, scikit-learn, statsmodels, dmba, and SHAP; it also installs LIME. To reproduce the analysis, supply the original dataset, update the Drive-specific paths, and install these dependencies in a compatible Python environment. Package versions were not recorded, so results may vary across environments.


## Project structure

- `notebooks/property_analysis.ipynb` — full analysis and saved outputs.
- `src/modeling.py` — script for preparing the data and comparing the three ensemble models.
- `results/model_comparison.png` — test-set RMSE values from the saved notebook output.
- `requirements.txt` — Python dependencies (versions are not pinned).

To run the model comparison after providing the dataset:

```bash
pip install -r requirements.txt
python src/modeling.py path/to/worlddata.csv
```
