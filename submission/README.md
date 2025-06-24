## Environment instructions

The commands below create an Anaconda environment sufficient to run `submission_1.ipynb`

```
conda create -n ticks_env python=3.10 -y
conda activate ticks_env
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn tqdm scipy
pip install cmdstanpy pyyaml gpflow tensorflow shap xarray cartopy geopandas 
python -m cmdstanpy.install_cmdstan
```

## How to process the data:
We obtained and processed the:
- Raw Earth observation data (climate, landcover, elevation)
- Tick occurrence data
- Host occurrence data
- Age demographics data

Run the notebook `https://github.com/harrisonzhu508/lyme_disease_working/blob/48127cf75a03953613e9e64946c1a1cadd9ee59e/submission/download_process_covariates.ipynb`

## How to execute the main model

```
cd submission; ./scripts/run_models.sh
```

## Obtain plots and results
Run the notebook `https://github.com/harrisonzhu508/lyme_disease_working/blob/a2f2f0ca5d77ac0fb773541edd8634e6560fc821/submission/plots.ipynb`