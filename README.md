# Build an ML Pipeline for Short-Term Rental Prices in NYC
Project *Build an ML Pipeline for Short-Term Rental Prices in NYC* of Machine Learning DevOps Engineer Nanodegree Udacity.

## Project Description
The goal of the project is to build a reproducible ML Pipeline for estimating a property rental price using [MLflow](https://mlflow.org) and [Weights & Biases](https://wandb.ai). A new data comes every week, and the model needs a regular re-training. An end-to-end reusable pipeline will enable an easy re-training process and reduce the time-to-production.

## Files and data description
Folder `src` contains the steps of the implemented ML pipeline:

`src/eda` Exploratory data analysis;

`src/basic_cleaning` Data cleaning step, which starts from the sample.csv artifact and create a new artifact clean_sample.csv with the cleaned data;

`src/data_check` Data testing;

`src/train_random_forest` Model training;

`main.py` Main script to run all the pipeline steps;

`config.yaml` Default hyperparameters used by Hydra.

## Installation
Clone the repo:

```bash
git clone git@github.com:raisadz/build-ml-pipeline-for-short-term-rental-prices.git
cd build-ml-pipeline-for-short-term-rental-prices
```

Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

## Running the project
To run the project, you need to signup/login to wandb, follow the [instructions](https://docs.wandb.ai/quickstart).

To run the whole project:
```bash
mlflow run .
```

You can run one step of the pipeline. For example, to download the data:
```bash
mlflow run . -P steps=download
```
Or to run the EDA step:
```bash
mlflow run src/eda
```
Hydra allows an easy hyperparameters tuning (option `hydra/lancher=joblib` enables parallel calculation):
```bash
mlflow run . -P steps=train_random_forest -P hydra_options="hydra/launcher=joblib modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

To test the model against the test data set:
```bash
mlflow run . -P steps=test_regression_model
```
## Weights & Biases project

[Weights&Biases project link](https://wandb.ai/raisadz/nyc_airbnb/overview?workspace=user-raisadz)

[ML pipeline visualisation](https://wandb.ai/raisadz/nyc_airbnb/artifacts/model_export/random_forest_export/8bc6ea5aded0026ecea7/lineage)

 



