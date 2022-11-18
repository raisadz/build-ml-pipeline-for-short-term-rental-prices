# Build an ML Pipeline for Short-Term Rental Prices in NYC
Project on ML Pipelines of Udacity MLOps Engineer nanodegree. 

You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Submission details

[W&B project link](https://wandb.ai/raisadz/nyc_airbnb/overview?workspace=user-raisadz)

[ML pipeline visualisation](https://wandb.ai/raisadz/nyc_airbnb/artifacts/model_export/random_forest_export/8bc6ea5aded0026ecea7/lineage)


## Hyperparameters optimisation

```bash
mlflow run . -P steps=train_random_forest -P hydra_options="hydra/launcher=joblib modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```



