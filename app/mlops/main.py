import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest_propensity",
    "train_random_forest_revenue",
    "train_lasso_revenue",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
   "test_model",
   "test_production"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.xlsx",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.xlsx:latest",
                    "output_artifact_train": "clean_sample.csv",
                    "output_type_train": "clean_sample",
                    "output_artifact_test": "clean_sample_test.csv",
                    "output_type_test": "clean_sample_test",
                    "output_description": "Data with outliers and null values removed",
                    "min_age": config['etl']['min_age'],
                    "max_age": config['etl']['max_age'],
                    "min_tenure": config['etl']['min_tenure'],
                    "max_tenure": config['etl']['max_tenure']
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": f"{config['data_check']['csv_to_check']}:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_age": config['etl']['min_age'],
                    "max_age": config['etl']['max_age'],
                    "min_tenure": config['etl']['min_tenure'],
                    "max_tenure": config['etl']['max_tenure']
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                },
            )

        if "train_random_forest_propensity" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest_classifier_propensity"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we created as the rf_config parameter for the train_random_forest
            # step
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest_propensity"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "ls_output_columns": ','.join(config['modeling']['ls_output_columns']),
                    "product": config['modeling']['product_to_train'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "n_folds": config['modeling']['n_folds'],
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export",
                },
            )
        
        if "train_random_forest_revenue" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config_revenue.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest_regression_revenue"].items()), fp)
            
            # NOTE: use the rf_config we created as the rf_config parameter for the train_random_forest
            # step
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest_revenue"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "ls_output_columns": ','.join(config['modeling']['ls_output_columns']),
                    "product": config['modeling']['product_to_train'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "n_folds": config['modeling']['n_folds'],
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export",
                },
            )

        if "train_lasso_revenue" in active_steps:

            # NOTE: use the lasso_config we created as the lasso_config parameter for the train_lasso
            lasso_config = os.path.abspath("lasso_config.json")
            with open(lasso_config, "w+") as fp:
                json.dump(dict(config["modeling"]["lasso_regression_revenue"].items()), fp)

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_lasso_revenue"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "ls_output_columns": ','.join(config['modeling']['ls_output_columns']),
                    "product": config['modeling']['product_to_train'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "n_folds": config['modeling']['n_folds'],
                    "lasso_config": lasso_config,
                    "output_artifact": "lasso_export",
                },
            )

        if "test_model" in active_steps:

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_model"),
                "main",
                parameters={
                    "model_propensity_cc": config['best_model_propensity']['propensity_cc'],
                    "model_propensity_cl": config['best_model_propensity']['propensity_cl'],
                    "model_propensity_mf": config['best_model_propensity']['propensity_mf'],
                    "model_revenue_cc": config['best_model_revenue']['revenue_cc'],
                    "model_revenue_cl": config['best_model_revenue']['revenue_cl'],
                    "model_revenue_mf": config['best_model_revenue']['revenue_mf'],
                    "test_dataset": "test_data.csv:latest",
                },
            )
        
        if "test_production" in active_steps:
            
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_production"),
                "main",
                parameters={
                    "model_propensity_cc": config['best_model_propensity']['propensity_cc'],
                    "model_propensity_cl": config['best_model_propensity']['propensity_cl'],
                    "model_propensity_mf": config['best_model_propensity']['propensity_mf'],
                    "model_revenue_cc": config['best_model_revenue']['revenue_cc'],
                    "model_revenue_cl": config['best_model_revenue']['revenue_cl'],
                    "model_revenue_mf": config['best_model_revenue']['revenue_mf'],
                    "test_dataset": f"{config['production']['test_csv']}:latest",
                },
            )

if __name__ == "__main__":
    go()
