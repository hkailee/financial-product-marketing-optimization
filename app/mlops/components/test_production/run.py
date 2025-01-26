#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import os
import pandas as pd
import numpy as np

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_production")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path_prop_cc = run.use_artifact(args.model_propensity_cc).download()
    model_local_path_prop_cl = run.use_artifact(args.model_propensity_cl).download()
    model_local_path_prop_mf = run.use_artifact(args.model_propensity_mf).download()
    model_local_path_revenue_cc = run.use_artifact(args.model_revenue_cc).download()
    model_local_path_revenue_cl = run.use_artifact(args.model_revenue_cl).download()
    model_local_path_revenue_mf = run.use_artifact(args.model_revenue_mf).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path, index_col="Client")
    X_test = X_test.drop(columns=["Sale_CC", "Sale_CL", "Sale_MF", "Revenue_CC", "Revenue_CL", "Revenue_MF"])

    logger.info("Loading model and performing inference on test set")
    sk_pipe_prop_cc = mlflow.sklearn.load_model(model_local_path_prop_cc)
    sk_pipe_prop_cl = mlflow.sklearn.load_model(model_local_path_prop_cl)
    sk_pipe_prop_mf = mlflow.sklearn.load_model(model_local_path_prop_mf)
    sk_pipe_revenue_cc = mlflow.sklearn.load_model(model_local_path_revenue_cc)
    sk_pipe_revenue_cl = mlflow.sklearn.load_model(model_local_path_revenue_cl)
    sk_pipe_revenue_mf = mlflow.sklearn.load_model(model_local_path_revenue_mf)

    y_pred_sale_cc = sk_pipe_prop_cc.predict(X_test)
    y_pred_sale_cl = sk_pipe_prop_cl.predict(X_test)
    y_pred_sale_mf = sk_pipe_prop_mf.predict(X_test)

    y_pred_prop_cc = sk_pipe_prop_cc.predict_proba(X_test)[:, 1]
    y_pred_prop_cl = sk_pipe_prop_cl.predict_proba(X_test)[:, 1]
    y_pred_prop_mf = sk_pipe_prop_mf.predict_proba(X_test)[:, 1]

    y_pred_revenue_cc_log = sk_pipe_revenue_cc.predict(X_test)
    y_pred_revenue_cl_log = sk_pipe_revenue_cl.predict(X_test)
    y_pred_revenue_mf_log = sk_pipe_revenue_mf.predict(X_test)
    
    y_pred_revenue_cc = np.expm1(y_pred_revenue_cc_log)
    y_pred_revenue_cl = np.expm1(y_pred_revenue_cl_log)
    y_pred_revenue_mf = np.expm1(y_pred_revenue_mf_log)


    # save result in dataframe and log as artifact
    df_result = pd.DataFrame({
        "y_pred_sale_cc": y_pred_sale_cc,
        f"y_pred_prop_cc_{sk_pipe_prop_cc.classes_[1]}": y_pred_prop_cc,
        "y_pred_sale_cl": y_pred_sale_cl,
        f"y_pred_prop_cl_{sk_pipe_prop_cl.classes_[1]}": y_pred_prop_cl,
        "y_pred_sale_mf": y_pred_sale_mf,
        f"y_pred_prop_mf_{sk_pipe_prop_mf.classes_[1]}": y_pred_prop_mf,
        "y_pred_revenue_cc": y_pred_revenue_cc,
        "y_pred_revenue_cl": y_pred_revenue_cl,
        "y_pred_revenue_mf": y_pred_revenue_mf,
    }, index=X_test.index)

    # calculate expected revenue
    df_result["expected_revenue_cc"] = df_result[f"y_pred_prop_cc_{sk_pipe_prop_cc.classes_[1]}"] * df_result["y_pred_revenue_cc"] * df_result["y_pred_sale_cc"]
    df_result["expected_revenue_cl"] = df_result[f"y_pred_prop_cl_{sk_pipe_prop_cl.classes_[1]}"] * df_result["y_pred_revenue_cl"] * df_result["y_pred_sale_cl"]
    df_result["expected_revenue_mf"] = df_result[f"y_pred_prop_mf_{sk_pipe_prop_mf.classes_[1]}"] * df_result["y_pred_revenue_mf"] * df_result["y_pred_sale_mf"]

    # save the result to a new artifact
    filename = "clean_sample_test_result.csv"
    df_result.to_csv(filename, index=True)

    logger.info("Uploading test result")
    artifact = wandb.Artifact(
        "clean_sample_test_result",
        type="clean_sample_test_result",
        description="Clean Sample Test result",
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    logger.info("Cleaning up")
    os.remove(filename)

    # select the top 150 clients to offer
    df_to_offer = df_result[["expected_revenue_cc", "expected_revenue_cl", "expected_revenue_mf"]]
    df_to_offer["product_to_offer"] = df_to_offer.idxmax(axis=1).str.replace("expected_revenue_", "")
    df_to_offer["expected_revenue"] = df_to_offer.max(axis=1)
    df_to_offer = df_to_offer.sort_values(by="expected_revenue", ascending=False)
    df_to_offer = df_to_offer.head(150)

    # reverse the log transformation for expected revenue
    df_to_offer["expected_revenue_reverseLog"] = np.expm1(df_to_offer["expected_revenue"])

    # save the result to a new artifact
    filename = "clients_to_offer.csv"
    df_to_offer.to_csv(filename, index=True)

    logger.info("Uploading clients to offer")
    artifact = wandb.Artifact(
        "clients_to_offer",
        type="clients_to_offer",
        description="Clients to offer",
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    logger.info("Cleaning up")
    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--model_propensity_cc",
        type=str, 
        help="Input MLFlow model for cc propensity model",
        required=True
    )

    parser.add_argument(
        "--model_propensity_cl",
        type=str, 
        help="Input MLFlow model for cl propensity model",
        required=True
    )

    parser.add_argument(
        "--model_propensity_mf",
        type=str, 
        help="Input MLFlow model for mf propensity model",
        required=True
    )

    parser.add_argument(
        "--model_revenue_cc",
        type=str, 
        help="Input MLFlow model for cc revenue model",
        required=True
    )

    parser.add_argument(
        "--model_revenue_cl",
        type=str, 
        help="Input MLFlow model for cl revenue model",
        required=True
    )

    parser.add_argument(
        "--model_revenue_mf",
        type=str, 
        help="Input MLFlow model for mf revenue model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
