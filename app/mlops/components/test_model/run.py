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
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score,
    roc_auc_score, 
    recall_score, 
    precision_score, 
    f1_score
    )
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
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
    y_test_sale_cc = X_test.pop("Sale_CC")
    y_test_sale_cl = X_test.pop("Sale_CL")
    y_test_sale_mf = X_test.pop("Sale_MF")
    y_test_revenue_cc = X_test.pop("Revenue_CC")
    y_test_revenue_cl = X_test.pop("Revenue_CL")
    y_test_revenue_mf = X_test.pop("Revenue_MF")
    y_test_revenue_cc_log = np.log1p(y_test_revenue_cc)
    y_test_revenue_cl_log = np.log1p(y_test_revenue_cl)
    y_test_revenue_mf_log = np.log1p(y_test_revenue_mf)

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


    logger.info("Scoring")

    roc_sale_cc = roc_auc_score(y_test_sale_cc, y_pred_sale_cc)
    roc_sale_cl = roc_auc_score(y_test_sale_cl, y_pred_sale_cl)
    roc_sale_mf = roc_auc_score(y_test_sale_mf, y_pred_sale_mf)

    recall_sale_cc = recall_score(y_test_sale_cc, y_pred_sale_cc)
    recall_sale_cl = recall_score(y_test_sale_cl, y_pred_sale_cl)
    recall_sale_mf = recall_score(y_test_sale_mf, y_pred_sale_mf)

    precision_sale_cc = precision_score(y_test_sale_cc, y_pred_sale_cc)
    precision_sale_cl = precision_score(y_test_sale_cl, y_pred_sale_cl)
    precision_sale_mf = precision_score(y_test_sale_mf, y_pred_sale_mf)

    f1_sale_cc = f1_score(y_test_sale_cc, y_pred_sale_cc)
    f1_sale_cl = f1_score(y_test_sale_cl, y_pred_sale_cl)
    f1_sale_mf = f1_score(y_test_sale_mf, y_pred_sale_mf)
    
    r_squared_revenue_cc = r2_score(y_test_revenue_cc_log, y_pred_revenue_cc_log)
    r_squared_revenue_cl = r2_score(y_test_revenue_cl_log, y_pred_revenue_cl_log)
    r_squared_revenue_mf = r2_score(y_test_revenue_mf_log, y_pred_revenue_mf_log)

    mae_revenue_cc = mean_absolute_error(y_test_revenue_cc_log, y_pred_revenue_cc_log)
    mae_revenue_cl = mean_absolute_error(y_test_revenue_cl_log, y_pred_revenue_cl_log)
    mae_revenue_mf = mean_absolute_error(y_test_revenue_mf_log, y_pred_revenue_mf_log)


    logger.info(f"ROC Sale_CC: {roc_sale_cc}")
    logger.info(f"ROC Sale_CL: {roc_sale_cl}")
    logger.info(f"ROC Sale_MF: {roc_sale_mf}")

    logger.info(f"Recall Sale_CC: {recall_sale_cc}")
    logger.info(f"Recall Sale_CL: {recall_sale_cl}")
    logger.info(f"Recall Sale_MF: {recall_sale_mf}")

    logger.info(f"Precision Sale_CC: {precision_sale_cc}")
    logger.info(f"Precision Sale_CL: {precision_sale_cl}")
    logger.info(f"Precision Sale_MF: {precision_sale_mf}")

    logger.info(f"F1 Sale_CC: {f1_sale_cc}")
    logger.info(f"F1 Sale_CL: {f1_sale_cl}")
    logger.info(f"F1 Sale_MF: {f1_sale_mf}")

    logger.info(f"R2 Revenue_CC: {r_squared_revenue_cc}")
    logger.info(f"R2 Revenue_CL: {r_squared_revenue_cl}")
    logger.info(f"R2 Revenue_MF: {r_squared_revenue_mf}")

    logger.info(f"MAE Revenue_CC: {mae_revenue_cc}")
    logger.info(f"MAE Revenue_CL: {mae_revenue_cl}")
    logger.info(f"MAE Revenue_MF: {mae_revenue_mf}")

    # Log metrics
    run.summary['roc_sale_cc'] = roc_sale_cc
    run.summary['roc_sale_cl'] = roc_sale_cl
    run.summary['roc_sale_mf'] = roc_sale_mf

    run.summary['recall_sale_cc'] = recall_sale_cc
    run.summary['recall_sale_cl'] = recall_sale_cl
    run.summary['recall_sale_mf'] = recall_sale_mf

    run.summary['precision_sale_cc'] = precision_sale_cc
    run.summary['precision_sale_cl'] = precision_sale_cl
    run.summary['precision_sale_mf'] = precision_sale_mf

    run.summary['f1_sale_cc'] = f1_sale_cc
    run.summary['f1_sale_cl'] = f1_sale_cl
    run.summary['f1_sale_mf'] = f1_sale_mf

    run.summary['r2_revenue_cc'] = r_squared_revenue_cc
    run.summary['r2_revenue_cl'] = r_squared_revenue_cl
    run.summary['r2_revenue_mf'] = r_squared_revenue_mf

    run.summary['mae_revenue_cc'] = mae_revenue_cc
    run.summary['mae_revenue_cl'] = mae_revenue_cl
    run.summary['mae_revenue_mf'] = mae_revenue_mf

    # save result in dataframe and log as artifact
    df_result = pd.DataFrame({
        "y_test_sale_cc": y_test_sale_cc,
        "y_pred_sale_cc": y_pred_sale_cc,
        f"y_pred_prop_cc_{sk_pipe_prop_cc.classes_[1]}": y_pred_prop_cc,
        "y_test_sale_cl": y_test_sale_cl,
        "y_pred_sale_cl": y_pred_sale_cl,
        f"y_pred_prop_cl_{sk_pipe_prop_cl.classes_[1]}": y_pred_prop_cl,
        "y_test_sale_mf": y_test_sale_mf,
        "y_pred_sale_mf": y_pred_sale_mf,
        f"y_pred_prop_mf_{sk_pipe_prop_mf.classes_[1]}": y_pred_prop_mf,
        "y_test_revenue_cc": y_test_revenue_cc,
        "y_pred_revenue_cc": y_pred_revenue_cc,
        "y_test_revenue_cl": y_test_revenue_cl,
        "y_pred_revenue_cl": y_pred_revenue_cl,
        "y_test_revenue_mf": y_test_revenue_mf,
        "y_pred_revenue_mf": y_pred_revenue_mf,
    }, index=X_test.index)

    # calculate expected revenue
    df_result["expected_revenue_cc"] = df_result[f"y_pred_prop_cc_{sk_pipe_prop_cc.classes_[1]}"] * df_result["y_pred_revenue_cc"]
    df_result["expected_revenue_cl"] = df_result[f"y_pred_prop_cl_{sk_pipe_prop_cl.classes_[1]}"] * df_result["y_pred_revenue_cl"]
    df_result["expected_revenue_mf"] = df_result[f"y_pred_prop_mf_{sk_pipe_prop_mf.classes_[1]}"] * df_result["y_pred_revenue_mf"]

    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(gs[0])
    plot_confusion_matrix(sk_pipe_prop_cc, X_test, y_test_sale_cc, ax)
    ax.set_title("Confusion Matrix Sale_CC")
    ax = fig.add_subplot(gs[1])
    plot_confusion_matrix(sk_pipe_prop_cl, X_test, y_test_sale_cl, ax)
    ax.set_title("Confusion Matrix Sale_CL")
    ax = fig.add_subplot(gs[2])
    plot_confusion_matrix(sk_pipe_prop_mf, X_test, y_test_sale_mf, ax)
    ax.set_title("Confusion Matrix Sale_MF")
    ax = fig.add_subplot(gs[3])
    plot_predicted_vs_actual(sk_pipe_revenue_cc, X_test, y_test_revenue_cc_log, ax)
    ax.set_title("Predicted vs Actual Revenue_CC")
    ax = fig.add_subplot(gs[4])
    plot_predicted_vs_actual(sk_pipe_revenue_cl, X_test, y_test_revenue_cl_log, ax)
    ax.set_title("Predicted vs Actual Revenue_CL")
    ax = fig.add_subplot(gs[5])
    plot_predicted_vs_actual(sk_pipe_revenue_mf, X_test, y_test_revenue_mf_log, ax)
    ax.set_title("Predicted vs Actual Revenue_MF")
    plt.tight_layout()

    run.log({"test_result_plots": wandb.Image(fig)})

    # save the result to a new artifact
    filename = "test_result.csv"
    df_result.to_csv(filename, index=True)

    logger.info("Uploading test result")
    artifact = wandb.Artifact(
        "test_result",
        type="test_result",
        description="Test result",
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    logger.info("Cleaning up")
    os.remove(filename)


def plot_confusion_matrix(pipe, X, y, ax):
    """
    Plot the confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    y_pred = pipe.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    for labels in disp.text_.ravel():
        labels.set_fontsize(16)
    ax.title.set_fontsize(6)
    return ax


def plot_predicted_vs_actual(pipe, X_val, y_val, ax):
    """
    Plot the predicted vs actual values
    """
    y_pred = pipe.predict(X_val)
    ax.scatter(y_val, y_pred)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual')
    ax.title.set_fontsize(6) 
    return ax

    # include the R2 and MAE scores in the plot
    ax.text(0.05, 0.95, f"R2: {r2_score(y_val, y_pred):.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"MAE: {mean_absolute_error(y_val, y_pred):.2f}", transform=ax.transAxes)

    return ax


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
