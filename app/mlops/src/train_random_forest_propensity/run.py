#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import gc
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, 
                             recall_score, 
                             precision_score, 
                             f1_score)
from sklearn.pipeline import Pipeline, make_pipeline

from feature_selection import tree_shap_collector


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config, allow_val_change=True)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    # Use run.use_artifact(...).file() to get the train and validation artifact (args.trainval_artifact)
    # and save the returned path in train_local_pat
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path, index_col='Client')

    args.ls_output_columns = None if args.ls_output_columns == 'None' else args.ls_output_columns.split(',')

    if args.product not in X.columns:
        raise ValueError(f"Product {args.product} not found in the dataset")
    
    array_stratification = X[args.stratify_by]
    y = X[args.product]  # this removes the column "price" from X and puts it into y
    X = X.drop(columns=args.ls_output_columns)


    logger.info(f"Training {args.product} propensity model")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=array_stratification, random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    preprocess_pipe, full_pipe, processed_features = get_inference_pipeline(rf_config)


    #### Shap values determination ####
    # feature selection using shap
    X_train_preprocessed_ml = preprocess_pipe.fit_transform(X_train[processed_features])
    X_train_preprocessed_ml = pd.DataFrame(X_train_preprocessed_ml, 
                                   columns=processed_features,
                                     index=X_train.index)
    
    X_test_preprocessed_ml = preprocess_pipe.transform(X_val[processed_features])
    X_test_preprocessed_ml = pd.DataFrame(X_test_preprocessed_ml,
                                    columns=processed_features,
                                    index=X_val.index)
    
     # Create random forest for shap values
    random_Forest = RandomForestClassifier()   
    shap_RF = tree_shap_collector(X_train_preprocessed_ml, y_train, X_test_preprocessed_ml, random_Forest, 'rf')

    logger.info("Shap:\n", shap_RF.model_shap_importances)

    #### CV to determine the performance of the model ####
    
    n_subfigures = args.n_folds
    n_cols = ceil(n_subfigures**0.5)
    n_rows = int(ceil(n_subfigures / n_cols))
    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure(figsize=(8, 6))

    # Create a StratifiedKFold object with 5 splits
    skf = StratifiedKFold(n_splits=args.n_folds, random_state=args.random_seed, shuffle=True)
    
    ls_auc = []
    ls_recall = []
    ls_precision = []
    ls_f1 = []
    
    fold = 0
    for train_index, val_index in skf.split(X, array_stratification):

        train_index = X.iloc[train_index].index
        val_index = X.iloc[val_index].index
        X_train_cv, X_val_cv = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
        y_train_cv, y_val_cv = y[y.index.isin(train_index)], y[y.index.isin(val_index)]

        _, full_pipe, processed_features = get_inference_pipeline(rf_config)
        full_pipe.fit(X_train_cv[processed_features], y_train_cv)
        y_pred_cv = full_pipe.predict(X_val_cv[processed_features])
        ls_auc.append(roc_auc_score(y_val_cv, y_pred_cv))
        ls_recall.append(recall_score(y_val_cv, y_pred_cv))
        ls_precision.append(precision_score(y_val_cv, y_pred_cv))
        ls_f1.append(f1_score(y_val_cv, y_pred_cv))

        # plot the confusion matrix 
        ax = fig.add_subplot(gs[fold])
        plot_confusion_matrix(full_pipe, X_val_cv[processed_features], y_val_cv, ax=ax)
        ax.set_title(f"Fold {fold}")
        fold += 1
        
    plt.tight_layout()

    logger.info(f"Mean AUC: {np.mean(ls_auc)}")
    logger.info(f"Mean Recall: {np.mean(ls_recall)}")
    logger.info(f"Mean Precision: {np.mean(ls_precision)}")
    logger.info(f"Mean F1: {np.mean(ls_f1)}")

    run.log(
        {
            "confusion_matrix": wandb.Image(fig),
        }
    )

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    full_pipe.fit(X_train[processed_features], y_train)

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"
    # Infer the signature of the model
    for col in X_val.columns:
        if X_val[col].dtype == 'object':
            X_val[col] = X_val[col].astype('string')
    signature = mlflow.models.infer_signature(X_val[processed_features], y_val)
    export_path = f"random_forest_dir/{args.product}/{args.output_artifact}"

    mlflow.sklearn.save_model(
            full_pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )

    ######################################
    # Upload the model we just exported to W&B
    # use wandb.Artifact to create an artifact. Use args.output_artifact as artifact name, "model_export" as
    # type, provide a description and add rf_config as metadata. Then, use the .add_dir method of the artifact instance
    # you just created to add the "random_forest_dir" directory to the artifact, and finally use
    # run.log_artifact to log the artifact to the run
    ######################################
    artifact = wandb.Artifact(
    args.output_artifact + f"_{args.product}",
    type=f"model_export_{args.product}",
    description="Random Forest pipeline export",
    )
    artifact.add_dir(export_path)

    run.log_artifact(artifact)

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(shap_RF.model_shap_importances)

    ######################################
    # Now log the performance metrics to W&B
    ######################################
    run.summary['Product']  = args.product
    run.summary['cv_mean_auc'] = np.mean(ls_auc)
    run.summary['cv_mean_recall'] = np.mean(ls_recall)
    run.summary['cv_mean_precision'] = np.mean(ls_precision)
    run.summary['cv_mean_f1'] = np.mean(ls_f1)


    # Upload to W&B the feature importance visualization
    run.log(
        {
        "feature_importance": wandb.Image(fig_feat_imp),
        }
    )

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
    return ax


def plot_feature_importance(df_sorted_feature_importance):
    """
    Plot the feature importance
    """
    df_sorted_feature_importance.set_index('feature_name', inplace=True)
    fig_feat_imp, ax = plt.subplots()
    df_sorted_feature_importance.plot(kind='barh', ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    return fig_feat_imp


def get_inference_pipeline(rf_config):

    non_ordinal_categorical = ["Sex"]
    ######################################
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"), 
        OneHotEncoder(drop="first")
    )
    ######################################

    # Let's impute the numerical columns to make sure we can handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    # zero_imputed = [
    #    'Count_CA', 'Count_SA', 'Count_OVD', 'Count_CC', 'Count_CL', 'Count_MF', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF',
    #    'ActBal_OVD', 'ActBal_CC', 'ActBal_CL', 'VolumeCred', 'VolumeCred_CA',
    #    'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA',
    #    'VolumeDebCash_Card', 'VolumeDebCashless_Card',
    #    'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA',
    #    'TransactionsDebCash_Card', 'TransactionsDebCashless_Card',
    #    'TransactionsDeb_PaymentOrder'
    # ]

    zero_imputed = ['ActBal_CA', 'VolumeCred_CA', 
                    'VolumeDeb']

    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # Let's impute the 2 numerical columns , Age and Tenure, with the median
    median_imputed = ["Age", "Tenure"]

    median_imputer = SimpleImputer(strategy="median")

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            # ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("impute_median", median_imputer, median_imputed)
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + median_imputed
    processed_features = non_ordinal_categorical + zero_imputed + median_imputed
    processed_features = zero_imputed + median_imputed

    # Create random forest
    random_Forest = RandomForestClassifier(**rf_config)

    ######################################
    # Create the inference pipeline. The pipeline must have 2 steps: a step called "preprocessor" applying the
    # ColumnTransformer instance that we saved in the `preprocessor` variable, and a step called "random_forest"
    # with the random forest instance that we just saved in the `random_forest` variable.
    # Use the explicit Pipeline constructor so you can assign the names to the steps, do not use make_pipeline

    preprocess_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )

    full_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_Forest),
        ]
    )

    return preprocess_pipe, full_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--ls_output_columns",
        type=str,
        help="List of columns to consider as output columns and to be excluded from the features",
        required=True,
    )

    parser.add_argument(
        "--product",
        type=str,
        help="Product to train model",
        required=True,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        help="Number of folds for cross-validation",
        default=5,
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
