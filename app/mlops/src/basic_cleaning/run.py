#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Run the data cleaning process
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading data")

    # Load the data
    df_demographics = pd.read_excel(artifact_local_path, sheet_name='Soc_Dem', index_col='Client')
    df_products = pd.read_excel(artifact_local_path, sheet_name='Products_ActBalance', index_col='Client')
    df_transactions = pd.read_excel(artifact_local_path, sheet_name='Inflow_Outflow', index_col='Client')
    df_sales = pd.read_excel(artifact_local_path, sheet_name='Sales_Revenues', index_col='Client')

    # merge the datasets
    df = pd.merge(df_demographics, df_products, left_index=True, right_index=True)
    df = pd.merge(df, df_transactions, left_index=True, right_index=True)
    # get rows that are not in df_sales
    df_woSales = df[~df.index.isin(df_sales.index)]
    # add df_sales column names to df_woSales with 0 values
    df_sales_columns = df_sales.columns
    for col in df_sales_columns:
        df_woSales[col] = -1

    # get rows that are in df_sales
    df = pd.merge(df, df_sales, left_index=True, right_index=True)

    # drop rows in the dataset that are not in the range of expected age and tenure
    logger.info("Drop rows in the dataset that are not in the proper age and tenure.")
    df['Age'] = df['Age'].astype(int)
    df['Tenure'] = df['Tenure'].astype(int)
    idx = df['Age'].between(args.min_age, args.max_age) & df['Tenure'].between(args.min_tenure, args.max_tenure)
    df = df[idx].copy()

    logger.info("Fill missing data")
    # fill missing values in categorical columns with the most frequent value
    df['Sex'] = df['Sex'].fillna(df['Sex'].mode().iloc[0])
    df_woSales['Sex'] = df_woSales['Sex'].fillna(df_woSales['Sex'].mode().iloc[0])
    # fill missing values in numerical columns (those that not categorical) with 0
    df = df.fillna(0)
    df_woSales = df_woSales.fillna(0)

    # Save the cleaned train data to a new artifact  
    filename = "clean_sample.csv"
    df.to_csv(filename, index=True)

    logger.info("Uploading cleaned train data")
    artifact = wandb.Artifact(
        args.output_artifact_train,
        type=args.output_type_train,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)

    # Save the cleaned test data to a new artifact
    filename = "clean_sample_test.csv"
    df_woSales.to_csv(filename, index=True)

    logger.info("Uploading cleaned test data")
    artifact = wandb.Artifact(
        args.output_artifact_test,
        type=args.output_type_test,
        description=args.output_description,
    )
    artifact.add_file("clean_sample_test.csv")
    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_train", 
        type=str,
        help="Name for the output artifact for the training data",
        required=True
    )

    parser.add_argument(
        "--output_type_train", 
        type=str,
        help="Type for the artifact for the training data",
        required=True
    )

    parser.add_argument(
        "--output_artifact_test", 
        type=str,
        help="Name for the output artifact for the test data",
        required=True
    )

    parser.add_argument(
        "--output_type_test", 
        type=str,
        help="Type for the artifact for the test data",
        required=True
    )   

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_age", 
        type=int,
        help="Minimum age to consider",
        required=True
    )

    parser.add_argument(
        "--max_age", 
        type=int,
        help="Maximum age to consider",
        required=True
    )

    parser.add_argument(
        "--min_tenure", 
        type=int,
        help="Minimum tenure to consider",
        required=True
    )

    parser.add_argument(
        "--max_tenure", 
        type=int,
        help="Maximum tenure to consider",
        required=True
    )


    args = parser.parse_args()

    go(args)