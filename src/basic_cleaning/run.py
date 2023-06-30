#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("\n*** Downloading basic_cleaning artifact ***\n")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info("\n*** Dropping duplicates and rows with null values ***\n")
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(inplace=True, drop=True)

    logger.info("\n*** Convert last_review col to datetime ***")
    logger.info("\n*** Drop rows with outlier price values ***\n")
    df["last_review"] = pd.to_datetime(df["last_review"])
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of artifact before any cleaning",
        required=True
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of output artifact data after cleaning",
        required=True
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of output artifact",
        required=True
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of output artifact",
        required=True
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum value of price in order for a record to be kept in dataset",
        required=True
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum value of price in order for a record to be kept in dataset",
        required=True
    )
    args = parser.parse_args()
    go(args)