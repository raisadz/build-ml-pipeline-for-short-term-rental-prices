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

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading raw data")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # set the proper geolocation
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(
        40.5, 41.2
    )
    df = df[idx].copy()

    # save preprocessed data
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Creating clean_sample artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    logger.info("Saving artifact to the project")
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="fully qualified name of the input artifact at w&b",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="name of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum price of the property, anything less is removed from the training",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="maximum price of the property, anything less is removed from the training",
        required=True,
    )

    args = parser.parse_args()

    go(args)
