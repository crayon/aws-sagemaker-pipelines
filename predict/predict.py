import argparse
import boto3
import itertools
import pandas as pd
import sagemaker.session
from sagemaker.predictor import Predictor, CSVSerializer


def main():
    parser = argparse.ArgumentParser("predict")
    parser.add_argument(
        "-e", "--endpoint-name",
        type=str,
        dest="endpoint_name",
        default="crayon-showcase-endpoint",
        help="Name of AWS Sagemaker endpoint used for predition."
    ),
    parser.add_argument(
        "-f", "--file-path",
        type=str,
        dest="file_path",
        default="sample_data.csv",
        help="Path to CSV file with prediction payload data"
    ),
    parser.add_argument(
        "-r", "--region",
        type=str,
        dest="region",
        default="eu-west-1",
        help="AWS region where endpoint is deployed."
    )
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_client = boto_session.client(service_name="sagemaker")
    sagemaker_runtime_client = boto_session.client(
        service_name="sagemaker-runtime")
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=sagemaker_runtime_client,
        default_bucket=None
    )

    predictor = Predictor(
        endpoint_name=args.endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer()
    )

    prediction_data = pd.read_csv(args.file_path, header=1)
    print(predictor.predict(prediction_data.values).decode('utf-8'))


if __name__ == "__main__":
    main()
