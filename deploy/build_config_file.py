import argparse
import json
import boto3
import logging

logger = logging.getLogger(__name__)
sagemaker_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    respose_get_packages = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=50
    )
    approved_packages = respose_get_packages["ModelPackageSummaryList"]
    if len(approved_packages) == 0:
        error_message = f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
        logger.error(error_message)
        raise Exception(error_message)
    
    model_arn = approved_packages[0]["ModelPackageArn"]
    logger.info(f"Latest approved model package: {model_arn}")
    return model_arn

def adjust_config_file(args, model_package_arn, endpoint_config):
    if not "Parameters" in endpoint_config:
        raise Exception("Endpoint configuration file must include parameters")
    if not "Tags" in endpoint_config:
        endpoint_config["Tags"] = {}
    
    additional_params = {
        "ModelPackageName": model_package_arn,
        "ModelExecutionRoleArn": args.model_execution_role,
        "EndpointName": args.endpoint_name,
        "EndpointConfigName": f"{args.endpoint_name}-config"
    }

    return {
        "Parameters": {**endpoint_config["Parameters"], **additional_params}
    }





def main():
    parser = argparse.ArgumentParser("build_config_file")
    parser.add_argument(
        "--endpoint-name",
        type=str,
        dest="endpoint_name",
        default="crayon-showcase-endpoint",
        help="Name of endpoint to be deployed."
    ),
    parser.add_argument(
        "--export-endpoint-config",
        type=str,
        dest="export_endpoint_config",
        default="endpoint_config_adj.json",
        help="Location of adjusted endpoint configuration file."
    ),
    parser.add_argument(
        "--import-endpoint-config",
        type=str,
        dest="import_endpoint_config",
        default="endpoint_config.json",
        help="Location of endpoint configuration file."
    ),
    parser.add_argument(
        "--log-level",
        type=str,
        dest="log_level",
        default="INFO",
        help="Logging level, possible options 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'"
    ),
    parser.add_argument(
        "--model-execution-role",
        type=str,
        required=True,
        dest="model_execution_role",
        help="Role arn for the endpoint service execution role."
    ),
    parser.add_argument(
        "--model-package-group-name",
        type=str,
        required=True,
        dest="model_package_group_name",
        help="Name of model package group name where model is registered."
    )
    args = parser.parse_args()

    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(
        format=log_format,
        level=args.log_level
    )

    model_package_arn = get_approved_package(args.model_package_group_name)

    with open(args.import_endpoint_config, "r") as f:
        endpoint_config = adjust_config_file(
            args=args,
            model_package_arn=model_package_arn,
            endpoint_config=json.load(f)
        )
        endpoint_config_dump = json.dumps(endpoint_config, indent=4)
    logger.info(f"Adjusted endpoint configuration: {endpoint_config_dump}")

    with open(args.export_endpoint_config, "w") as f:
        json.dump(endpoint_config, f, indent=4)


if __name__ == "__main__":
    main()
