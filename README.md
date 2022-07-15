# AWS Sagemaker Pipelines
This repository includes code to implement MLOps principles on AWS Sagemaker.


## Workflow
![Workflow](https://github.com/crayon/aws-sagemaker-pipelines/blob/main/images/workflow.png)

Presented workflow is a basic showcase of AWS Sagemaker capabilities. It includes following steps:
* push your code changes to version control
* trigger GitHub Actions workflow, which performs following:
  * generate ML training pipeline definition and publish it to Sagemaker
  * trigger ML training pipeline and wait for completion
  * deploy newly trained ML model (if performance is acceptable)


## Sagemaker train pipeline
Showcase training pipeline consists of following steps:
* **Prepare data**: fetches publicly available Abalone data and splits it into train, validation and test datasets
* **Train the model**: uses prepared train and validation datasets to train model using XGBoost algorithm
* **Evaluate performance**: uses test dataset to evaluate performance of newly trained model
* **Register model**: in case model performance is up to standards, it registers newly trained model in Model package group

Publishing and triggering AWS Sagemaker pipeline is done using `run_pipeline.py` in `pipelines` folder. Script takes as an input following parameters:
* `--module-name`: name of Python module where pipeline definition is stored (example in repository is `showcase`)
* `--role-arn`: ARN of execution role that will be used to publish and trigger pipeline
* `--description` (optional): description of pipeline
* `--tags` (optional): tags to be used for created pipeline
* `--kwargs`: dictionary of keyword arguments to be used in pipeline definition. Following arguments are supported:
  * `region`: specifies AWS region of Sagemaker instance
  * `role`: ARN of execution role that will be used within each step of training pipeline
  * `default_bucket` (optional): specify S3 bucket where training artifacts are to be stored(defaults to default Sagemaker bucket is used).
  * `repo_data_branch` (optional): specify git repository branch for correct data version (defaults to "main").
  * `repo_data_path` (optional): specify path to data in git repository (defaults to "data/abalone-dataset.csv").
  * `model_name` (optional): specify name of model artifact that is produced (defaults to "crayonShowcase").
  * `model_package_group_name` (optional): specify name of model package group where model is going to be registered (defaults to "crayonShowcasePackageGroup").
  * `pipeline_name` (optional): specify name of pipeline to be published (defaults to "crayonShowcasePipeline").
  * `base_job_prefix` (optional): prefix for each job that will be triggered by pipeline steps (defaults to "crayonShowcase").

Example for manually triggering pipeline publishing and running:
```sh
python -m pip install pipelines/requirements.txt
python pipelines/run_pipeline.py \ 
  -n showcase.pipeline \
  -r ${SAGEMAKER_EXECUTIONROLE_ARN} \
  -k "{\"region\": \"eu-west-1\", \"role\": \"${SAGEMAKER_EXECUTIONROLE_ARN}\", \"pipeline_name\": \"showcase-pipeline\"}"
  -t "[{\"Key\":\"createdBy\", \"Value\":\"manual\"}]"
```


## Model deployment pipeline
Deploying model to real-time inference, CloudFormation is used. Following steps are required:
* generate endpoint configuration file
* package CloudFormation template
* deploy CloudFormation template together with adjusting parameters based on endpoint configuration file

Generation of endpoint configuration uses `build_config_file.py` script in `deploy` folder to adjust `endpoint_config.json` based on user input. Script takes as an input following parameters:
* `--endpoint-name` (optional): name of endpoint that is going to be deployed (defaults to "crayon-showcase-endpoint").
* `--datacapture-s3`: S3 bucket used for data capture
* `--model-execution-role`: ARN of execution role that will be used by deployed model
* `--model-package-group-name`: specify name of model package group where model is registered (defaults to "crayonShowcasePackageGroup").
* `--import-endpoint-config` (optional): file location of endpoint configuration file (defaults to "endpoint_config.json").
* `--export-endpoint-config` (optional): file location of adjusted endpoint configuration file (defaults to "endpoint_config_adj.json").
* `--log-level` (optional): Logging level of script, possible options 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL' (defaults to "INFO").

Once adjusted endpoint configuration file is available, `aws cli` is used to package and deploy CloudFormation template. Packaging CloudFormation template requires S3 bucket to store artifact (you can use any S3 bucket available).

Example for manually deploying model:
```sh
# Generate endpoint configuration file
python -m pip install deploy/requirements.txt
python deploy/build_config_file.py \
  --model-package-group-name crayonShowcasePackageGroup \
  --model-execution-role ${SAGEMAKER_EXECUTIONROLE_ARN} \
  --datacapture-s3 crayon-showcase \
  --import-endpoint-config deploy/endpoint_config.json \
  --export-endpoint-config deploy/endpoint_config_adj.json

# Package CloudFormation template
aws cloudformation package \
  --template deploy/endpoint-cloudformation.yml
  --output-template deploy/cloudformation.yml
  --s3-bucket crayonshowcase

# Overwrite CloudFormation parameters and deploy
aws cloudformation deploy \
  --template-file deploy/cloudformation.yml \
  --stack-name showcase \
  --parameter-overrides $(jq -r '.Parameters | to_entries[] | "\(.key)=\(.value)"' deploy/endpoint_config_adj.json)
```


## GitHub Actions
In addition to manually deploying all components, GitHub Actions workflow definition can be found under `.github/workflows/train_deploy_model.yml`. It incorporates all steps that were described in manual steps above.

Following GitHub repository secrets are required:
* `AWS_ACCESS_KEY_ID`: used to authenticate toward AWS account
* `AWS_SECRET_ACCESS_KEY`: used to authenticate toward AWS account
* `SAGEMAKER_DEFAULT_S3_BUCKET`: S3 bucket used for inference data capture feature
* `SAGEMAKER_EXECUTIONROLE_ARN`: ARN of execution role used across all parts of lifecycle

For additional parameter configuration, following must be adjusted:
* add additional GitHub repository secret
* adjust workflow definition to include those parameters together with appropriate script flags


## Test inference
For testing inference, simple script `predict.py` is available in `predict` folder. It generates pandas dataframe from provided CSV file and triggers prediction. Generated model in showcase pipeline only supports `text/csv` content type for input.

Following arguments are available:
* `endpoint-name` (optional): name of endpoint that is going to be deployed (defaults to "crayon-showcase-endpoint").
* `region` (optional): specifies AWS region of Sagemaker instance (defaults to "eu-west-1").
* `file-path` (optional): file location with prediction data (defaults to "sample_data.csv")

Example for triggering prediction:
```sh
python predict/predict.py \
  --endpoint-name crayon-showcase-endpoint \
  --file-path predict/sample_data.csv
```
