import boto3
import os
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel

base_dir = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker")
    
    return sagemaker_client

def get_sagemaker_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker")
    sagemaker_runtime_client = boto_session.client(service_name="sagemaker-runtime")

    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=sagemaker_runtime_client,
        default_bucket=default_bucket
    )


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    repo_data_branch="main",
    repo_data_path="data/abalone-dataset.csv",
    model_name="crayonShowcase",
    model_package_group_name="crayonShowcasePackageGroup",
    pipeline_name="crayonShowcasePipeline",
    base_job_prefix="crayonShowcase"
):
    # Prepare session info
    sagemaker_session = get_sagemaker_session(
        region=region,
        default_bucket=default_bucket
    )
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session=sagemaker_session)
    

    # Pipeline parameters
    eval_instance_count = ParameterInteger(name="evalInstanceCount", default_value=1)
    eval_instance_type = ParameterString(name="evalInstanceType", default_value="ml.m5.large")
    prep_data_input_data = ParameterString(name="dataPrepInputData", default_value=repo_data_path)
    prep_data_input_repo_branch = ParameterString(name="dataPrepInputRepoBranch", default_value=repo_data_branch)
    prep_data_instance_count = ParameterInteger(name="dataPrepInstanceCount", default_value=1)
    prep_data_instance_type = ParameterString(name="dataPrepInstanceType", default_value="ml.m5.xlarge")
    register_inference_instance_type = ParameterString(name="registerInferenceInstanceType", default_value="ml.m5.large")
    register_transform_instance_type = ParameterString(name="registerTransformInstanceType", default_value="ml.m5.large")
    train_instance_count = ParameterInteger(name="trainInstanceCount", default_value=1)
    train_instance_type = ParameterString(name="trainInstanceType", default_value="ml.m5.xlarge")
    train_output_path= ParameterString(name="trainOutputPath", default_value=f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/Model")


    # Data processing step
    prep_data_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=prep_data_instance_type,
        instance_count=prep_data_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-prep-data",
        sagemaker_session=sagemaker_session,
    )

    step_prepare = ProcessingStep(
        name="dataPreparation",
        display_name="Data preparation",
        description="Split data to train, test and validation datasets.",
        code=os.path.join(base_dir, "preprocess.py"),
        job_arguments=[
            "--input-data", prep_data_input_data,
            "--repo-branch", prep_data_input_repo_branch
        ],
        processor=prep_data_processor,
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train"
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test"
            )
        ]
    )

    
    # Training step
    xgb_image_url = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=train_instance_type
    )

    xgb_estimator = Estimator(
        image_uri=xgb_image_url,
        role=role,
        instance_count=train_instance_count,
        instance_type=train_instance_type,
        output_path=train_output_path,
        base_job_name=f"{base_job_prefix}/xgb-train",
        sagemaker_session=sagemaker_session
    )
    xgb_estimator.set_hyperparameters(
        objective="reg:linear",
        num_round=20,
        max_depth=3,
        eta=0.3,
        gamma=3,
        min_child_weight=5,
        subsample=0.8,
        silent=0
    )

    step_train = TrainingStep(
        name="trainModel",
        display_name="Train new model",
        description="Train model and store it on S3",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_prepare.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_prepare.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )


    # Evaluation step
    eval_processor = ScriptProcessor(
        image_uri=xgb_image_url,
        role=role,
        command=["python3"],
        instance_count=eval_instance_count,
        instance_type=eval_instance_type,
        base_job_name=f"{base_job_prefix}/evaluation",
        sagemaker_session=sagemaker_session
    )

    eval_prop_file = PropertyFile(
        name="evaluationReport",
        output_name="eval_report",
        path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name="evaluateModel",
        display_name="Evaluate new model",
        description="Evaluate performance of newly trained model",
        code=os.path.join(base_dir, "evaluate.py"),
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_prepare.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="eval_report",
                source="/opt/ml/processing/evaluation"
            )
        ],
        property_files=[eval_prop_file]
    )


    # Register new model step
    model = Model(
        name=model_name,
        image_uri=xgb_image_url,
        role=role,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/", values=[step_eval.properties.ProcessingOutputConfig.Outputs["eval_report"].S3Output.S3Uri, "evaluation.json"]),
            content_type="application/json"
        )
    )

    step_register = RegisterModel(
        name="registerModel",
        display_name="Register new model",
        description="Register newly trained model in model registry",
        estimator=xgb_estimator,
        model=model,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[register_inference_instance_type],
        transform_instances=[register_transform_instance_type],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status="Approved"
    )


    # Check evaluation step
    condition_eval = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=eval_prop_file,
            json_path="regression_metrics.mse.value"
        ),
        right=7.0
    )

    step_fail = FailStep(
      name="failModelBadMetric",
      display_name="Bad model metrics",
      error_message="Fail pipeline due to model metrics below requirements."
    )

    step_condition = ConditionStep(
        name="reviewModelMetrics",
        display_name="Review model metrics",
        description="Review newly trained model metrics and continue based on results",
        conditions=[condition_eval],
        if_steps=[step_register],
        else_steps=[step_fail]
    )


    # Create pipeline definition
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_prepare, step_train, step_eval, step_condition],
        parameters=[
            eval_instance_count,
            eval_instance_type,
            prep_data_input_data,
            prep_data_input_repo_branch,
            prep_data_instance_count,
            prep_data_instance_type,
            register_inference_instance_type,
            register_transform_instance_type,
            train_instance_count,
            train_instance_type,
            train_output_path
        ],
        sagemaker_session=sagemaker_session
    )

    return pipeline
