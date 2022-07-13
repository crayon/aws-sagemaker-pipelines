import argparse
import ast
import json
import sys

def main():
    parser = argparse.ArgumentParser("run_pipeline")
    parser.add_argument(
        "-n", "--module-name",
        type=str,
        dest="module_name",
        help="Module name of the pipeline to import."
    )
    parser.add_argument(
        "-k", "--kwargs",
        dest="kwargs",
        help="Dictionary of keyword arguments for the pipeline."
    )
    parser.add_argument(
        "-r", "--role-arn",
        type=str,
        dest="role_arn",
        help="Role arn for the pipeline service execution role."
    )
    parser.add_argument(
        "-d", "--description",
        type=str,
        dest="description",
        default=None,
        help="Description of ML pipeline."
    )
    parser.add_argument(
        "-t", "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'"""
    )
    args = parser.parse_args()

    if (args.module_name is None) or (args.role_arn is None):
        parser.print_help()
        sys.exit(2)
    
    if args.tags:
        pipeline_tags = ast.literal_eval(args.tags)
    else:
        pipeline_tags = []
    
    module_import = __import__(args.module_name, fromlist=["get_pipeline"])
    kwargs = ast.literal_eval(args.kwargs)

    pipeline = module_import.get_pipeline(**kwargs)
    pipeline_json = json.loads(pipeline.definition())
    print("\n###### Pipeline definition being used:")
    print(json.dumps(obj=pipeline_json, indent=2, sort_keys=True))

    pipeline_upsert = pipeline.upsert(
        role_arn=args.role_arn,
        description=args.description,
        tags=pipeline_tags
    )

    print("\n###### Created pipeline, following response received:")
    print(pipeline_upsert)

    pipeline_execution = pipeline.start()
    print(f"\n###### Pipeline for module {args.module_name} started with execution Arn {pipeline_execution.arn}")
    print("\n###### Waiting for pipeline execution to finish...")
    pipeline_execution.wait()
    print("\n###### Pipeline run completed. Details about run steps:")
    print(pipeline_execution.list_steps())


if __name__ == "__main__":
    main()
