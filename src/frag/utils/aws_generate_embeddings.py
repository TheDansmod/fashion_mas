# first step: need to generate jsonl files for submitting batch job
import time
from pathlib import Path
from typing import Callable, Any
import json

import boto3
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.config.container import Container

cfg = Container.config.provided

@inject
def generate_image_jsonl_files(
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    get_image_s3_key_from_index: Callable[[int], str] = PV[cfg.data.aws_fashion_gen.fashion_gen_image_s3_key_lambda],
    embedding_dim: int = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_dim],
    image_bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    account_id: str = PV[cfg.env.aws_account_id],
    records_per_file: int = PV[cfg.data.data_processing.aws_vec_db_gen.records_per_jsonl_file],
    jsonl_folder: str = PV[cfg.data.data_processing.aws_vec_db_gen.jsonl_folder_path],
):
    # create all the records
    records = []
    for idx in range(num_datapoints):
        image_s3_uri  = f"s3://{image_bucket_name}/{get_image_s3_key_from_index(idx)}"
        record = {
            # record id is expected to be a unique 11 char alpha-numeric string
            "recordId": f"{idx:a>11}",
            "modelInput": {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": embedding_dim,
                    "image": {
                        "detailLevel": "STANDARD_IMAGE",
                        "format": "png",
                        "source": {
                            "s3Location": {
                                "uri": image_s3_uri,
                                "bucketOwner": account_id
                            }
                        }
                    }
                }
            }
        }
        records.append(record)
    log.debug("Generated in-memory records for {} images.", num_datapoints)

    # split the records into jsonl files
    file_idx = 0
    for start in range(0, num_datapoints, records_per_file):
        chunk = records[start: start + records_per_file]
        local_path = f"{jsonl_folder}/batch_input_image_{file_idx:02d}.jsonl"
        with open(local_path, "w") as f:
            for rcrd in chunk:
                f.write(json.dumps(rcrd) + "\n")
        log.debug("created jsonl file number {}", file_idx)
        file_idx += 1
    log.debug("created jsonl files for all {} images.", num_datapoints)

@inject
async def generate_text_jsonl_files(
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    metadata_lookup: dict[int, Any] = PV[Container.metadata_lookup.provided],
    embedding_dim: int = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_dim],
    descriptions_key: str = PV[cfg.data.fashion_gen.descriptions_key],
    records_per_file: int = PV[cfg.data.data_processing.aws_vec_db_gen.records_per_jsonl_file],
    jsonl_folder: str = PV[cfg.data.data_processing.aws_vec_db_gen.jsonl_folder_path],
):
    # create all records
    records = []
    for idx in range(num_datapoints):
        record = {
            "recordId": f"{idx:b>11}",
            "modelInput": {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": embedding_dim,
                    "text": {
                        "value": metadata_lookup[idx][descriptions_key],
                        "truncationMode": "END"
                    }
                }
            }
        }
        records.append(record)
    log.debug("Generated in-memory records for {} text.", num_datapoints)

    # split the records into jsonl files
    file_idx = 0
    for start in range(0, num_datapoints, records_per_file):
        chunk = records[start: start + records_per_file]
        local_path = f"{jsonl_folder}/batch_input_text_{file_idx:02d}.jsonl"
        with open(local_path, "w") as f:
            for rcrd in chunk:
                f.write(json.dumps(rcrd) + "\n")
        log.debug("created jsonl file number {}", file_idx)
        file_idx += 1
    log.debug("created jsonl files for all {} text.", num_datapoints)

@inject
async def upload_jsonl_to_s3(
    jsonl_folder: str = PV[cfg.data.data_processing.aws_vec_db_gen.jsonl_folder_path],
    bucket_name: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_s3_bucket_name],
    get_s3_key: Callable[[str], str] = PV[cfg.data.data_processing.aws_vec_db_gen.jsonl_s3_key_lambda],
    s3_client = PV[Container.s3_client.provided],
):
    for item in Path(jsonl_folder).iterdir():
        if not item.is_dir():
            s3_key = get_s3_key(item.name)
            s3_client.upload_file(item, bucket_name, s3_key)
            log.debug("uploaded file {}", item.name)
    log.debug("uploaded all jsonl files.")

@inject
def create_bedrock_batch_role(
    account_id: str = PV[cfg.env.aws_account_id],
    bucket_name: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_s3_bucket_name],
    role_name: str = PV[cfg.data.data_processing.aws_vec_db_gen.bedrock_batch_inference_role_name],
    s3_access_policy: str = PV[cfg.data.data_processing.aws_vec_db_gen.bedrock_batch_s3_access_policy_name],
):
    iam = boto3.client("iam")

    # trust policy - allows bedrock to assume the role
    trust_policy = {
        "Version":"2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": account_id,
                    }
                }
            }
        ]
    }

    # permissions policy - allow access to s3 bucket with input and output data
    permissions_policy = {
        "Version":"2012-10-17",
        "Statement": [
            {
                "Sid": "S3Access",
                "Effect": "Allow",
                "Action": [
                   "s3:GetObject",
                   "s3:PutObject",
                   "s3:ListBucket"
                ],
                "Resource": [
                   f"arn:aws:s3:::{bucket_name}",
                   f"arn:aws:s3:::{bucket_name}/*",
                ],
                "Condition": {
                   "StringEquals": {
                       "aws:ResourceAccount": [account_id]
                   }
                }
            }
        ]
    }

    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Role for Bedrock Batch Inference Jobs",
    )
    role_arn = role["Role"]["Arn"]
    log.info("Created role ARN: {}", role_arn)

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=s3_access_policy,
        PolicyDocument=json.dumps(permissions_policy),
    )
    log.debug("attached permissions policy")


# THIS FUNCTION DOES NOT WORK FOR SOME REASON
@inject
def submit_batch_inference_jobs(
    bucket_name: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_s3_bucket_name],
    jsonl_folder: str = PV[cfg.data.data_processing.aws_vec_db_gen.jsonl_s3_folder_name],
    output_folder: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_output_folder_s3],
    model_id: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_model_id],
    get_s3_key: Callable[[str], str] = PV[cfg.data.data_processing.aws_vec_db_gen.jsonl_s3_key_lambda],
    role_arn: str = PV[cfg.env.aws_bedrock_batch_inference_role_arn],
    job_prefix: str = PV[cfg.data.data_processing.aws_vec_db_gen.batch_embedding_job_prefix],
):
    # we will have one submitted job per jsonl file
    bedrock = boto3.client(service_name="bedrock")
    for item_num, item in enumerate(Path(jsonl_folder).iterdir()):
        if item.is_dir():
            continue
        input_data_config = ({
            "s3InputDataConfig": {
                "s3Uri": f"s3://{bucket_name}/{get_s3_key(item.name)}",
            }
        })
        output_data_config = ({
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{bucket_name}/{output_folder}/"
            }
        })
        response = bedrock.create_model_invocation_job(
            roleArn=role_arn,
            modelId=model_id,
            jobName=f"{job_prefix}-item-{item_num}-ts-{time.time()}",
            inputDataConfig=input_data_config,
            outputDataConfig=output_data_config,
        )
        log.debug("job submitted. item num {}. arn {}", item_num, response.get('jobArn'))

# THIS FUNCTION DOES NOT WORK FOR SOME REASON
@inject
def submit_test_inference_job(
    bucket_name: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_s3_bucket_name],
    output_folder: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_output_folder_s3],
    job_prefix: str = PV[cfg.data.data_processing.aws_vec_db_gen.batch_embedding_job_prefix],
    role_arn: str = PV[cfg.env.aws_bedrock_batch_inference_role_arn],
    model_id: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_model_id],
):
    bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")
    input_data_config = ({
        "s3InputDataConfig": {
            "s3Uri": f"s3://{bucket_name}/jsonl_files/batch_test_text_01.jsonl",
        }
    })
    output_data_config = ({
        "s3OutputDataConfig": {
            "s3Uri": f"s3://{bucket_name}/{output_folder}/"
        }
    })
    response = bedrock.create_model_invocation_job(
        roleArn=role_arn,
        modelId=model_id,
        jobName=f"{job_prefix}-test-job-{time.time()}",
        inputDataConfig=input_data_config,
        outputDataConfig=output_data_config,
    )
    log.debug("job submitted. arn {}", item_num, response.get('jobArn'))

# below function works fine
@inject
def test_simple_model_invocation(
    model_id: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_model_id],
    embedding_dim: int = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_dim],
):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_dim,
                "text": {"value": "hello world", "truncationMode": "END"}
            }
        }),
        contentType="application/json",
        accept="application/json"
    )
    resp_body = json.loads(response["body"].read())
    log.debug(json.dumps(resp_body, indent=2))

@inject
def test_sync_embed_from_s3_src(
    model_id: str = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_model_id],
    embedding_dim: int = PV[cfg.data.data_processing.aws_vec_db_gen.embedding_dim],
    image_bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    get_image_s3_key_from_index: Callable[[int], str] = PV[cfg.data.aws_fashion_gen.fashion_gen_image_s3_key_lambda],
    account_id: str = PV[cfg.env.aws_account_id],
):
    client = boto3.client("bedrock-runtime")
    image_s3_uri  = f"s3://{image_bucket_name}/{get_image_s3_key_from_index(100)}"
    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": embedding_dim,
            "image": {
                "detailLevel": "STANDARD_IMAGE",
                "format": "png",
                "source": {
                    "s3Location": {
                        "uri": image_s3_uri,
                        "bucketOwner": account_id
                    }
                }
            }
        }
    }
    try:
        response = client.invoke_model(
            body=json.dumps(request_body, indent=2),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
    except Exception as e:
        log.exception("There was an exception while getting embedding of an image on s3.")

    resp_body = json.loads(response["body"].read())
    log.debug(json.dumps(resp_body, indent=2))
    embedding = resp_body["embeddings"][0]["embedding"]
    log.debug("embedding length: {}", len(embedding))

async def main():
    test_sync_embed_from_s3_src()
