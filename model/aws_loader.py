import os
import boto3
import argparse

from dotenv import load_dotenv

load_dotenv()


def load_model_from_s3(bucket_name: str, object_name: str, save_path: str):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3_client.download_file(bucket_name, object_name, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Load model from S3.')
    parser.add_argument("--bucket-name", type=str, default="mlops-study-project", help="bucket name")
    parser.add_argument("--object-name", type=str, default="dog_cat_classification_models/resnet152.onnx", help="object name")
    parser.add_argument("--save-path", type=str, default="./model/resnet152.onnx", help="Model save path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model_from_s3(args.bucket_name, args.object_name, args.save_path)
