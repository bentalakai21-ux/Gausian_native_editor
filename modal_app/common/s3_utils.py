from __future__ import annotations

import os
import pathlib
from typing import Iterable, Tuple, Optional, List

def s3_client():
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "boto3 is required for S3 operations. "
            "Install requirements with `pip install -r modal_app/requirements.txt`."
        ) from exc
    return boto3.client("s3")


def parse_s3_url(url: str) -> Tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"Not an s3 url: {url}")
    rest = url[5:]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in {url}")
    return bucket, key


def upload_file(local_path: str | pathlib.Path, bucket: str, key: str) -> None:
    local_path = str(local_path)
    s3_client().upload_file(local_path, bucket, key)


def download_file(bucket: str, key: str, local_path: str | pathlib.Path) -> None:
    local_path = str(local_path)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(bucket, key, local_path)


def list_keys(bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        if token:
            resp = s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token)
        else:
            resp = s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix)
        for c in resp.get("Contents", []):
            keys.append(c["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def put_json(bucket: str, key: str, data: dict) -> None:
    import json
    body = json.dumps(data).encode("utf-8")
    s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def get_json(bucket: str, key: str) -> dict:
    import json
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())
