#!/bin/bash

export AWS_ACCESS_KEY_ID=AKIA4PPQC67BSKF5VRMJ
export AWS_SECRET_ACCESS_KEY="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"
export BUCKET_NAME="autotag-storage-us-east"
export REGION_NAME="us-east-2"

export PRETAGGER_PORT=5000

venv/bin/python3 app.py
