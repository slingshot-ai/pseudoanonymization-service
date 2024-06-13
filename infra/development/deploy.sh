#!/bin/bash

set -e

echo "Deploy Cloud Run Service"
gcloud run services replace pii-service-service.yaml
