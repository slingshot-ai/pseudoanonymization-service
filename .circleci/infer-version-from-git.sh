#!/bin/bash
set -e

if [ -n "${CIRCLE_TAG}" ]; then
  export VERSION=${CIRCLE_TAG}
else
  export VERSION=$(git describe --tags)
fi

echo $VERSION
