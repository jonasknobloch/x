#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PROTO_URL="https://raw.githubusercontent.com/onnx/onnx/v1.21.0/onnx/onnx.proto"
PROTO_SHA256="a05cfbcd1370608b809c5b84c44e3198d3369036458e0b5f297e76ceaf9c4e1b"

if [ ! -f onnx.proto ]; then
    curl -fsSL "$PROTO_URL" -o onnx.proto
fi

echo "$PROTO_SHA256  onnx.proto" | shasum -a 256 -c

protoc \
    --go_out=../internal/pb \
    --go_opt=paths=source_relative \
    --go_opt=Monnx.proto=go.jknobloc.com/x/onnx/internal/pb \
    onnx.proto
