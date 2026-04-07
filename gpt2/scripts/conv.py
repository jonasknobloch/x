#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from optimum.onnxruntime import ORTModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, help='directory containing model.safetensors')

    return parser.parse_args()


def add_log_probs(model_path: str, output_path: str):
    m = onnx.load(model_path)

    g = m.graph

    INT64_MAX = 9223372036854775807

    def init(name, values, dtype=np.int64):
        g.initializer.append(numpy_helper.from_array(np.array(values, dtype=dtype), name=name))

    # 1D scalar-like constants (shape [1]) for use in Slice starts/ends/axes
    init("_lp_zero", [0])
    init("_lp_one", [1])
    init("_lp_int_max", [INT64_MAX])
    init("_lp_axis1", [1])
    init("_lp_axis2", [2])

    def node(op, inputs, outputs, **attrs):
        g.node.append(helper.make_node(op, inputs=inputs, outputs=outputs, **attrs))

    # log_softmax(logits) over vocab dim → [1, seq, vocab]
    node("LogSoftmax", ["logits"], ["_lp_lsm"], axis=2)

    # seq_len as shape-[1] tensor
    node("Shape", ["logits"], ["_lp_shape"])
    node("Gather", ["_lp_shape", "_lp_one"], ["_lp_seq_len"])  # shape [1]

    # seq_len - 1 → shape [1]
    node("Sub", ["_lp_seq_len", "_lp_one"], ["_lp_seq_len_m1"])

    # log_softmax[:, 0:seq_len-1, :] → [1, seq-1, vocab]
    node("Slice", ["_lp_lsm", "_lp_zero", "_lp_seq_len_m1", "_lp_axis1"], ["_lp_lsm_s"])

    # input_ids[:, 1:] → [1, seq-1]
    node("Slice", ["input_ids", "_lp_one", "_lp_int_max", "_lp_axis1"], ["_lp_ids_s"])

    # unsqueeze → [1, seq-1, 1]  (axes as input for opset 13+)
    node("Unsqueeze", ["_lp_ids_s", "_lp_axis2"], ["_lp_ids_3d"])

    # gather the log prob for each target token → [1, seq-1, 1]
    node("GatherElements", ["_lp_lsm_s", "_lp_ids_3d"], ["_lp_gathered"], axis=2)

    # squeeze trailing dim → [1, seq-1]
    node("Squeeze", ["_lp_gathered", "_lp_axis2"], ["token_logprobs"])

    g.output.append(
        helper.make_tensor_value_info("token_logprobs", TensorProto.FLOAT, [None, None])
    )

    onnx.save(m, output_path)


def main():
    args = parse_args()

    model_dir = os.path.abspath(args.model)

    with tempfile.TemporaryDirectory() as tmp:
        no_cache_dir = os.path.join(tmp, "base")

        ORTModelForCausalLM.from_pretrained(model_dir, export=True, use_cache=False).save_pretrained(no_cache_dir)
        shutil.copy2(os.path.join(no_cache_dir, "model.onnx"), os.path.join(model_dir, "model.onnx"))

        cache_dir = os.path.join(tmp, "cache")

        ORTModelForCausalLM.from_pretrained(model_dir, export=True, use_cache=True).save_pretrained(cache_dir)
        shutil.copy2(os.path.join(cache_dir, "model.onnx"), os.path.join(model_dir, "model_cache.onnx"))

    add_log_probs(os.path.join(model_dir, "model.onnx"), os.path.join(model_dir, "model_eval.onnx"))


if __name__ == '__main__':
    main()
