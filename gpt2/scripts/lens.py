#!/usr/bin/env python3

import argparse
import copy
import os
import re

import onnx


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, help='directory containing model.onnx')
    parser.add_argument('--name', default='model.onnx', help='model filename within --model dir')

    return parser.parse_args()


def find_layer_outputs(graph):
    pattern = re.compile(r'^/transformer/h\.(\d+)/Add_1$')

    layer_outputs = {}

    for node in graph.node:
        m = pattern.match(node.name)

        if m:
            layer_outputs[int(m.group(1))] = node.output[0]

    return [layer_outputs[i] for i in sorted(layer_outputs)]


def make_lens_model(base_model, cut_tensor):
    m = copy.deepcopy(base_model)

    g = m.graph

    for node in g.node:
        if node.name == '/transformer/ln_f/LayerNormalization':
            node.input[0] = cut_tensor

            break

    return m


def main():
    args = parse_args()

    model_dir = os.path.abspath(args.model)

    base_model = onnx.load(os.path.join(model_dir, args.name), load_external_data=False)

    layer_outputs = find_layer_outputs(base_model.graph)

    n = len(layer_outputs)

    stem = os.path.splitext(args.name)[0]

    for k in range(n - 1):
        cut_tensor = layer_outputs[k]
        m = make_lens_model(base_model, cut_tensor)
        onnx.save(m, os.path.join(model_dir, f"{stem}_lens_{k}.onnx"))


if __name__ == '__main__':
    main()
