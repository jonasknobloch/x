#!/usr/bin/env python3

import json

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

cpu_provider = "CPUExecutionProvider"
cuda_provider = "CUDAExecutionProvider"

model = ORTModelForCausalLM.from_pretrained(
    "../models/base",
    provider=cuda_provider,
    local_files_only=True,
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained("../models/base")

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

print(inputs)

outputs = model(**inputs)
logits = outputs.logits

export = logits.squeeze(0).detach().cpu().numpy().astype("float32")

export.tofile("logits.f32")

seq_len, vocab = export.shape

with open("shape.json", "w") as f:
    json.dump({"seq_len": int(seq_len), "vocab": int(vocab)}, f)
