#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("../models/base")

model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained("../models/base")
