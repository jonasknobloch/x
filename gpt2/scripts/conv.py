from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_id = "gpt2"

cpu_provider = "CPUExecutionProvider"
cuda_provider = "CUDAExecutionProvider"

model = ORTModelForCausalLM.from_pretrained(
    model_id,
    provider=cpu_provider,
    export=True,
    use_cache=True,
)

model.save_pretrained("../models/base")

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.save_pretrained("../models/base")
