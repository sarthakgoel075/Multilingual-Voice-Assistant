
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, os

def load_mistral(model_path):
    assert os.path.exists(model_path), " Model path doesn't exist"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        legacy=True,
        local_files_only=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    return model, tokenizer
