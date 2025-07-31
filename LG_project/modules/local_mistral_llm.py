from langchain.llms.base import LLM
from typing import Optional, List, Any
import torch

class LocalMistralLLM(LLM):
    model: Any
    tokenizer: Any
    max_new_tokens: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "local_mistral_llm"
