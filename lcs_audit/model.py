import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BlackBoxModel:
    def __init__(self, name, device=None, dtype=None):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

    def encode(self, text):
        return self.tokenizer.encode(text)

    def query(self, prefix_ids):
        input_ids = torch.tensor([prefix_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=1, do_sample=False,
                pad_token_id=self.pad_id,
            )
        return int(out[0, -1].item())
