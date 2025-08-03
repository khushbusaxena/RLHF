# data/dataset.py

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class HHRLHFPreferenceDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="google/flan-t5-small", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.dataset = load_dataset("Dahoas/full-hh-rlhf", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Tokenize the preferred (chosen) response
        preferred = self.tokenizer(
            prompt + " " + chosen,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize the non-preferred (rejected) response
        non_preferred = self.tokenizer(
            prompt + " " + rejected,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "preferred_input_ids": preferred["input_ids"].squeeze(0),
            "preferred_attention_mask": preferred["attention_mask"].squeeze(0),
            "non_preferred_input_ids": non_preferred["input_ids"].squeeze(0),
            "non_preferred_attention_mask": non_preferred["attention_mask"].squeeze(0),
            "label": 1  # Always 1, because chosen > rejected
        }
