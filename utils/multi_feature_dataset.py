import torch
from torch.utils.data import Dataset

class MultiFeatureDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        """
        Args:
            data: List of samples (each a dict with keys "parent_text", "axes", and "target_values").
            tokenizer: A HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.axis_names = ["main_outcome_axis", "static_axis", "generic_axis", "hypothetical_axis", "negation_axis", "intention_axis", "opinion_axis", "recurrent_axis"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        parent_txt = sample["parent_text"]
        parent_enc = self.tokenizer(
            parent_txt,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        axis_encs = []
        for axis in self.axis_names:
            axis_txt = sample["axes"].get(axis, "")
            enc = self.tokenizer(
                axis_txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            axis_encs.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)
            })
        parent_feature = {
            "input_ids": parent_enc["input_ids"].squeeze(0),
            "attention_mask": parent_enc["attention_mask"].squeeze(0)
        }
        target = torch.tensor([
            sample["target_values"]["location"],
            sample["target_values"]["scale"],
            sample["target_values"]["skewness"]
        ], dtype=torch.float32)
        return {
            "parent_feature": parent_feature,
            "axis_features": axis_encs,
            "targets": target
        }

def collate_fn(batch):
    """
    Collate function to combine samples into a batch.
    It creates tensors of shape:
      - input_ids: [n_features, batch_size, seq_len]
      - attention_masks: [n_features, batch_size, seq_len]
      - targets: [batch_size, 3]
    """
    bsz = len(batch)
    n_axes = len(batch[0]["axis_features"])
    seq_len = batch[0]["axis_features"][0]["input_ids"].shape[0]
    n_features = 1 + n_axes  # Parent + axes
    input_ids = torch.zeros(n_features, bsz, seq_len, dtype=torch.long)
    attn_masks = torch.zeros(n_features, bsz, seq_len, dtype=torch.long)
    targets = torch.zeros(bsz, 3, dtype=torch.float32)
    for i, sample in enumerate(batch):
        input_ids[0, i] = sample["parent_feature"]["input_ids"]
        attn_masks[0, i] = sample["parent_feature"]["attention_mask"]
        for j in range(n_axes):
            input_ids[j+1, i] = sample["axis_features"][j]["input_ids"]
            attn_masks[j+1, i] = sample["axis_features"][j]["attention_mask"]
        targets[i] = sample["targets"]
    return {
        "input_ids": input_ids,
        "attention_masks": attn_masks,
        "targets": targets
    }
