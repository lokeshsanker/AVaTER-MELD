import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

from config import TEXT_MODEL_NAME, MAX_TEXT_LEN, LABEL2IDX
from text.asr import transcribe

_tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL_NAME)


class TextEmotionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["transcript"]
        if not text or (isinstance(text, float)):  # NaN or empty -> transcribe
            text = transcribe(row["wav_path"])

        enc = _tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        label = LABEL2IDX[row["label"]]
        return input_ids, attention_mask, label, row["clip_id"]
