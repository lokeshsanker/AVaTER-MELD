import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

from config import AUDIO_SAMPLE_RATE, WAV2VEC2_MODEL_NAME, LABEL2IDX

_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_NAME)


class AudioEmotionDataset(Dataset):
    def __init__(self, df, max_seconds: float = 3.5):
        self.df = df.reset_index(drop=True)
        self.max_samples = int(max_seconds * AUDIO_SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, _ = librosa.load(row["wav_path"], sr=AUDIO_SAMPLE_RATE, mono=True)

        # pad/truncate to a fixed max length for simple batching
        if len(wav) > self.max_samples:
            wav = wav[: self.max_samples]
        else:
            wav = np.pad(wav, (0, self.max_samples - len(wav)))

        inputs = _feature_extractor(wav, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")
        input_values = inputs["input_values"].squeeze(0)  # (num_samples,)

        label = LABEL2IDX[row["label"]]
        return input_values, label, row["clip_id"]
