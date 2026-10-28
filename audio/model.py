"""
Audio branch: Wav2Vec2-base embeddings, attention-pooled over time, fed into a
small MLP classifier head.

Swap WAV2VEC2_MODEL_NAME in config.py for a smaller model (e.g. a distilled
wav2vec2) if the base model is too heavy for your deployment target even after
quantization.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from config import WAV2VEC2_MODEL_NAME, AUDIO_HIDDEN_DIM, NUM_CLASSES


class AttentionPool1D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, seq, attention_mask=None):
        # seq: (B, T, H)
        scores = self.attn(seq).squeeze(-1)  # (B, T)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)
        return pooled


class AudioEmotionModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.pool = AttentionPool1D(AUDIO_HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(AUDIO_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        # input_values: (B, num_samples) raw waveform, 16kHz
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T', AUDIO_HIDDEN_DIM)

        # wav2vec2 downsamples time; its own feature-level attention mask, if needed,
        # can be derived from `encoder._get_feature_vector_attention_mask`.
        pooled = self.pool(hidden)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, input_values, attention_mask=None):
        self.eval()
        logits = self.forward(input_values, attention_mask)
        return torch.softmax(logits, dim=-1)
