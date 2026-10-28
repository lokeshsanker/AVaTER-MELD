"""
Text branch: DistilBERT fine-tuned for EMOTION classification (not generic
sentiment) so its output label space matches the video and audio branches
exactly.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

from config import TEXT_MODEL_NAME, NUM_CLASSES


class TextEmotionModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(TEXT_MODEL_NAME)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS]-equivalent token
        logits = self.classifier(cls_token)
        return logits

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask):
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)
