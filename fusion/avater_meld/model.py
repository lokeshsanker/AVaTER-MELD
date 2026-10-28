import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """One directional cross-modal attention: query modality attends to source."""

    def __init__(self, dim=256, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, query_features, source_features):
        q = self.query(query_features).unsqueeze(1)
        k = self.key(source_features).unsqueeze(1)
        v = self.value(source_features).unsqueeze(1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).squeeze(1)
        return self.output(attended + query_features)


class AVaTERMELD(nn.Module):
    """
    AVaTER-inspired feature-level model for seven-class MELD emotion recognition.
    Uses six directional cross-modal attention operations.
    """

    def __init__(self, text_dim=768, audio_dim=768,
                 video_dim=512, hidden_dim=256,
                 num_classes=7, dropout=0.3):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.att_av = CrossModalAttention(hidden_dim, dropout)
        self.att_va = CrossModalAttention(hidden_dim, dropout)
        self.att_at = CrossModalAttention(hidden_dim, dropout)
        self.att_ta = CrossModalAttention(hidden_dim, dropout)
        self.att_vt = CrossModalAttention(hidden_dim, dropout)
        self.att_tv = CrossModalAttention(hidden_dim, dropout)

        # Six attended representations, each hidden_dim wide.
        fused_dim = 6 * hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, text, audio, video):
        t = self.text_projection(text)
        a = self.audio_projection(audio)
        v = self.video_projection(video)

        # The first letter is the query modality; the second is the source.
        attended = [
            self.att_av(a, v),  # Audio attends to video
            self.att_at(a, t),  # Audio attends to text
            self.att_vt(v, t),  # Video attends to text
            self.att_va(v, a),  # Video attends to audio
            self.att_ta(t, a),  # Text attends to audio
            self.att_tv(t, v),  # Text attends to video
        ]

        fused = torch.cat(attended, dim=1)
        return self.classifier(fused)
