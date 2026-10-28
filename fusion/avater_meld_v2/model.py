
import torch
import torch.nn as nn


class PairwiseCrossModalAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, source_tokens):
        attended, weights = self.attention(
            query=query_tokens,
            key=source_tokens,
            value=source_tokens,
            need_weights=True,
            average_attn_weights=False
        )

        output = self.norm1(query_tokens + self.dropout(attended))
        output = self.norm2(
            output + self.dropout(self.feed_forward(output))
        )

        return output, weights


class AVaTERMELDV2(nn.Module):
    def __init__(
        self,
        text_dim=768,
        audio_dim=1024,
        video_dim=7,
        hidden_dim=128,
        num_heads=4,
        num_classes=7,
        dropout=0.3
    ):
        super().__init__()

        if text_dim % 8 != 0:
            raise ValueError("text_dim must be divisible by 8")

        if audio_dim % 8 != 0:
            raise ValueError("audio_dim must be divisible by 8")

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.text_tokens_count = 8
        self.audio_tokens_count = 8
        self.video_tokens_count = video_dim

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim // 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim // 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.video_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.text_position = nn.Parameter(
            torch.randn(1, 8, hidden_dim) * 0.02
        )

        self.audio_position = nn.Parameter(
            torch.randn(1, 8, hidden_dim) * 0.02
        )

        self.video_position = nn.Parameter(
            torch.randn(1, video_dim, hidden_dim) * 0.02
        )

        self.text_modality = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.audio_modality = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.video_modality = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.text_to_audio = PairwiseCrossModalAttention(
            hidden_dim, num_heads, dropout
        )
        self.text_to_video = PairwiseCrossModalAttention(
            hidden_dim, num_heads, dropout
        )
        self.audio_to_text = PairwiseCrossModalAttention(
            hidden_dim, num_heads, dropout
        )
        self.audio_to_video = PairwiseCrossModalAttention(
            hidden_dim, num_heads, dropout
        )
        self.video_to_text = PairwiseCrossModalAttention(
            hidden_dim, num_heads, dropout
        )
        self.video_to_audio = PairwiseCrossModalAttention(
            hidden_dim, num_heads, dropout
        )

        combined_dim = hidden_dim * 6

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def encode_modalities(self, text, audio, video):
        batch_size = text.size(0)

        text = text.reshape(batch_size, 8, -1)
        audio = audio.reshape(batch_size, 8, -1)
        video = video.unsqueeze(-1)

        text = (
            self.text_projection(text)
            + self.text_position
            + self.text_modality
        )

        audio = (
            self.audio_projection(audio)
            + self.audio_position
            + self.audio_modality
        )

        video = (
            self.video_projection(video)
            + self.video_position
            + self.video_modality
        )

        return text, audio, video

    @staticmethod
    def pool(tokens):
        return tokens.mean(dim=1)

    def forward(self, text, audio, video, return_attention=False):
        text, audio, video = self.encode_modalities(
            text, audio, video
        )

        text_audio, weights_ta = self.text_to_audio(text, audio)
        text_video, weights_tv = self.text_to_video(text, video)

        audio_text, weights_at = self.audio_to_text(audio, text)
        audio_video, weights_av = self.audio_to_video(audio, video)

        video_text, weights_vt = self.video_to_text(video, text)
        video_audio, weights_va = self.video_to_audio(video, audio)

        fused = torch.cat([
            self.pool(text_audio),
            self.pool(text_video),
            self.pool(audio_text),
            self.pool(audio_video),
            self.pool(video_text),
            self.pool(video_audio)
        ], dim=1)

        logits = self.classifier(fused)

        if return_attention:
            attention = {
                "text_to_audio": weights_ta,
                "text_to_video": weights_tv,
                "audio_to_text": weights_at,
                "audio_to_video": weights_av,
                "video_to_text": weights_vt,
                "video_to_audio": weights_va
            }
            return logits, attention

        return logits
