"""
Multi-modal diagnostic model for federated learning
Combines radiology and genomics data for comprehensive diagnosis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class RadiologyEncoder(nn.Module):
    """Encoder for radiological images (DICOM, MRI, CT)"""
    
    def __init__(self, input_channels: int = 1, embedding_dim: int = 768):
        super().__init__()
        
        # 3D CNN for volumetric medical images
        self.conv_layers = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.fc = nn.Linear(256, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, depth, height, width)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class GenomicsEncoder(nn.Module):
    """Encoder for genomic sequence data"""
    
    def __init__(self, vocab_size: int = 4, max_seq_len: int = 100, embedding_dim: int = 768):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, 256)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Use pooling instead of flattening to reduce memory
        self.fc = nn.Linear(256, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)
        transformed = self.transformer(embedded)
        # Global average pooling over sequence dimension
        pooled = transformed.mean(dim=1)
        return self.fc(pooled)


class MultiModalFusion(nn.Module):
    """Fusion module for combining radiology and genomics embeddings"""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        
        # Cross-modal attention
        self.radiology_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        self.genomics_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, radiology_emb: torch.Tensor, genomics_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # radiology_emb: (batch, embedding_dim)
        # genomics_emb: (batch, embedding_dim)
        
        # Add sequence dimension for attention
        r_seq = radiology_emb.unsqueeze(0)
        g_seq = genomics_emb.unsqueeze(0)
        
        # Cross-modal attention
        r_attended, _ = self.radiology_attention(r_seq, g_seq, g_seq)
        g_attended, _ = self.genomics_attention(g_seq, r_seq, r_seq)
        
        # Remove sequence dimension
        r_attended = r_attended.squeeze(0)
        g_attended = g_attended.squeeze(0)
        
        # Concatenate and fuse
        combined = torch.cat([r_attended, g_attended], dim=-1)
        fused = self.fusion_layers(combined)
        
        # Classification logits
        logits = self.classifier(fused)
        
        return fused, logits


class FederatedDiagnosticModel(nn.Module):
    """Complete multi-modal diagnostic model for federated learning"""
    
    def __init__(
        self,
        radiology_channels: int = 1,
        genomics_vocab_size: int = 4,
        genomics_max_len: int = 1000,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 10
    ):
        super().__init__()
        
        self.radiology_encoder = RadiologyEncoder(
            input_channels=radiology_channels,
            embedding_dim=embedding_dim
        )
        
        self.genomics_encoder = GenomicsEncoder(
            vocab_size=genomics_vocab_size,
            max_seq_len=genomics_max_len,
            embedding_dim=embedding_dim
        )
        
        self.fusion = MultiModalFusion(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
    def forward(
        self,
        radiology_data: torch.Tensor,
        genomics_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            radiology_data: (batch, channels, depth, height, width)
            genomics_data: (batch, seq_len)
            
        Returns:
            radiology_embeddings, genomics_embeddings, fusion_features, logits
        """
        radiology_emb = self.radiology_encoder(radiology_data)
        genomics_emb = self.genomics_encoder(genomics_data)
        fused_features, logits = self.fusion(radiology_emb, genomics_emb)
        
        return radiology_emb, genomics_emb, fused_features, logits
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension for vector storage"""
        return 768
