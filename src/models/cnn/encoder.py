import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import numpy as np

#CNN Encoder: Transforma layouts em embeddings e heatmaps
#Arquitetura baseada em ResNet com U-Net style decoder

class ResidualBlock(nn.Module):
    #"""Bloco residual básico"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class LayoutCNNEncoder(nn.Module):
    # """
    # CNN Encoder para layouts de nesting.
    
    # Input: Imagem 6-channel (256×256)
    #   - Channel 0: Ocupação (1=ocupado, 0=vazio)
    #   - Channel 1: Bordas
    #   - Channel 2: Mapa de distância
    #   - Channel 3: Próxima peça
    #   - Channel 4: Densidade
    #   - Channel 5: Acessibilidade
    
    # Output:
    #   - embedding: Vetor 256-dim representando estado
    #   - heatmap: Mapa 256×256 com "qualidade" de cada posição
    # """
    
    def __init__(self,
                 input_channels: int = 6,
                 embedding_dim: int = 256,
                 pretrained_backbone: bool = False):
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        
        # =====================================================================
        # ENCODER (Downsampling)
        # =====================================================================
        
        # Stem: Primeiro conv
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Output: 64 × 64 × 64
        
        # ResNet-style layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        # Output: 64 × 64 × 64
        
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        # Output: 128 × 32 × 32
        
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        # Output: 256 × 16 × 16
        
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        # Output: 512 × 8 × 8
        
        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # =====================================================================
        # DECODER (Upsampling) - U-Net style
        # =====================================================================
        
        # Upsampling layers
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 = 256 + 256 (skip)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Output: 256 × 16 × 16

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 + 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Output: 128 × 32 × 32

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Output: 64 × 64 × 64

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Apenas 32 (sem skip)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Output: 16 × 128 × 128

        # Final upsampling to 256×256
        self.final_up = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        # Output: 1 × 256 × 256
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int,
                   num_blocks: int, stride: int) -> nn.Sequential:
        #"""Cria uma sequência de blocos residuais"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        #"""Inicializa pesos da rede"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # """
    # Forward pass.
    
    # Args:
    #     x: Input tensor (batch_size, 6, 256, 256)
    
    # Returns:
    #     embedding: (batch_size, embedding_dim)
    #     heatmap: (batch_size, 1, 256, 256) - valores em [0,1]
    # """
    # # Encoder
        x0 = self.stem(x)           # 64 × 64 × 64
        x1 = self.layer1(x0)        # 64 × 64 × 64
        x2 = self.layer2(x1)        # 128 × 32 × 32
        x3 = self.layer3(x2)        # 256 × 16 × 16
        x4 = self.layer4(x3)        # 512 × 8 × 8
        
        # =========================================================================
        # BOTTLENECK → EMBEDDING
        # =========================================================================
        pooled = self.global_pool(x4)  # 512 × 1 × 1
        pooled = torch.flatten(pooled, 1)  # 512
        embedding = self.fc_embedding(pooled)  # embedding_dim
        
        # =========================================================================
        # DECODER (com skip connections nos níveis superiores)
        # =========================================================================
        
        # Level 4: 8×8 → 16×16
        d4 = self.up4(x4)  # 256 × 16 × 16
        d4 = torch.cat([d4, x3], dim=1)  # 512 × 16 × 16 (skip connection)
        d4 = self.dec4(d4)  # 256 × 16 × 16
        
        # Level 3: 16×16 → 32×32
        d3 = self.up3(d4)  # 128 × 32 × 32
        d3 = torch.cat([d3, x2], dim=1)  # 256 × 32 × 32 (skip connection)
        d3 = self.dec3(d3)  # 128 × 32 × 32
        
        # Level 2: 32×32 → 64×64
        d2 = self.up2(d3)  # 64 × 64 × 64
        d2 = torch.cat([d2, x1], dim=1)  # 128 × 64 × 64 (skip connection)
        d2 = self.dec2(d2)  # 64 × 64 × 64
        
        # Level 1: 64×64 → 128×128 (SEM skip connection)
        d1 = self.up1(d2)  # 32 × 128 × 128
        d1 = self.dec1(d1)  # 16 × 128 × 128
        
        # Final: 128×128 → 256×256
        d0 = self.final_up(d1)  # 8 × 256 × 256
        heatmap = self.final_conv(d0)  # 1 × 256 × 256
        heatmap = torch.sigmoid(heatmap)  # [0, 1]
        
        return embedding, heatmap


class LayoutCNNWithQuality(nn.Module):
    # """
    # CNN com predição de qualidade adicional.
    
    # Além do embedding e heatmap, prediz a qualidade geral do layout.
    # """
    
    def __init__(self,
                 input_channels: int = 6,
                 embedding_dim: int = 256):
        super().__init__()
        
        self.encoder = LayoutCNNEncoder(input_channels, embedding_dim)
        
        # Cabeça de qualidade
        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Qualidade em [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # """
        # Forward pass.
        
        # Returns:
        #     embedding: (batch_size, embedding_dim)
        #     heatmap: (batch_size, 1, 256, 256)
        #     quality: (batch_size, 1) - qualidade prevista [0,1]
        # """
        embedding, heatmap = self.encoder(x)
        quality = self.quality_head(embedding)
        
        return embedding, heatmap, quality


# =============================================================================
# Funções Auxiliares
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    #"""Conta número de parâmetros treináveis"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    #"""Testa o modelo com input dummy"""
    print("="*70)
    print("TESTE DO MODELO CNN")
    print("="*70)
    
    # Criar modelo
    model = LayoutCNNEncoder(input_channels=6, embedding_dim=256)
    
    # Contar parâmetros
    n_params = count_parameters(model)
    print(f"\nNúmero de parâmetros: {n_params:,}")
    print(f"Tamanho aproximado: {n_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Input dummy
    batch_size = 4
    x = torch.randn(batch_size, 6, 256, 256)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    print("\nExecutando forward pass...")
    model.eval()
    with torch.no_grad():
        embedding, heatmap = model(x)
    
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Heatmap shape: {heatmap.shape}")
    print(f"✓ Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Teste com qualidade
    print("\n" + "="*70)
    print("TESTE DO MODELO COM QUALIDADE")
    print("="*70)
    
    model_q = LayoutCNNWithQuality(input_channels=6, embedding_dim=256)
    n_params_q = count_parameters(model_q)
    print(f"\nNúmero de parâmetros: {n_params_q:,}")
    
    model_q.eval()
    with torch.no_grad():
        embedding, heatmap, quality = model_q(x)
    
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Heatmap shape: {heatmap.shape}")
    print(f"✓ Quality shape: {quality.shape}")
    print(f"✓ Quality range: [{quality.min():.3f}, {quality.max():.3f}]")
    
    # Teste de gradientes
    print("\n" + "="*70)
    print("TESTE DE GRADIENTES")
    print("="*70)
    
    model.train()
    x_train = torch.randn(2, 6, 256, 256, requires_grad=True)
    embedding, heatmap = model(x_train)
    
    # Loss dummy
    loss = embedding.mean() + heatmap.mean()
    loss.backward()
    
    print("✓ Backward pass executado com sucesso")
    print(f"✓ Gradiente do input: {x_train.grad is not None}")
    
    print("\n" + "="*70)
    print("✓ TODOS OS TESTES PASSARAM!")
    print("="*70)


if __name__ == "__main__":
    test_model()
    
    # Exemplo de uso
    print("\n\n" + "="*70)
    print("EXEMPLO DE USO")
    print("="*70)
    
    # Criar modelo
    model = LayoutCNNEncoder(
        input_channels=6,
        embedding_dim=256,
        pretrained_backbone=False
    )
    
    # Mover para GPU se disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nModelo movido para: {device}")
    
    # Criar input de exemplo
    batch_size = 8
    layout_images = torch.randn(batch_size, 6, 256, 256).to(device)
    
    # Inferência
    model.eval()
    with torch.no_grad():
        embeddings, heatmaps = model(layout_images)
    
    print(f"\nProcessado batch de {batch_size} layouts")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Heatmaps: {heatmaps.shape}")
    
    # Salvar modelo
    torch.save(model.state_dict(), 'cnn_encoder_example.pth')
    print("\n✓ Modelo salvo em 'cnn_encoder_example.pth'")
    
    # Carregar modelo
    model_loaded = LayoutCNNEncoder(input_channels=6, embedding_dim=256)
    model_loaded.load_state_dict(torch.load('cnn_encoder_example.pth'))
    print("✓ Modelo carregado com sucesso")