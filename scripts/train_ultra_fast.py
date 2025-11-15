# """
# train_ultra_fast.py

# Versão ULTRA RÁPIDA para teste
# - CNN menor
# - Menos steps
# - Iterações reduzidas
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time

from src.environment.nesting_env import NestingEnvironment, NestingConfig
from src.geometry.polygon import create_rectangle

# =============================================================================
# CNN LEVE (para teste rápido)
# =============================================================================

class LightweightCNN(nn.Module):
    #"""CNN muito mais leve para testes rápidos"""
    
    def __init__(self, input_channels=6, embedding_dim=64):
        super().__init__()
        
        # Encoder ultra-leve
        self.conv1 = nn.Conv2d(input_channels, 16, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim)
        
        # Decoder para heatmap
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 16, 4, stride=4, padding=0)
        self.up3 = nn.ConvTranspose2d(16, 1, 4, stride=4, padding=0)
    
    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.conv1(x))  # 64×64
        x2 = torch.relu(self.conv2(x1))  # 16×16
        x3 = torch.relu(self.conv3(x2))  # 8×8
        
        # Embedding
        pooled = self.pool(x3)
        embedding = self.fc(pooled.flatten(1))
        
        # Heatmap
        h = torch.relu(self.up1(x3))
        h = torch.relu(self.up2(h))
        heatmap = torch.sigmoid(self.up3(h))
        
        return embedding, heatmap


# =============================================================================
# Actor-Critic LEVE
# =============================================================================

class LightActorCritic(nn.Module):
    def __init__(self, cnn_embedding_dim=64, hidden_dim=128, rotation_bins=36):
        super().__init__()
        
        self.cnn = LightweightCNN(input_channels=6, embedding_dim=cnn_embedding_dim)
        
        total_input = cnn_embedding_dim + 10 + 10 + 5
        
        self.shared = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_position_mean = nn.Linear(hidden_dim, 2)
        self.actor_position_logstd = nn.Parameter(torch.zeros(2))
        self.actor_rotation = nn.Linear(hidden_dim, rotation_bins)
        
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, observation):
        layout_image = observation['layout_image']
        cnn_embedding, heatmap = self.cnn(layout_image)
        
        combined = torch.cat([
            cnn_embedding,
            observation['current_piece'],
            observation['remaining_pieces'],
            observation['stats']
        ], dim=1)
        
        shared = self.shared(combined)
        
        position_mean = torch.sigmoid(self.actor_position_mean(shared))
        position_logstd = self.actor_position_logstd.expand_as(position_mean)
        rotation_logits = self.actor_rotation(shared)
        value = self.critic(shared)
        
        return position_mean, position_logstd, rotation_logits, value
    
    def get_action(self, observation, deterministic=False):
        pos_mean, pos_logstd, rot_logits, value = self.forward(observation)
        
        if deterministic:
            position = pos_mean
            rotation = torch.argmax(rot_logits, dim=-1)
        else:
            pos_std = torch.exp(pos_logstd)
            pos_dist = torch.distributions.Normal(pos_mean, pos_std)
            position = pos_dist.sample()
            position = torch.clamp(position, 0, 1)
            
            rot_probs = torch.softmax(rot_logits, dim=-1)
            rotation = torch.multinomial(rot_probs, 1).squeeze(-1)
        
        pos_dist = torch.distributions.Normal(pos_mean, torch.exp(pos_logstd))
        pos_log_prob = pos_dist.log_prob(position).sum(dim=-1)
        
        rot_probs = torch.softmax(rot_logits, dim=-1)
        rot_log_prob = torch.log(rot_probs.gather(1, rotation.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        total_log_prob = pos_log_prob + rot_log_prob
        
        return {
            'position': position,
            'rotation': rotation
        }, total_log_prob, value


# =============================================================================
# Trainer Simplificado
# =============================================================================

def train_ultra_fast():
    print("="*70)
    print("TREINAMENTO ULTRA RÁPIDO - TESTE")
    print("="*70)
    
    # Config MUITO leve
    config = {
        'n_steps': 64,       # ← 64 em vez de 512
        'n_iterations': 10,  # ← 10 em vez de 100
    }
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Peças simples
    pieces = [create_rectangle(50, 30)]
    pieces[0].id = 0
    
    # Environment
    env_config = NestingConfig(max_steps=10)
    env = NestingEnvironment(config=env_config)
    env.reset(options={'pieces': pieces})
    
    # Agent LEVE
    agent = LightActorCritic(
        cnn_embedding_dim=64,  # ← 64 em vez de 256
        hidden_dim=128,         # ← 128 em vez de 512
        rotation_bins=36
    ).to(device)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {n_params:,} parâmetros (muito menor!)\n")
    
    # Treinar
    print("Treinando...")
    start = time.time()
    
    for iteration in tqdm(range(config['n_iterations']), desc="Training"):
        # Coleta simplificada
        obs, _ = env.reset(options={'pieces': pieces})
        
        for step in range(config['n_steps']):
            obs_tensor = {
                'layout_image': torch.from_numpy(obs['layout_image']).unsqueeze(0),
                'current_piece': torch.from_numpy(obs['current_piece']).unsqueeze(0),
                'remaining_pieces': torch.from_numpy(obs['remaining_pieces']).unsqueeze(0),
                'stats': torch.from_numpy(obs['stats']).unsqueeze(0)
            }
            
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_tensor)
            
            action_dict = {
                'position': action['position'][0].numpy(),
                'rotation': int(action['rotation'][0].item())
            }
            
            obs, reward, terminated, truncated, info = env.step(action_dict)
            
            if terminated or truncated:
                obs, _ = env.reset(options={'pieces': pieces})
        
        if iteration % 2 == 0:
            print(f"  Iter {iteration}: ✓")
    
    elapsed = time.time() - start
    
    print(f"\n✓ Concluído em {elapsed:.1f}s")
    print(f"  Tempo/iter: {elapsed/config['n_iterations']:.2f}s")
    
    env.close()


if __name__ == "__main__":
    train_ultra_fast()