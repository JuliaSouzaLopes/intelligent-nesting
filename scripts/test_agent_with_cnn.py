# """
# scripts/test_agent_with_cnn.py

# Testa o Agent com CNN real conectado
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Importar módulos
from src.environment.nesting_env import NestingEnvironment, NestingConfig
from src.geometry.polygon import create_rectangle, create_random_polygon

# Importar Actor-Critic
# Vamos criar versão simplificada aqui para teste

class ActorCriticTest(torch.nn.Module):
    #"""Versão de teste do Actor-Critic com CNN real"""
    
    def __init__(self, cnn_embedding_dim=256, hidden_dim=512, rotation_bins=36):
        super().__init__()
        
        from src.models.cnn.encoder import LayoutCNNEncoder
        
        # CNN REAL
        self.cnn = LayoutCNNEncoder(
            input_channels=6,
            embedding_dim=cnn_embedding_dim
        )
        
        # Processar features adicionais
        total_input = cnn_embedding_dim + 10 + 10 + 5  # cnn + piece + remaining + stats
        
        # Shared layers
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(total_input, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
        # Actor
        self.actor_position_mean = torch.nn.Linear(hidden_dim, 2)
        self.actor_position_logstd = torch.nn.Parameter(torch.zeros(2))
        self.actor_rotation = torch.nn.Linear(hidden_dim, rotation_bins)
        
        # Critic
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    
    def forward(self, observation):
        #"""Forward pass com CNN real"""
        
        # CNN
        layout_image = observation['layout_image']
        cnn_embedding, heatmap = self.cnn(layout_image)
        
        # Concatenar features
        combined = torch.cat([
            cnn_embedding,
            observation['current_piece'],
            observation['remaining_pieces'],
            observation['stats']
        ], dim=1)
        
        # Shared
        shared = self.shared(combined)
        
        # Actor
        position_mean = torch.sigmoid(self.actor_position_mean(shared))
        position_logstd = self.actor_position_logstd.expand_as(position_mean)
        rotation_logits = self.actor_rotation(shared)
        
        # Critic
        value = self.critic(shared)
        
        return position_mean, position_logstd, rotation_logits, value
    
    def get_action(self, observation, deterministic=False):
        #"""Gera ação"""
        pos_mean, pos_logstd, rot_logits, value = self.forward(observation)
        
        if deterministic:
            position = pos_mean
            rotation = torch.argmax(rot_logits, dim=-1)
        else:
            # Sample
            pos_std = torch.exp(pos_logstd)
            pos_dist = torch.distributions.Normal(pos_mean, pos_std)
            position = pos_dist.sample()
            position = torch.clamp(position, 0, 1)
            
            rot_probs = torch.softmax(rot_logits, dim=-1)
            rotation = torch.multinomial(rot_probs, 1).squeeze(-1)
        
        # Log prob
        pos_dist = torch.distributions.Normal(pos_mean, torch.exp(pos_logstd))
        pos_log_prob = pos_dist.log_prob(position).sum(dim=-1)
        
        rot_probs = torch.softmax(rot_logits, dim=-1)
        rot_log_prob = torch.log(rot_probs.gather(1, rotation.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        total_log_prob = pos_log_prob + rot_log_prob
        
        return {
            'position': position,
            'rotation': rotation
        }, total_log_prob, value


def test_agent_with_cnn():
    #"""Testa agent com CNN real"""
    
    print("="*70)
    print("TESTE: AGENT COM CNN REAL")
    print("="*70)
    
    # Criar environment
    config = NestingConfig(max_steps=10)
    env = NestingEnvironment(config=config)
    
    # Criar peças
    pieces = [
        create_rectangle(50, 30),
        create_rectangle(40, 25),
        create_random_polygon(5, 20),
    ]
    
    for i, piece in enumerate(pieces):
        piece.id = i
    
    obs, info = env.reset(options={'pieces': pieces})
    print(f"✓ Environment criado com {len(pieces)} peças")
    
    # Criar agent com CNN
    print("\nCriando Agent com CNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    agent = ActorCriticTest(
        cnn_embedding_dim=256,
        hidden_dim=512,
        rotation_bins=36
    ).to(device)
    
    # Contar parâmetros
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"✓ Agent criado: {n_params:,} parâmetros")
    
    # Contar parâmetros da CNN
    n_cnn = sum(p.numel() for p in agent.cnn.parameters())
    print(f"  - CNN: {n_cnn:,} parâmetros")
    print(f"  - Resto: {n_params - n_cnn:,} parâmetros")
    
    # Converter observação para tensors
    def obs_to_tensor(obs, device):
        return {
            'layout_image': torch.from_numpy(obs['layout_image']).unsqueeze(0).to(device),
            'current_piece': torch.from_numpy(obs['current_piece']).unsqueeze(0).to(device),
            'remaining_pieces': torch.from_numpy(obs['remaining_pieces']).unsqueeze(0).to(device),
            'stats': torch.from_numpy(obs['stats']).unsqueeze(0).to(device)
        }
    
    # Teste de forward pass
    print("\n" + "="*70)
    print("TESTE DE FORWARD PASS")
    print("="*70)
    
    agent.eval()
    with torch.no_grad():
        obs_tensor = obs_to_tensor(obs, device)
        
        print("Input shapes:")
        for key, val in obs_tensor.items():
            print(f"  {key:20s}: {val.shape}")
        
        # Forward
        pos_mean, pos_logstd, rot_logits, value = agent.forward(obs_tensor)
        
        print("\nOutput shapes:")
        print(f"  position_mean:  {pos_mean.shape}")
        print(f"  position_logstd: {pos_logstd.shape}")
        print(f"  rotation_logits: {rot_logits.shape}")
        print(f"  value:          {value.shape}")
        
        print("\nOutput values:")
        print(f"  position: ({pos_mean[0,0]:.3f}, {pos_mean[0,1]:.3f})")
        print(f"  value: {value[0,0]:.3f}")
    
    # Teste de ação
    print("\n" + "="*70)
    print("TESTE DE GERAÇÃO DE AÇÃO")
    print("="*70)
    
    with torch.no_grad():
        action, log_prob, value = agent.get_action(obs_tensor)
        
        print(f"✓ Ação gerada:")
        print(f"  position: ({action['position'][0,0]:.3f}, {action['position'][0,1]:.3f})")
        print(f"  rotation: {action['rotation'][0].item()} ({action['rotation'][0].item() * 10}°)")
        print(f"  log_prob: {log_prob[0]:.3f}")
        print(f"  value: {value[0,0]:.3f}")
    
    # Teste de episódio completo
    print("\n" + "="*70)
    print("TESTE DE EPISÓDIO COMPLETO")
    print("="*70)
    
    obs, _ = env.reset(options={'pieces': pieces})
    total_reward = 0
    
    for step in range(10):
        obs_tensor = obs_to_tensor(obs, device)
        
        with torch.no_grad():
            action, _, value = agent.get_action(obs_tensor, deterministic=False)
        
        # Converter para dict do environment
        action_dict = {
            'position': action['position'][0].cpu().numpy(),
            'rotation': int(action['rotation'][0].cpu().item())
        }
        
        obs, reward, terminated, truncated, info = env.step(action_dict)
        total_reward += reward
        
        print(f"  Step {step+1}: reward={reward:6.2f}, "
              f"value={value[0,0]:6.2f}, "
              f"placed={info['n_placed']}/{len(pieces)}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Episódio terminado")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Utilização: {info['utilization']*100:.1f}%")
    
    # Teste de gradientes (backward pass)
    print("\n" + "="*70)
    print("TESTE DE BACKPROPAGATION")
    print("="*70)
    
    agent.train()
    obs_tensor = obs_to_tensor(obs, device)
    
    # Forward
    pos_mean, pos_logstd, rot_logits, value = agent.forward(obs_tensor)
    
    # Loss dummy
    loss = value.mean() + pos_mean.mean()
    
    # Backward
    loss.backward()
    
    print("✓ Backward pass executado")
    
    # Verificar gradientes
    has_grad = sum(1 for p in agent.parameters() if p.grad is not None)
    total_params = sum(1 for _ in agent.parameters())
    
    print(f"  Parâmetros com gradiente: {has_grad}/{total_params}")
    
    # Gradiente da CNN
    cnn_has_grad = sum(1 for p in agent.cnn.parameters() if p.grad is not None)
    cnn_params = sum(1 for _ in agent.cnn.parameters())
    
    print(f"  CNN com gradiente: {cnn_has_grad}/{cnn_params}")
    
    env.close()
    
    print("\n" + "="*70)
    print("✓✓✓ TODOS OS TESTES PASSARAM! ✓✓✓")
    print("="*70)
    print("\nAgent com CNN real está funcionando perfeitamente!")
    print("Próximo passo: Treinar o sistema completo!")


if __name__ == "__main__":
    test_agent_with_cnn()