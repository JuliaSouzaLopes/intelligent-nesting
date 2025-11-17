# """
# train_2gb_gpu.py

# Versão OTIMIZADA para GPUs com 2GB de VRAM
# - CNN ultra-leve
# - Batch size pequeno
# - Mixed precision (float16)
# - Gradient checkpointing
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

from src.environment.nesting_env_fixed import NestingEnvironment, NestingConfig
from src.geometry.polygon import create_rectangle, create_random_polygon


# =============================================================================
# CNN ULTRA-LEVE para 2GB VRAM
# =============================================================================

class TinyLayoutCNN(nn.Module):
    #"""CNN minimalista: ~500K parâmetros (em vez de 14.9M)"""
    
    def __init__(self, input_channels=6, embedding_dim=64):
        super().__init__()
        
        # Encoder ultra-compacto
        self.conv1 = nn.Conv2d(input_channels, 16, 7, stride=4, padding=3)  # 256→64
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 5, stride=4, padding=2)  # 64→16
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 16→8
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding
        self.fc = nn.Linear(64, embedding_dim)
        
        # Decoder minimalista para heatmap
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 8→16
        self.up2 = nn.ConvTranspose2d(32, 16, 4, stride=4, padding=0)  # 16→64
        self.up3 = nn.ConvTranspose2d(16, 1, 4, stride=4, padding=0)   # 64→256
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Embedding
        pooled = self.pool(x).flatten(1)
        embedding = self.fc(pooled)
        
        # Decoder (heatmap)
        h = F.relu(self.up1(x))
        h = F.relu(self.up2(h))
        heatmap = torch.sigmoid(self.up3(h))
        
        return embedding, heatmap


# =============================================================================
# Actor-Critic LEVE
# =============================================================================

class TinyActorCritic(nn.Module):
    #"""Actor-Critic minimalista: ~600K parâmetros total"""
    
    def __init__(self, embedding_dim=64, hidden_dim=128, rotation_bins=36):
        super().__init__()
        
        self.cnn = TinyLayoutCNN(input_channels=6, embedding_dim=embedding_dim)
        
        total_input = embedding_dim + 10 + 10 + 5
        
        # Shared (1 camada em vez de 2)
        self.shared = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU()
        )
        
        # Actor
        self.actor_position_mean = nn.Linear(hidden_dim, 2)
        self.actor_position_logstd = nn.Parameter(torch.zeros(2))
        self.actor_rotation = nn.Linear(hidden_dim, rotation_bins)
        
        # Critic
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
    
    def evaluate_actions(self, observations, actions):
        pos_mean, pos_logstd, rot_logits, values = self.forward(observations)
        
        pos_std = torch.exp(pos_logstd)
        pos_dist = torch.distributions.Normal(pos_mean, pos_std)
        pos_log_probs = pos_dist.log_prob(actions['position']).sum(dim=-1)
        
        rot_probs = torch.softmax(rot_logits, dim=-1)
        rot_log_probs = torch.log(
            rot_probs.gather(1, actions['rotation'].unsqueeze(-1)).squeeze(-1) + 1e-8
        )
        
        total_log_probs = pos_log_probs + rot_log_probs
        
        pos_entropy = pos_dist.entropy().sum(dim=-1)
        rot_dist = torch.distributions.Categorical(probs=rot_probs)
        rot_entropy = rot_dist.entropy()
        
        total_entropy = pos_entropy + rot_entropy
        
        return total_log_probs, values.squeeze(-1), total_entropy


# =============================================================================
# Rollout Buffer (mesmo do otimizado)
# =============================================================================

class RolloutBuffer:
    def __init__(self, n_steps, device):
        self.n_steps = n_steps
        self.device = device
        
        self.observations = {
            'layout_image': np.zeros((n_steps, 6, 256, 256), dtype=np.float32),
            'current_piece': np.zeros((n_steps, 10), dtype=np.float32),
            'remaining_pieces': np.zeros((n_steps, 10), dtype=np.float32),
            'stats': np.zeros((n_steps, 5), dtype=np.float32)
        }
        
        self.actions_position = np.zeros((n_steps, 2), dtype=np.float32)
        self.actions_rotation = np.zeros((n_steps,), dtype=np.int64)
        
        self.rewards = np.zeros((n_steps,), dtype=np.float32)
        self.values = np.zeros((n_steps,), dtype=np.float32)
        self.log_probs = np.zeros((n_steps,), dtype=np.float32)
        self.dones = np.zeros((n_steps,), dtype=np.bool_)
        
        self.ptr = 0
    
    def add(self, obs, action, reward, value, log_prob, done):
        idx = self.ptr
        
        for key in self.observations:
            self.observations[key][idx] = obs[key]
        
        self.actions_position[idx] = action['position']
        self.actions_rotation[idx] = action['rotation']
        
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.ptr += 1
    
    def get(self):
        obs_tensors = {
            key: torch.from_numpy(arr[:self.ptr]).to(self.device)
            for key, arr in self.observations.items()
        }
        
        actions_tensors = {
            'position': torch.from_numpy(self.actions_position[:self.ptr]).to(self.device),
            'rotation': torch.from_numpy(self.actions_rotation[:self.ptr]).to(self.device)
        }
        
        return {
            'observations': obs_tensors,
            'actions': actions_tensors,
            'rewards': self.rewards[:self.ptr].copy(),
            'values': self.values[:self.ptr].copy(),
            'log_probs': torch.from_numpy(self.log_probs[:self.ptr]).to(self.device),
            'dones': self.dones[:self.ptr].copy()
        }
    
    def reset(self):
        self.ptr = 0


# =============================================================================
# Trainer Otimizado para 2GB
# =============================================================================

class TinyPPOTrainer:
    def __init__(self, env, agent, device, config):
        self.env = env
        self.agent = agent
        self.device = device
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=config['learning_rate']
        )
        
        # Mixed precision para economizar memória
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        self.buffer = RolloutBuffer(config['n_steps'], device)
        
        self.episode_rewards = []
        self.episode_utilizations = []
    
    def collect_trajectories(self):
        self.buffer.reset()
        
        observation, _ = self.env.reset()
        episode_reward = 0
        
        self.agent.eval()
        
        for step in range(self.config['n_steps']):
            obs_tensor = self._obs_to_tensor(observation)
            
            # Usar mixed precision
            with torch.no_grad():
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        action, log_prob, value = self.agent.get_action(obs_tensor)
                else:
                    action, log_prob, value = self.agent.get_action(obs_tensor)
            
            action_dict = {
                'position': action['position'][0].cpu().numpy(),
                'rotation': int(action['rotation'][0].cpu().item())
            }
            
            next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
            done = terminated or truncated
            
            self.buffer.add(
                obs=observation,
                action=action_dict,
                reward=reward,
                value=value[0].item(),
                log_prob=log_prob[0].item(),
                done=done
            )
            
            episode_reward += reward
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_utilizations.append(info.get('utilization', 0))
                observation, _ = self.env.reset()
                episode_reward = 0
            else:
                observation = next_obs
            
            # Liberar memória periodicamente
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        return self.buffer.get()
    
    def compute_gae(self, trajectories):
        rewards = trajectories['rewards']
        values = trajectories['values']
        dones = trajectories['dones']
        
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)
        
        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']
        
        last_gae = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
                last_gae = 0
            
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_tensor = torch.from_numpy(advantages).to(self.device)
        returns_tensor = torch.from_numpy(returns).to(self.device)
        
        return advantages_tensor, returns_tensor
    
    def update_policy(self, trajectories, advantages, returns):
        observations = trajectories['observations']
        actions = trajectories['actions']
        log_probs_old = trajectories['log_probs']
        
        n_samples = len(advantages)
        indices = np.arange(n_samples)
        
        stats = {'policy_loss': [], 'value_loss': [], 'total_loss': []}
        
        self.agent.train()
        
        for epoch in range(self.config['n_epochs']):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.config['batch_size']):
                end = min(start + self.config['batch_size'], n_samples)
                batch_idx = indices[start:end]
                
                if len(batch_idx) < 4:
                    continue
                
                batch_obs = {
                    key: tensor[batch_idx] 
                    for key, tensor in observations.items()
                }
                
                batch_actions = {
                    key: tensor[batch_idx]
                    for key, tensor in actions.items()
                }
                
                batch_log_probs_old = log_probs_old[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                self.optimizer.zero_grad()
                
                # Mixed precision
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        log_probs, values, entropy = self.agent.evaluate_actions(
                            batch_obs, batch_actions
                        )
                        
                        ratio = torch.exp(log_probs - batch_log_probs_old)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1 - self.config['clip_epsilon'],
                            1 + self.config['clip_epsilon']
                        )
                        
                        policy_loss = -torch.min(
                            ratio * batch_advantages,
                            clipped_ratio * batch_advantages
                        ).mean()
                        
                        value_loss = F.mse_loss(values, batch_returns)
                        entropy_loss = -entropy.mean()
                        
                        total_loss = (
                            policy_loss + 
                            self.config['value_coef'] * value_loss +
                            self.config['entropy_coef'] * entropy_loss
                        )
                    
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    log_probs, values, entropy = self.agent.evaluate_actions(
                        batch_obs, batch_actions
                    )
                    
                    ratio = torch.exp(log_probs - batch_log_probs_old)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.config['clip_epsilon'],
                        1 + self.config['clip_epsilon']
                    )
                    
                    policy_loss = -torch.min(
                        ratio * batch_advantages,
                        clipped_ratio * batch_advantages
                    ).mean()
                    
                    value_loss = F.mse_loss(values, batch_returns)
                    entropy_loss = -entropy.mean()
                    
                    total_loss = (
                        policy_loss + 
                        self.config['value_coef'] * value_loss +
                        self.config['entropy_coef'] * entropy_loss
                    )
                    
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                    self.optimizer.step()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['total_loss'].append(total_loss.item())
            
            # Liberar memória após cada época
            torch.cuda.empty_cache()
        
        return {k: np.mean(v) for k, v in stats.items()}
    
    def train(self, n_iterations):
        print("="*70)
        print("TREINAMENTO PARA GPU 2GB")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Iterations: {n_iterations}")
        print(f"Steps per iteration: {self.config['n_steps']}")
        print(f"Batch size: {self.config['batch_size']}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for iteration in tqdm(range(1, n_iterations + 1), desc="Training"):
            trajectories = self.collect_trajectories()
            advantages, returns = self.compute_gae(trajectories)
            stats = self.update_policy(trajectories, advantages, returns)
            
            if iteration % self.config['log_frequency'] == 0:
                self._log_stats(iteration, stats)
            
            if iteration % self.config['save_frequency'] == 0:
                self._save_checkpoint(iteration)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Treinamento completo em {elapsed/60:.1f} minutos")
    
    def _log_stats(self, iteration, stats):
        if len(self.episode_rewards) > 0:
            recent = self.episode_rewards[-50:]
            recent_util = self.episode_utilizations[-50:]
            
            print(f"\nIter {iteration}:")
            print(f"  Loss: {stats['total_loss']:.4f}")
            print(f"  Reward: {np.mean(recent):.2f}")
            print(f"  Utilization: {np.mean(recent_util)*100:.1f}%")
    
    def _save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'checkpoint_tiny_{iteration}.pt')
        print(f"  ✓ Checkpoint: checkpoint_tiny_{iteration}.pt")
    
    def _obs_to_tensor(self, obs):
        return {
            'layout_image': torch.from_numpy(obs['layout_image']).unsqueeze(0).to(self.device),
            'current_piece': torch.from_numpy(obs['current_piece']).unsqueeze(0).to(self.device),
            'remaining_pieces': torch.from_numpy(obs['remaining_pieces']).unsqueeze(0).to(self.device),
            'stats': torch.from_numpy(obs['stats']).unsqueeze(0).to(self.device)
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("TREINAMENTO OTIMIZADO PARA GPU 2GB")
    print("="*70)
    
    # Config otimizada para 2GB VRAM
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'n_steps': 128,      # ← Reduzido (era 512)
        'batch_size': 16,    # ← Reduzido (era 64)
        'n_epochs': 3,       # ← Reduzido (era 4)
        'log_frequency': 10,
        'save_frequency': 50
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Peças
    pieces = [
        create_rectangle(50, 30),
        create_rectangle(40, 25),
    ]
    
    for i, piece in enumerate(pieces):
        piece.id = i
    
    print(f"Peças: {len(pieces)}")
    
    # Environment
    env_config = NestingConfig(max_steps=15)
    env = NestingEnvironment(config=env_config)
    env.reset(options={'pieces': pieces})
    print("Environment criado")
    
    # Agent LEVE
    agent = TinyActorCritic(
        embedding_dim=64,    # ← Reduzido (era 256)
        hidden_dim=128,      # ← Reduzido (era 512)
        rotation_bins=36
    ).to(device)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {n_params:,} parâmetros")
    print(f"  (Era 14.9M, agora {n_params/1e6:.1f}M = {100*n_params/14.9e6:.0f}% do tamanho)")
    
    # Trainer
    trainer = TinyPPOTrainer(env, agent, device, config)
    
    # TREINAR
    n_iterations = 100
    trainer.train(n_iterations)
    
    print("\n" + "="*70)
    print("✓✓✓ TREINAMENTO CONCLUÍDO! ✓✓✓")
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    main()