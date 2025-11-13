# """
# scripts/train_complete_system.py
# Script COMPLETO de treinamento do sistema de Nesting Inteligente
# Integra: Geometry + CNN + Environment + PPO + Curriculum
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import yaml
import argparse

# Importar módulos do sistema
from src.environment.nesting_env import NestingEnvironment, NestingConfig
from src.models.cnn.encoder import LayoutCNNEncoder
from src.training.curriculum import CurriculumScheduler
from src.geometry.polygon import create_rectangle, create_random_polygon

# =============================================================================
# ACTOR-CRITIC NETWORK
# =============================================================================

class ActorCritic(nn.Module):
    #"""Rede Actor-Critic para PPO com CNN real"""
    
    def __init__(self,
                 cnn_embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 rotation_bins: int = 36):
        super().__init__()
        
        # CNN Encoder (REAL)
        self.cnn = LayoutCNNEncoder(
            input_channels=6,
            embedding_dim=cnn_embedding_dim
        )
        
        # Input = CNN embedding + piece features + remaining + stats
        total_input = cnn_embedding_dim + 10 + 10 + 5
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor heads
        self.actor_position_mean = nn.Linear(hidden_dim, 2)
        self.actor_position_logstd = nn.Parameter(torch.zeros(2))
        self.actor_rotation = nn.Linear(hidden_dim, rotation_bins)
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, observation):
        #"""Forward pass"""
        # CNN
        cnn_embedding, _ = self.cnn(observation['layout_image'])
        
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
    
    def evaluate_actions(self, observations, actions):
        #"""Avalia ações para treinamento"""
        pos_mean, pos_logstd, rot_logits, values = self.forward(observations)
        
        pos_std = torch.exp(pos_logstd)
        
        # Position log prob
        pos_dist = torch.distributions.Normal(pos_mean, pos_std)
        pos_log_probs = pos_dist.log_prob(actions['position']).sum(dim=-1)
        
        # Rotation log prob
        rot_probs = torch.softmax(rot_logits, dim=-1)
        rot_log_probs = torch.log(
            rot_probs.gather(1, actions['rotation'].unsqueeze(-1)).squeeze(-1) + 1e-8
        )
        
        total_log_probs = pos_log_probs + rot_log_probs
        
        # Entropy
        pos_entropy = pos_dist.entropy().sum(dim=-1)
        rot_dist = torch.distributions.Categorical(probs=rot_probs)
        rot_entropy = rot_dist.entropy()
        
        total_entropy = pos_entropy + rot_entropy
        
        return total_log_probs, values.squeeze(-1), total_entropy


# =============================================================================
# PPO TRAINER
# =============================================================================

class PPOTrainer:
    #"""Treinador PPO completo"""
    
    def __init__(self, env, agent, curriculum, config, device):
        self.env = env
        self.agent = agent
        self.curriculum = curriculum
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            agent.parameters(),
            lr=config['learning_rate'],
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_decay_steps', 1000),
            gamma=config.get('lr_decay_gamma', 0.95)
        )
        
        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.global_step = 0
        self.iteration = 0
        
        # Stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_utilizations = []
        
        # Best model
        self.best_utilization = 0.0
    
    def train(self, n_iterations: int):
        #"""Loop principal de treinamento"""
        print("\n" + "="*70)
        print("🚀 INICIANDO TREINAMENTO PPO + CNN + CURRICULUM")
        print("="*70)
        print(f"Iterations: {n_iterations}")
        print(f"Steps per iteration: {self.config['n_steps']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Device: {self.device}")
        print(f"Agent parameters: {sum(p.numel() for p in self.agent.parameters()):,}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for iteration in range(1, n_iterations + 1):
            self.iteration = iteration
            
            # Coletar trajetórias
            trajectories, traj_stats = self._collect_trajectories()
            
            # Calcular vantagens
            advantages, returns = self._compute_gae(trajectories)
            
            # Treinar com PPO
            train_stats = self._update_policy(trajectories, advantages, returns)
            
            # Logging
            if iteration % self.config['log_frequency'] == 0:
                self._log_stats(iteration, train_stats, traj_stats)
            
            # Avaliação
            if iteration % self.config['eval_frequency'] == 0:
                eval_stats = self._evaluate()
                self._log_eval(iteration, eval_stats)
                
                # Salvar melhor modelo
                if eval_stats['utilization_mean'] > self.best_utilization:
                    self.best_utilization = eval_stats['utilization_mean']
                    self._save_checkpoint(iteration, is_best=True)
            
            # Salvar checkpoint
            if iteration % self.config['save_frequency'] == 0:
                self._save_checkpoint(iteration)
            
            # Learning rate decay
            self.lr_scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ TREINAMENTO COMPLETO!")
        print(f"   Tempo total: {elapsed/3600:.2f} horas")
        print(f"   Melhor utilização: {self.best_utilization*100:.2f}%")
        print("="*70 + "\n")
        
        self.writer.close()
    
    def _collect_trajectories(self):
        #"""Coleta trajetórias"""
        trajectories = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        stats = {
            'episodes_completed': 0,
            'avg_reward': 0,
            'avg_utilization': 0,
            'avg_length': 0
        }
        
        # Gerar peças com curriculum
        problem_config = self.curriculum.get_problem_config()
        pieces = self.curriculum.generate_pieces(problem_config)
        
        observation, _ = self.env.reset(options={'pieces': pieces})
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config['n_steps']):
            # Converter observação para tensors
            obs_tensors = self._obs_to_tensors(observation)
            
            # Selecionar ação
            with torch.no_grad():
                action, log_prob, value = self.agent.get_action(obs_tensors)
            
            # Executar no ambiente
            action_dict = {
                'position': action['position'][0].cpu().numpy(),
                'rotation': int(action['rotation'][0].cpu().item())
            }
            
            next_observation, reward, terminated, truncated, info = self.env.step(action_dict)
            done = terminated or truncated
            
            # Armazenar
            trajectories['observations'].append(observation)
            trajectories['actions'].append(action_dict)
            trajectories['rewards'].append(reward)
            trajectories['values'].append(value[0].item())
            trajectories['log_probs'].append(log_prob[0].item())
            trajectories['dones'].append(done)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Atualizar curriculum
                utilization = info.get('utilization', 0)
                self.curriculum.update(utilization)
                
                # Salvar stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_utilizations.append(utilization)
                
                stats['episodes_completed'] += 1
                
                # Reset para novo episódio
                problem_config = self.curriculum.get_problem_config()
                pieces = self.curriculum.generate_pieces(problem_config)
                observation, _ = self.env.reset(options={'pieces': pieces})
                episode_reward = 0
                episode_length = 0
            else:
                observation = next_observation
        
        # Calcular stats
        if stats['episodes_completed'] > 0:
            recent_rewards = self.episode_rewards[-stats['episodes_completed']:]
            recent_utils = self.episode_utilizations[-stats['episodes_completed']:]
            recent_lengths = self.episode_lengths[-stats['episodes_completed']:]
            
            stats['avg_reward'] = np.mean(recent_rewards)
            stats['avg_utilization'] = np.mean(recent_utils)
            stats['avg_length'] = np.mean(recent_lengths)
        
        return trajectories, stats
    
    def _compute_gae(self, trajectories):
        #"""Generalized Advantage Estimation"""
        rewards = np.array(trajectories['rewards'])
        values = np.array(trajectories['values'])
        dones = np.array(trajectories['dones'])
        
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        
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
        
        # Normalizar vantagens
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_policy(self, trajectories, advantages, returns):
        #"""Atualiza política usando PPO"""
        n_samples = len(trajectories['rewards'])
        indices = np.arange(n_samples)
        
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'clip_fraction': []
        }
        
        # Múltiplas épocas
        for epoch in range(self.config['n_epochs']):
            np.random.shuffle(indices)
            
            # Mini-batches
            for start in range(0, n_samples, self.config['batch_size']):
                end = min(start + self.config['batch_size'], n_samples)
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 4:  # Skip very small batches
                    continue
                
                # Preparar batch
                batch_obs = self._batch_observations(
                    [trajectories['observations'][i] for i in batch_indices]
                )
                batch_actions = self._batch_actions(
                    [trajectories['actions'][i] for i in batch_indices]
                )
                batch_log_probs_old = torch.tensor(
                    [trajectories['log_probs'][i] for i in batch_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                batch_advantages = torch.tensor(
                    advantages[batch_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                batch_returns = torch.tensor(
                    returns[batch_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Avaliar ações
                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # PPO loss
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
                
                # Value loss (clipped)
                value_pred_clipped = values
                value_loss = F.mse_loss(value_pred_clipped, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.config['value_coef'] * value_loss +
                    self.config['entropy_coef'] * entropy_loss
                )
                
                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                
                # Stats
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.config['clip_epsilon']).float().mean().item()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['total_loss'].append(total_loss.item())
                stats['clip_fraction'].append(clip_fraction)
                
                self.global_step += 1
        
        # Média das stats
        return {k: np.mean(v) if v else 0 for k, v in stats.items()}
    
    def _evaluate(self, n_episodes: int = 5):
        #"""Avalia política"""
        eval_rewards = []
        eval_utilizations = []
        eval_lengths = []
        
        for _ in range(n_episodes):
            # Gerar problema
            problem_config = self.curriculum.get_problem_config()
            pieces = self.curriculum.generate_pieces(problem_config)
            
            observation, _ = self.env.reset(options={'pieces': pieces})
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                obs_tensors = self._obs_to_tensors(observation)
                
                with torch.no_grad():
                    action, _, _ = self.agent.get_action(obs_tensors, deterministic=True)
                
                action_dict = {
                    'position': action['position'][0].cpu().numpy(),
                    'rotation': int(action['rotation'][0].cpu().item())
                }
                
                observation, reward, terminated, truncated, info = self.env.step(action_dict)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if episode_length >= 200:  # Timeout
                    break
            
            eval_rewards.append(episode_reward)
            eval_utilizations.append(info.get('utilization', 0))
            eval_lengths.append(episode_length)
        
        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'utilization_mean': np.mean(eval_utilizations),
            'utilization_std': np.std(eval_utilizations),
            'length_mean': np.mean(eval_lengths)
        }
    
    def _log_stats(self, iteration, train_stats, traj_stats):
        #"""Logging de estatísticas"""
        # Console
        print(f"\n[Iter {iteration:4d}] "
              f"Loss: {train_stats['total_loss']:.4f} | "
              f"Policy: {train_stats['policy_loss']:.4f} | "
              f"Value: {train_stats['value_loss']:.4f} | "
              f"Entropy: {train_stats['entropy']:.4f}")
        
        if traj_stats['episodes_completed'] > 0:
            print(f"           "
                  f"Episodes: {traj_stats['episodes_completed']} | "
                  f"Reward: {traj_stats['avg_reward']:.2f} | "
                  f"Util: {traj_stats['avg_utilization']*100:.1f}% | "
                  f"Len: {traj_stats['avg_length']:.1f}")
        
        # Curriculum stats
        curr_stats = self.curriculum.get_stats()
        print(f"           "
              f"Stage: {curr_stats['current_stage']}/{curr_stats['total_stages']-1} | "
              f"{curr_stats['stage_name']} | "
              f"Success: {curr_stats['success_rate']*100:.1f}%")
        
        # TensorBoard
        for key, value in train_stats.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        
        for key, value in traj_stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'collection/{key}', value, iteration)
        
        for key, value in curr_stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'curriculum/{key}', value, iteration)
        
        self.writer.add_scalar('train/learning_rate', 
                              self.optimizer.param_groups[0]['lr'], iteration)
    
    def _log_eval(self, iteration, eval_stats):
        #"""Logging de avaliação"""
        print(f"\n   [EVAL] Util: {eval_stats['utilization_mean']*100:.2f}% ± {eval_stats['utilization_std']*100:.2f}% | "
              f"Reward: {eval_stats['reward_mean']:.2f} ± {eval_stats['reward_std']:.2f}")
        
        for key, value in eval_stats.items():
            self.writer.add_scalar(f'eval/{key}', value, iteration)
    
    def _save_checkpoint(self, iteration, is_best=False):
        #"""Salva checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'curriculum_stage': self.curriculum.current_stage,
            'best_utilization': self.best_utilization,
            'config': self.config
        }
        
        if is_best:
            path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
            print(f"   ✅ Best model saved: {path}")
        else:
            path = checkpoint_dir / f'checkpoint_{iteration:05d}.pt'
            torch.save(checkpoint, path)
            print(f"   💾 Checkpoint saved: {path}")
    
    def _obs_to_tensors(self, observation):
        #"""Converte observação para tensors"""
        return {
            'layout_image': torch.from_numpy(observation['layout_image']).unsqueeze(0).to(self.device),
            'current_piece': torch.from_numpy(observation['current_piece']).unsqueeze(0).to(self.device),
            'remaining_pieces': torch.from_numpy(observation['remaining_pieces']).unsqueeze(0).to(self.device),
            'stats': torch.from_numpy(observation['stats']).unsqueeze(0).to(self.device)
        }
    
    def _batch_observations(self, observations):
        #"""Agrupa observações em batch"""
        return {
            'layout_image': torch.stack([
                torch.from_numpy(obs['layout_image']) for obs in observations
            ]).to(self.device),
            'current_piece': torch.stack([
                torch.from_numpy(obs['current_piece']) for obs in observations
            ]).to(self.device),
            'remaining_pieces': torch.stack([
                torch.from_numpy(obs['remaining_pieces']) for obs in observations
            ]).to(self.device),
            'stats': torch.stack([
                torch.from_numpy(obs['stats']) for obs in observations
            ]).to(self.device)
        }
    
    def _batch_actions(self, actions):
        #"""Agrupa ações em batch"""
        return {
            'position': torch.tensor(
                [a['position'] for a in actions],
                dtype=torch.float32,
                device=self.device
            ),
            'rotation': torch.tensor(
                [a['rotation'] for a in actions],
                dtype=torch.long,
                device=self.device
            )
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Intelligent Nesting System')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Config file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of training iterations')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Config
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'log_frequency': 10,
        'eval_frequency': 50,
        'save_frequency': 100,
        'log_dir': 'logs/ppo_nesting',
        'checkpoint_dir': 'checkpoints',
        'lr_decay_steps': 1000,
        'lr_decay_gamma': 0.95
    }
    
    # Environment
    env_config = NestingConfig(
        max_steps=100,
        container_width=1000,
        container_height=600,
        rotation_bins=36
    )
    env = NestingEnvironment(config=env_config)
    print(f"✅ Environment criado")
    
    # Agent
    agent = ActorCritic(
        cnn_embedding_dim=256,
        hidden_dim=512,
        rotation_bins=36
    ).to(device)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"✅ Agent criado: {n_params:,} parâmetros")
    
    # Curriculum
    curriculum_config = {
        'min_episodes_per_stage': 100
    }
    curriculum = CurriculumScheduler(curriculum_config)
    print(f"✅ Curriculum criado com {len(curriculum.stages)} estágios")
    
    # Trainer
    trainer = PPOTrainer(env, agent, curriculum, config, device)
    
    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Resumed from {args.resume}")
    
    # Train!
    trainer.train(n_iterations=args.iterations)
    
    print("\n🎉 Treinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()