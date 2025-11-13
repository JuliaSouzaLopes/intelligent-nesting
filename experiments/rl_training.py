import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cnn.encoder import LayoutCNNEncoder

from src.models.cnn.encoder import LayoutCNNEncoder
from src.environment.nesting_env import NestingEnvironment, NestingConfig
from src.training.curriculum import CurriculumScheduler

#Script completo de treinamento do sistema de nesting com PPO.
#Integra CNN, Environment e algoritmo de RL.

# =============================================================================
# PPO Agent
# =============================================================================

class ActorCritic(nn.Module):
    # """
    # Rede Actor-Critic para PPO.
    
    # Combina CNN encoder com cabeças de política e valor.
    # """
    
    def __init__(self,
                 cnn_embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 rotation_bins: int = 36):
        super().__init__()
        
        # CNN Encoder (assumindo já implementado)
        self.cnn = LayoutCNNEncoder(
            input_channels=6,
            embedding_dim=cnn_embedding_dim
        )
        
        # Placeholder para CNN
        self.cnn_embedding_dim = cnn_embedding_dim
        
        # Processar features adicionais (piece features + stats)
        # Total input: cnn_embedding + current_piece (10) + remaining (10) + stats (5)
        total_input_dim = cnn_embedding_dim + 10 + 10 + 5
        
        # Camadas compartilhadas
        self.shared_layers = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (política)
        # Output: mean e log_std para posição (contínua)
        #         logits para rotação (discreta)
        self.actor_position_mean = nn.Linear(hidden_dim, 2)  # (x, y)
        self.actor_position_logstd = nn.Parameter(torch.zeros(2))
        self.actor_rotation = nn.Linear(hidden_dim, rotation_bins)
        
        # Critic (valor)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> Tuple:
        # """
        # Forward pass.
        
        # Returns:
        #     position_mean, position_logstd, rotation_logits, value
        # """
        # Processar layout com CNN
        layout_image = observation['layout_image']
        
        # PLACEHOLDER: simular embedding CNN
        #batch_size = layout_image.shape[0]
        #cnn_embedding = torch.randn(batch_size, self.cnn_embedding_dim,
        #                           device=layout_image.device)
        
        cnn_embedding, heatmap = self.cnn(layout_image)  # Real implementation
        
        # Concatenar todas as features
        current_piece = observation['current_piece']
        remaining = observation['remaining_pieces']
        stats = observation['stats']
        
        combined = torch.cat([cnn_embedding, current_piece, remaining, stats], dim=1)
        
        # Camadas compartilhadas
        shared = self.shared_layers(combined)
        
        # Actor
        position_mean = torch.sigmoid(self.actor_position_mean(shared))  # [0, 1]
        position_logstd = self.actor_position_logstd.expand_as(position_mean)
        rotation_logits = self.actor_rotation(shared)
        
        # Critic
        value = self.critic(shared)
        
        return position_mean, position_logstd, rotation_logits, value
    
    def get_action(self, observation: Dict[str, torch.Tensor],
                  deterministic: bool = False):
        # """
        # Gera ação a partir da observação.
        
        # Returns:
        #     action, log_prob, value
        # """
        position_mean, position_logstd, rotation_logits, value = self.forward(observation)
        
        # Posição (distribuição Gaussiana)
        position_std = torch.exp(position_logstd)
        
        if deterministic:
            position = position_mean
            rotation = torch.argmax(rotation_logits, dim=-1)
        else:
            # Sample
            position_dist = torch.distributions.Normal(position_mean, position_std)
            position = position_dist.sample()
            position = torch.clamp(position, 0, 1)  # Garantir [0, 1]
            
            rotation_probs = torch.softmax(rotation_logits, dim=-1)
            rotation = torch.multinomial(rotation_probs, 1).squeeze(-1)
        
        # Calcular log probability
        position_dist = torch.distributions.Normal(position_mean, position_std)
        position_log_prob = position_dist.log_prob(position).sum(dim=-1)
        
        rotation_probs = torch.softmax(rotation_logits, dim=-1)
        rotation_log_prob = torch.log(rotation_probs.gather(1, rotation.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        total_log_prob = position_log_prob + rotation_log_prob
        
        return {
            'position': position,
            'rotation': rotation
        }, total_log_prob, value
    
    def evaluate_actions(self, observations: Dict[str, torch.Tensor],
                        actions: Dict[str, torch.Tensor]):
        # """
        # Avalia ações para treinamento.
        
        # Returns:
        #     log_probs, values, entropy
        # """
        position_mean, position_logstd, rotation_logits, values = self.forward(observations)
        
        position_std = torch.exp(position_logstd)
        
        # Position log prob
        position_dist = torch.distributions.Normal(position_mean, position_std)
        position_log_probs = position_dist.log_prob(actions['position']).sum(dim=-1)
        
        # Rotation log prob
        rotation_probs = torch.softmax(rotation_logits, dim=-1)
        rotation_log_probs = torch.log(
            rotation_probs.gather(1, actions['rotation'].unsqueeze(-1)).squeeze(-1) + 1e-8
        )
        
        total_log_probs = position_log_probs + rotation_log_probs
        
        # Entropy
        position_entropy = position_dist.entropy().sum(dim=-1)
        rotation_dist = torch.distributions.Categorical(probs=rotation_probs)
        rotation_entropy = rotation_dist.entropy()
        
        total_entropy = position_entropy + rotation_entropy
        
        return total_log_probs, values.squeeze(-1), total_entropy


# =============================================================================
# PPO Trainer
# =============================================================================

class PPOTrainer:
    #"""Treinador PPO para nesting"""
    
    def __init__(self,
                 env,
                 agent: ActorCritic,
                 config: dict):
        # """
        # Args:
        #     env: Gymnasium environment
        #     agent: Rede Actor-Critic
        #     config: Configuração de treinamento
        # """
        self.env = env
        self.agent = agent
        self.config = config
        self.curriculum = None  # Será injetado
        
        # Otimizador
        self.optimizer = optim.Adam(
            agent.parameters(),
            lr=config['learning_rate']
        )
        
        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.global_step = 0
        
        # Estatísticas
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_utilizations = []
    
    def train(self, n_iterations: int):
        # """
        # Loop principal de treinamento.
        
        # Args:
        #     n_iterations: Número de iterações
        # """
        print("="*70)
        print("INICIANDO TREINAMENTO PPO")
        print("="*70)
        print(f"Iterations: {n_iterations}")
        print(f"Steps per iteration: {self.config['n_steps']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"PPO epochs: {self.config['n_epochs']}")
        print("="*70)
        
        start_time = time.time()
        
        for iteration in range(1, n_iterations + 1):
            # Coletar trajetórias
            trajectories = self._collect_trajectories()
            
            # Calcular vantagens
            advantages, returns = self._compute_gae(trajectories)
            
            # Treinar com PPO
            stats = self._update_policy(trajectories, advantages, returns)
            
            # Atualizar curriculum
            if self.curriculum and len(self.episode_utilizations) > 0:
                last_util = self.episode_utilizations[-1]
                success = last_util > 0.3
                self.curriculum.record_episode(last_util, success)
                
                if self.curriculum.should_advance(min_episodes=50):
                    self.curriculum.advance_stage()
                    self.writer.add_scalar('curriculum/stage', 
                                        self.curriculum.current_stage.stage_id,
                                        iteration)
            
            # Logging
            if iteration % self.config['log_frequency'] == 0:
                self._log_stats(iteration, stats)
            
            # Avaliação
            if iteration % self.config['eval_frequency'] == 0:
                eval_stats = self._evaluate()
                self._log_eval(iteration, eval_stats)
            
            # Salvar checkpoint
            if iteration % self.config['save_frequency'] == 0:
                self._save_checkpoint(iteration)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Treinamento completo em {elapsed/3600:.2f} horas")
        
        self.writer.close()
    
    def _collect_trajectories(self, pieces=None):

        # Gerar problema do curriculum se disponível
        if self.curriculum and pieces is None:
            problem = self.curriculum.generate_problem()
            pieces = problem['pieces']
            # Atualizar tamanho do container
            self.env.config.container_width = problem['container_size'][0]
            self.env.config.container_height = problem['container_size'][1]
        
        # Reset com peças
        reset_options = {'pieces': pieces} if pieces else None
        observation, _ = self.env.reset(options=reset_options)

        #"""Coleta trajetórias interagindo com o ambiente"""
        trajectories = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        observation, _ = self.env.reset()
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
                # Episódio terminou
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_utilizations.append(info.get('utilization', 0))
                
                observation, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                observation = next_observation
        
        return trajectories
    
    def _compute_gae(self, trajectories: Dict) -> Tuple[np.ndarray, np.ndarray]:
        # """
        # Calcula Generalized Advantage Estimation.
        
        # Returns:
        #     advantages, returns
        # """
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
    
    def _update_policy(self, trajectories: Dict, 
                      advantages: np.ndarray,
                      returns: np.ndarray) -> Dict:
        # """Atualiza política usando PPO"""
        
        # Preparar dados
        n_samples = len(trajectories['rewards'])
        indices = np.arange(n_samples)
        
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
        
        # Múltiplas épocas
        for epoch in range(self.config['n_epochs']):
            np.random.shuffle(indices)
            
            # Mini-batches
            for start in range(0, n_samples, self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_indices = indices[start:end]
                
                if len(batch_indices) < self.config['batch_size']:
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
                    dtype=torch.float32
                )
                batch_advantages = torch.tensor(
                    advantages[batch_indices],
                    dtype=torch.float32
                )
                batch_returns = torch.tensor(
                    returns[batch_indices],
                    dtype=torch.float32
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
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.config['value_coef'] * value_loss +
                            self.config['entropy_coef'] * entropy_loss)
                
                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                
                # Stats
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['total_loss'].append(total_loss.item())
                
                self.global_step += 1
        
        # Média das stats
        return {k: np.mean(v) for k, v in stats.items()}
    
    def _evaluate(self, n_episodes: int = 5) -> Dict:
        #"""Avalia política atual"""
        eval_rewards = []
        eval_lengths = []
        eval_utilizations = []
        
        for _ in range(n_episodes):
            observation, _ = self.env.reset()
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
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_utilizations.append(info.get('utilization', 0))
        
        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'length_mean': np.mean(eval_lengths),
            'utilization_mean': np.mean(eval_utilizations),
            'utilization_std': np.std(eval_utilizations)
        }
    
    def _log_stats(self, iteration: int, stats: Dict):
        #"""Logging de estatísticas"""
        print(f"\nIteration {iteration}/{self.config['n_iterations']}")
        print(f"  Policy Loss: {stats['policy_loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  Total Loss: {stats['total_loss']:.4f}")
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            print(f"  Episode Reward (last 100): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
        
        # TensorBoard
        for key, value in stats.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        
        if self.episode_rewards:
            self.writer.add_scalar('train/episode_reward', np.mean(recent_rewards), iteration)
    
    def _log_eval(self, iteration: int, stats: Dict):
        #"""Logging de avaliação"""
        print(f"\n  [EVAL] Utilization: {stats['utilization_mean']*100:.2f}% ± {stats['utilization_std']*100:.2f}%")
        print(f"  [EVAL] Reward: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}")
        
        for key, value in stats.items():
            self.writer.add_scalar(f'eval/{key}', value, iteration)
    
    def _save_checkpoint(self, iteration: int):
        #"""Salva checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_{iteration}.pt'
        
        torch.save({
            'iteration': iteration,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        print(f"  ✓ Checkpoint salvo: {checkpoint_path}")
    
    def _obs_to_tensors(self, observation: Dict) -> Dict:
        #"""Converte observação para tensors"""
        return {
            'layout_image': torch.tensor(observation['layout_image'], dtype=torch.float32).unsqueeze(0),
            'current_piece': torch.tensor(observation['current_piece'], dtype=torch.float32).unsqueeze(0),
            'remaining_pieces': torch.tensor(observation['remaining_pieces'], dtype=torch.float32).unsqueeze(0),
            'stats': torch.tensor(observation['stats'], dtype=torch.float32).unsqueeze(0)
        }
    
    def _batch_observations(self, observations: List[Dict]) -> Dict:
        #"""Agrupa observações em batch"""
        return {
            'layout_image': torch.stack([torch.tensor(obs['layout_image'], dtype=torch.float32) for obs in observations]),
            'current_piece': torch.stack([torch.tensor(obs['current_piece'], dtype=torch.float32) for obs in observations]),
            'remaining_pieces': torch.stack([torch.tensor(obs['remaining_pieces'], dtype=torch.float32) for obs in observations]),
            'stats': torch.stack([torch.tensor(obs['stats'], dtype=torch.float32) for obs in observations])
        }
    
    def _batch_actions(self, actions: List[Dict]) -> Dict:
        #"""Agrupa ações em batch"""
        return {
            'position': torch.tensor([a['position'] for a in actions], dtype=torch.float32),
            'rotation': torch.tensor([a['rotation'] for a in actions], dtype=torch.long)
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Carregar config (placeholder)
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
        'n_iterations': 1000,
        'log_frequency': 10,
        'eval_frequency': 50,
        'save_frequency': 100,
        'log_dir': 'runs/ppo_nesting',
        'checkpoint_dir': 'checkpoints'
    }
    
    print("Configuração carregada")
    
    # Criar ambiente (placeholder - usar implementação real)
    env_config = NestingConfig()

    env = NestingEnvironment(config=env_config)
    
    # Para teste, usar ambiente dummy
    print("Ambiente: SIMULADO (implementar NestingEnvironment real)")
    
    # Criar agente
    agent = ActorCritic(
        cnn_embedding_dim=256,
        hidden_dim=512,
        rotation_bins=36
    )

    # Criar curriculum
    curriculum = CurriculumScheduler(
        base_container_size=(1000, 600),
        auto_advance=True,
        advancement_threshold=0.6
    )
    print(f"✓ Curriculum: {curriculum.current_stage.name}")
    
    # Criar trainer
    trainer = PPOTrainer(env, agent, config)
    
    # Injetar curriculum no trainer
    trainer.curriculum = curriculum
    
    # Treinar
    trainer.train(config['n_iterations'])
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    print(f"Agente criado e movido para {device}")
    print(f"Parâmetros: {sum(p.numel() for p in agent.parameters()):,}")
    
    print("\n✓ Script de treinamento implementado!")
    print("  Para treinar de verdade, descomentar código acima e implementar módulos reais")


    if __name__ == "__main__":
        main()