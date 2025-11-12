# """
# Script de Treinamento do Agent PPO com CNN
# ============================================

# Treina o agent de otimização de layout usando:
# - CNN para processar estados visuais
# - PPO para aprendizado por reforço
# - Dataset de layouts reais
# """

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn.encoder import LayoutCNNEncoder
from src.rl.ppo_agent import PPOAgent
from src.rl.layout_env import LayoutEnvironment
from src.utils.visualization import visualize_layout, plot_training_metrics


class TrainingConfig:
    #"""Configurações de treinamento"""
    # Treinamento
    num_episodes = 1000
    max_steps_per_episode = 50
    save_interval = 50  # Salvar checkpoint a cada N episódios
    
    # PPO
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    learning_rate = 3e-4
    
    # CNN
    embedding_dim = 256
    
    # Ambiente
    num_pieces = 10
    sheet_size = (1000, 1000)
    
    # Paths
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    
    def __init__(self):
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Timestamp para esta sessão
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.results_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)


class Trainer:
    #"""Gerenciador de treinamento"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Iniciando Treinamento")
        print(f"📱 Device: {self.device}")
        print(f"📁 Sessão: {config.session_id}")
        
        # Inicializar componentes
        self.cnn = LayoutCNNEncoder(embedding_dim=config.embedding_dim).to(self.device)
        self.env = LayoutEnvironment(
            num_pieces=config.num_pieces,
            sheet_size=config.sheet_size
        )
        self.agent = PPOAgent(
            state_dim=config.embedding_dim,
            action_dim=self.env.action_space.n,
            gamma=config.gamma,
            lambda_gae=config.lambda_gae,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            learning_rate=config.learning_rate
        )
        
        # Métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.utilization_rates = []
        self.policy_losses = []
        self.value_losses = []
        
    def process_visual_state(self, visual_state: np.ndarray) -> torch.Tensor:
        # """
        # Processa estado visual através da CNN
        
        # Args:
        #     visual_state: Array numpy (6, 256, 256)
            
        # Returns:
        #     embedding: Tensor (embedding_dim,)
        # """
        # Converter para tensor e adicionar batch dimension
        state_tensor = torch.FloatTensor(visual_state).unsqueeze(0).to(self.device)
        
        # Processar através da CNN
        with torch.no_grad():
            embedding, _ = self.cnn(state_tensor)
        
        # Remover batch dimension e retornar no CPU
        return embedding.squeeze(0).cpu()
    
    def train_episode(self, episode: int) -> dict:
        # """
        # Treina um episódio completo
        
        # Returns:
        #     Métricas do episódio
        # """
        visual_state = self.env.reset()
        state = self.process_visual_state(visual_state)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Coletar experiências
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        while not done and episode_length < self.config.max_steps_per_episode:
            # Agent escolhe ação
            action, log_prob, value = self.agent.select_action(state)
            
            # Executar ação no ambiente
            next_visual_state, reward, done, info = self.env.step(action)
            next_state = self.process_visual_state(next_visual_state)
            
            # Armazenar experiência
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            # Atualizar
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Treinar agent com experiências coletadas
        policy_loss, value_loss = self.agent.update(
            states, actions, rewards, log_probs, values
        )
        
        # Calcular utilização final
        utilization = info.get('utilization', 0.0)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'utilization': utilization,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def train(self):
        #"""Loop principal de treinamento"""
        print("\n" + "="*60)
        print("🎮 INICIANDO TREINAMENTO")
        print("="*60)
        
        best_reward = -float('inf')
        
        for episode in range(1, self.config.num_episodes + 1):
            # Treinar episódio
            metrics = self.train_episode(episode)
            
            # Armazenar métricas
            self.episode_rewards.append(metrics['reward'])
            self.episode_lengths.append(metrics['length'])
            self.utilization_rates.append(metrics['utilization'])
            self.policy_losses.append(metrics['policy_loss'])
            self.value_losses.append(metrics['value_loss'])
            
            # Log a cada 10 episódios
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_util = np.mean(self.utilization_rates[-10:])
                
                print(f"\n📊 Episódio {episode}/{self.config.num_episodes}")
                print(f"   Reward médio (últimos 10): {avg_reward:.2f}")
                print(f"   Utilização média: {avg_util:.1%}")
                print(f"   Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"   Value Loss: {metrics['value_loss']:.4f}")
            
            # Salvar checkpoint
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode, metrics['reward'])
                
                # Se for o melhor modelo, salvar separadamente
                if metrics['reward'] > best_reward:
                    best_reward = metrics['reward']
                    self.save_checkpoint(episode, metrics['reward'], best=True)
                    print(f"   ⭐ Novo melhor modelo! Reward: {metrics['reward']:.2f}")
        
        print("\n" + "="*60)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("="*60)
        
        # Salvar métricas finais e gerar visualizações
        self.save_final_results()
    
    def save_checkpoint(self, episode: int, reward: float, best: bool = False):
        #"""Salvar checkpoint do modelo"""
        if best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_ep{episode}.pt"
        
        checkpoint_path = self.config.checkpoint_dir / filename
        
        torch.save({
            'episode': episode,
            'cnn_state_dict': self.cnn.state_dict(),
            'agent_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'reward': reward,
            'config': self.config
        }, checkpoint_path)
        
        if not best:
            print(f"   💾 Checkpoint salvo: {filename}")
    
    def save_final_results(self):
        #"""Salvar resultados finais e gerar visualizações"""
        print("\n📈 Gerando visualizações...")
        
        # Plot 1: Rewards ao longo do tempo
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards, alpha=0.3, label='Reward')
        plt.plot(self._moving_average(self.episode_rewards, 50), 
                linewidth=2, label='Média (50 eps)')
        plt.xlabel('Episódio')
        plt.ylabel('Reward')
        plt.title('Reward ao Longo do Treinamento')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Taxa de utilização
        plt.subplot(1, 3, 2)
        plt.plot(self.utilization_rates, alpha=0.3, label='Utilização')
        plt.plot(self._moving_average(self.utilization_rates, 50),
                linewidth=2, label='Média (50 eps)')
        plt.xlabel('Episódio')
        plt.ylabel('Utilização (%)')
        plt.title('Taxa de Utilização da Chapa')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Losses
        plt.subplot(1, 3, 3)
        plt.plot(self.policy_losses, alpha=0.5, label='Policy Loss')
        plt.plot(self.value_losses, alpha=0.5, label='Value Loss')
        plt.xlabel('Episódio')
        plt.ylabel('Loss')
        plt.title('Losses durante Treinamento')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.session_dir / 'training_metrics.png', dpi=150)
        print(f"   ✓ Gráficos salvos em: {self.config.session_dir / 'training_metrics.png'}")
        
        # Salvar métricas em arquivo
        metrics_file = self.config.session_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write(f"Sessão de Treinamento: {self.config.session_id}\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"Configuração:\n")
            f.write(f"  Episódios: {self.config.num_episodes}\n")
            f.write(f"  Learning Rate: {self.config.learning_rate}\n")
            f.write(f"  Gamma: {self.config.gamma}\n")
            f.write(f"  Clip Epsilon: {self.config.clip_epsilon}\n\n")
            f.write(f"Resultados Finais:\n")
            f.write(f"  Reward médio (últimos 100): {np.mean(self.episode_rewards[-100:]):.2f}\n")
            f.write(f"  Melhor reward: {max(self.episode_rewards):.2f}\n")
            f.write(f"  Utilização média (últimos 100): {np.mean(self.utilization_rates[-100:]):.1%}\n")
            f.write(f"  Melhor utilização: {max(self.utilization_rates):.1%}\n")
        
        print(f"   ✓ Métricas salvas em: {metrics_file}")
        print(f"\n✨ Resultados completos em: {self.config.session_dir}")
    
    @staticmethod
    def _moving_average(data, window):
        #"""Calcular média móvel"""
        return np.convolve(data, np.ones(window)/window, mode='valid')


def main():
    #"""Função principal"""
    # Configuração
    config = TrainingConfig()
    
    # Criar trainer
    trainer = Trainer(config)
    
    # Treinar
    trainer.train()
    
    print("\n🎉 Treinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()