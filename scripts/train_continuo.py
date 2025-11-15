# """
# train_continuo.py

# Sistema de treinamento CONTINUADO
# - Carrega checkpoint anterior
# - Continua aprendendo
# - Cada execução melhora o modelo
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
import argparse
import glob

from src.environment.nesting_env import NestingEnvironment, NestingConfig
from src.geometry.polygon import create_rectangle, create_random_polygon

# Importar modelo do train_2gb_gpu.py
from train_2gb_gpu import (
    TinyActorCritic, 
    RolloutBuffer, 
    TinyPPOTrainer
)


# =============================================================================
# Sistema de Treinamento Continuado
# =============================================================================

class ContinuousTrainer(TinyPPOTrainer):
    #"""Trainer que continua de onde parou"""
    
    def __init__(self, env, agent, device, config, start_iteration=0):
        super().__init__(env, agent, device, config)
        self.start_iteration = start_iteration
        
        # Histórico de performance
        self.history = {
            'iterations': [],
            'rewards': [],
            'utilizations': [],
            'losses': []
        }
    
    def train(self, n_iterations):
        print("="*70)
        print("TREINAMENTO CONTINUADO")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Iteração inicial: {self.start_iteration}")
        print(f"Iterações a treinar: {n_iterations}")
        print(f"Iteração final: {self.start_iteration + n_iterations}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for iteration in tqdm(range(1, n_iterations + 1), desc="Training"):
            current_iter = self.start_iteration + iteration
            
            trajectories = self.collect_trajectories()
            advantages, returns = self.compute_gae(trajectories)
            stats = self.update_policy(trajectories, advantages, returns)
            
            # Salvar no histórico
            self.history['iterations'].append(current_iter)
            self.history['losses'].append(stats['total_loss'])
            if len(self.episode_rewards) > 0:
                self.history['rewards'].append(np.mean(self.episode_rewards[-50:]))
                self.history['utilizations'].append(np.mean(self.episode_utilizations[-50:]))
            
            # Logging
            if iteration % self.config['log_frequency'] == 0:
                self._log_stats(current_iter, stats)
            
            # Salvar checkpoint
            if iteration % self.config['save_frequency'] == 0:
                self._save_checkpoint(current_iter)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Treinamento completo em {elapsed/60:.1f} minutos")
        
        # Salvar histórico
        self._save_history()
    
    def _save_checkpoint(self, iteration):
        #"""Salva checkpoint com informações extras"""
        checkpoint = {
            'iteration': iteration,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'episode_rewards': self.episode_rewards[-100:],
            'episode_utilizations': self.episode_utilizations[-100:]
        }
        
        filepath = f'checkpoint_continuous_{iteration}.pt'
        torch.save(checkpoint, filepath)
        print(f"  ✓ Checkpoint: {filepath}")
    
    def _save_history(self):
        #"""Salva histórico de treinamento"""
        import json
        
        history_data = {
            'iterations': self.history['iterations'],
            'rewards': self.history['rewards'],
            'utilizations': self.history['utilizations'],
            'losses': self.history['losses']
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"\n✓ Histórico salvo: training_history.json")


# =============================================================================
# Funções Auxiliares
# =============================================================================

def find_latest_checkpoint():
    #"""Encontra o checkpoint mais recente"""
    checkpoints = glob.glob('checkpoint_continuous_*.pt')
    
    if not checkpoints:
        return None
    
    # Extrair números das iterações
    iterations = []
    for cp in checkpoints:
        try:
            iter_num = int(cp.split('_')[-1].replace('.pt', ''))
            iterations.append((iter_num, cp))
        except:
            continue
    
    if not iterations:
        return None
    
    # Retornar o mais recente
    latest = max(iterations, key=lambda x: x[0])
    return latest[1]


def load_checkpoint(checkpoint_path, device):
    #"""Carrega checkpoint e retorna modelo e informações"""
    print(f"\n{'='*70}")
    print(f"CARREGANDO CHECKPOINT")
    print(f"{'='*70}")
    print(f"Arquivo: {checkpoint_path}")
    
    # Fix para PyTorch 2.6+
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Criar modelo
    agent = TinyActorCritic(
        embedding_dim=64,
        hidden_dim=128,
        rotation_bins=36
    ).to(device)
    
    # Carregar pesos
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    # Informações
    iteration = checkpoint.get('iteration', 0)
    config = checkpoint.get('config', None)
    history = checkpoint.get('history', None)
    
    print(f"✓ Modelo carregado")
    print(f"✓ Iteração: {iteration}")
    
    if history and len(history.get('rewards', [])) > 0:
        recent_reward = np.mean(history['rewards'][-10:])
        recent_util = np.mean(history['utilizations'][-10:])
        print(f"✓ Performance anterior:")
        print(f"  - Reward médio: {recent_reward:.2f}")
        print(f"  - Utilização: {recent_util*100:.1f}%")
    
    print(f"{'='*70}\n")
    
    return agent, iteration, config, checkpoint


def plot_training_progress(history_file='training_history.json'):
    #"""Plota progresso do treinamento"""
    import json
    import matplotlib.pyplot as plt
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except:
        print("⚠️  Arquivo de histórico não encontrado")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward
    axes[0, 0].plot(history['iterations'], history['rewards'], 'b-', linewidth=2)
    axes[0, 0].set_title('Reward Médio', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Iteração')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Utilização
    if history['utilizations']:
        axes[0, 1].plot(history['iterations'], 
                       [u*100 for u in history['utilizations']], 
                       'g-', linewidth=2)
        axes[0, 1].set_title('Utilização', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Iteração')
        axes[0, 1].set_ylabel('Utilização (%)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 0].plot(history['iterations'], history['losses'], 'r-', linewidth=2)
    axes[1, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Iteração')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resumo
    axes[1, 1].axis('off')
    summary_text = f"""
    RESUMO DO TREINAMENTO
    
    Total de iterações: {history['iterations'][-1] if history['iterations'] else 0}
    
    Reward:
    • Inicial: {history['rewards'][0]:.2f if history['rewards'] else 0}
    • Final: {history['rewards'][-1]:.2f if history['rewards'] else 0}
    • Melhoria: {((history['rewards'][-1] / max(history['rewards'][0], 0.01) - 1) * 100):.1f}%
    
    Utilização:
    • Inicial: {history['utilizations'][0]*100:.1f}% if history['utilizations'] else 0
    • Final: {history['utilizations'][-1]*100:.1f}% if history['utilizations'] else 0
    
    Loss:
    • Inicial: {history['losses'][0]:.4f if history['losses'] else 0}
    • Final: {history['losses'][-1]:.4f if history['losses'] else 0}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                   verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: training_progress.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Treinamento Continuado')
    parser.add_argument('--continue', dest='checkpoint', type=str, default=None,
                       help='Checkpoint para continuar (default: busca o mais recente)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Número de iterações (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device: cuda ou cpu (default: cuda)')
    parser.add_argument('--plot', action='store_true',
                       help='Plotar progresso e sair')
    
    args = parser.parse_args()
    
    # Se só quer plotar
    if args.plot:
        plot_training_progress()
        return
    
    print("="*70)
    print("TREINAMENTO CONTINUADO - SISTEMA INTELIGENTE DE NESTING")
    print("="*70)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Config
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'n_steps': 128,
        'batch_size': 16,
        'n_epochs': 3,
        'log_frequency': 10,
        'save_frequency': 50
    }
    
    # Verificar se deve continuar de checkpoint
    checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        # Buscar checkpoint mais recente
        checkpoint_path = find_latest_checkpoint()
        
        if checkpoint_path:
            print(f"\n✓ Checkpoint encontrado: {checkpoint_path}")
            response = input("Continuar deste checkpoint? (s/n): ").strip().lower()
            
            if response != 's':
                checkpoint_path = None
    
    # Carregar ou criar modelo
    if checkpoint_path and Path(checkpoint_path).exists():
        agent, start_iteration, old_config, checkpoint = load_checkpoint(checkpoint_path, device)
        
        # Atualizar config se necessário
        if old_config:
            config.update(old_config)
        
        # Carregar otimizador também
        optimizer_state = checkpoint.get('optimizer_state_dict')
    else:
        print("\n" + "="*70)
        print("INICIANDO TREINAMENTO DO ZERO")
        print("="*70 + "\n")
        
        agent = TinyActorCritic(
            embedding_dim=64,
            hidden_dim=128,
            rotation_bins=36
        ).to(device)
        
        start_iteration = 0
        optimizer_state = None
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {n_params:,} parâmetros\n")
    
    # Peças
    pieces = [
        create_rectangle(50, 30),
        create_rectangle(40, 25),
        create_rectangle(60, 35),
    ]
    
    for i, piece in enumerate(pieces):
        piece.id = i
    
    # Environment
    env_config = NestingConfig(max_steps=20)
    env = NestingEnvironment(config=env_config)
    env.reset(options={'pieces': pieces})
    
    # Trainer
    trainer = ContinuousTrainer(env, agent, device, config, start_iteration)
    
    # Se tinha otimizador salvo, carregar
    if optimizer_state:
        trainer.optimizer.load_state_dict(optimizer_state)
        print("✓ Estado do otimizador carregado\n")
    
    # TREINAR
    trainer.train(args.iterations)
    
    # Plotar progresso
    print("\n" + "="*70)
    print("PLOTANDO PROGRESSO")
    print("="*70)
    
    try:
        plot_training_progress()
    except Exception as e:
        print(f"⚠️  Não foi possível plotar: {e}")
    
    print("\n" + "="*70)
    print("✓✓✓ TREINAMENTO CONCLUÍDO! ✓✓✓")
    print("="*70)
    print(f"""
Para continuar treinando:
  python train_continuo.py --iterations 100

Para ver progresso:
  python train_continuo.py --plot

Para usar o modelo:
  python use_trained_model.py
    """)
    
    env.close()


if __name__ == "__main__":
    main()