# """
# Use Trained Model - Usa modelo treinado para fazer nesting de peças reais
# Carrega automaticamente o checkpoint mais recente
# """
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional
from checkpoint_manager import CheckpointManager


class NestingPredictor:
    #"""Classe para usar modelo treinado em produção"""
    
    def __init__(self, checkpoint_dir: str = "scripts", device: str = None):
        # """
        # Inicializa o preditor
        
        # Args:
        #     checkpoint_dir: Diretório com checkpoints
        #     device: 'cpu', 'cuda', ou None (auto-detecta)
        # """
        # Auto-detecta device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"🖥️  Usando device: {device}")
        
        # Carrega checkpoint
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.checkpoint = self.checkpoint_manager.load_checkpoint(device=device)
        
        if self.checkpoint is None:
            raise RuntimeError("Nenhum checkpoint encontrado! Execute o treinamento primeiro.")
        
        self.actor = None
        self.env = None
        
    def setup_model(self, obs_shape: tuple, n_actions: int):
        # """
        # Configura o modelo com as dimensões corretas
        
        # Args:
        #     obs_shape: Shape da observação visual (C, H, W)
        #     n_actions: Número de ações
        # """
        import torch.nn as nn
        
        class SimpleActor(nn.Module):
            def __init__(self, obs_channels, n_actions):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(obs_channels, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.fc = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, n_actions),
                    nn.Tanh()
                )
                
            def forward(self, x):
                if isinstance(x, dict):
                    x = x['visual']
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        self.actor = SimpleActor(obs_shape[0], n_actions).to(self.device)
        
        # Carrega pesos
        if 'actor_state_dict' in self.checkpoint:
            self.actor.load_state_dict(self.checkpoint['actor_state_dict'])
            print("✓ Modelo carregado com sucesso")
        else:
            print("⚠️  Checkpoint sem pesos do actor - usando modelo aleatório")
        
        self.actor.eval()
    
    def nest_pieces(self, pieces: List, sheet_width: float, sheet_height: float,
                   verbose: bool = True) -> Dict[str, Any]:
        # """
        # Faz nesting de peças
        
        # Args:
        #     pieces: Lista de polígonos (objetos Polygon)
        #     sheet_width: Largura da chapa
        #     sheet_height: Altura da chapa
        #     verbose: Se True, mostra progresso
            
        # Returns:
        #     Dicionário com resultados do nesting
        # """
        from src.environment.nesting_env_fixed import NestingEnv
        
        # Cria ambiente
        self.env = NestingEnv(
            pieces=pieces,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            render_mode=None
        )
        
        # Configura modelo se ainda não foi configurado
        if self.actor is None:
            obs_shape = self.env.observation_space['visual'].shape
            n_actions = self.env.action_space.shape[0]
            self.setup_model(obs_shape, n_actions)
        
        # Executa nesting
        obs, info = self.env.reset()
        placements = []
        total_reward = 0
        
        if verbose:
            print(f"\n🎯 Fazendo nesting de {len(pieces)} peças...")
            print("-" * 80)
        
        for step in range(len(pieces)):
            # Converte observação
            visual_obs = torch.FloatTensor(obs['visual']).unsqueeze(0).to(self.device)
            
            # Predição
            with torch.no_grad():
                action = self.actor({'visual': visual_obs}).cpu().squeeze(0).numpy()
            
            # Executa
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Registra posicionamento
            placement = {
                'piece_index': step,
                'x': float(action[0]),
                'y': float(action[1]),
                'rotation': float(action[2]),
                'reward': float(reward),
                'valid': info.get('valid_placement', True)
            }
            placements.append(placement)
            
            if verbose:
                status = "✓" if placement['valid'] else "✗"
                print(f"Peça {step + 1:2d}/{len(pieces)} {status}: "
                      f"pos=({action[0]:6.3f}, {action[1]:6.3f}), "
                      f"rot={action[2]:6.3f}, "
                      f"reward={reward:7.4f}")
            
            if terminated or truncated:
                break
        
        if verbose:
            print("-" * 80)
        
        # Resultado final
        result = {
            'placements': placements,
            'total_reward': float(total_reward),
            'utilization': float(info.get('utilization', 0)),
            'pieces_placed': info.get('pieces_placed', 0),
            'total_pieces': len(pieces),
            'success_rate': info.get('pieces_placed', 0) / len(pieces) if pieces else 0,
            'sheet_width': sheet_width,
            'sheet_height': sheet_height,
            'checkpoint_info': {
                'epoch': self.checkpoint.get('epoch', 'unknown'),
                'iteration': self.checkpoint.get('iteration', 'unknown'),
                'training_reward': self.checkpoint.get('avg_reward', 'unknown'),
                'training_utilization': self.checkpoint.get('avg_utilization', 'unknown')
            }
        }
        
        return result
    
    def save_result(self, result: Dict[str, Any], output_file: str):
        #"""Salva resultado em JSON"""
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Resultado salvo em: {output_file}")


def criar_pecas_teste():
    #"""Cria conjunto de peças para teste"""
    from src.environment.nesting_env_fixed import Polygon
    
    pecas = [
        # Retângulos variados
        Polygon([(0, 0), (120, 0), (120, 60), (0, 60)]),
        Polygon([(0, 0), (100, 0), (100, 80), (0, 80)]),
        Polygon([(0, 0), (80, 0), (80, 50), (0, 50)]),
        Polygon([(0, 0), (90, 0), (90, 70), (0, 70)]),
        Polygon([(0, 0), (110, 0), (110, 55), (0, 55)]),
        
        # Formas em L
        Polygon([(0, 0), (60, 0), (60, 30), (30, 30), (30, 60), (0, 60)]),
        Polygon([(0, 0), (50, 0), (50, 25), (25, 25), (25, 50), (0, 50)]),
        
        # Triângulos
        Polygon([(0, 0), (70, 0), (35, 60)]),
        Polygon([(0, 0), (60, 0), (30, 50)]),
    ]
    
    return pecas


def demonstracao_completa():
    #"""Demonstração completa do uso do modelo"""
    
    print("=" * 80)
    print("USO DE MODELO TREINADO - Nesting Automático")
    print("=" * 80)
    
    try:
        # 1. Inicializa preditor (carrega checkpoint automaticamente)
        print("\n📥 Inicializando preditor...")
        predictor = NestingPredictor(checkpoint_dir="scripts")
        
        # 2. Cria peças de teste
        print("\n📦 Criando peças de teste...")
        pecas = criar_pecas_teste()
        print(f"   Criadas {len(pecas)} peças diversas")
        
        # 3. Executa nesting
        result = predictor.nest_pieces(
            pieces=pecas,
            sheet_width=600,
            sheet_height=500,
            verbose=True
        )
        
        # 4. Mostra resultado
        print("\n" + "=" * 80)
        print("📊 RESULTADO DO NESTING")
        print("=" * 80)
        print(f"Peças posicionadas: {result['pieces_placed']}/{result['total_pieces']}")
        print(f"Taxa de sucesso: {result['success_rate']:.1%}")
        print(f"Utilização da chapa: {result['utilization']:.2%}")
        print(f"Recompensa total: {result['total_reward']:.4f}")
        
        print(f"\nDimensões da chapa: {result['sheet_width']} x {result['sheet_height']}")
        print(f"Área total: {result['sheet_width'] * result['sheet_height']:.0f}")
        
        # 5. Informações do checkpoint usado
        ckpt = result['checkpoint_info']
        print(f"\n🔖 Checkpoint usado:")
        print(f"   Época: {ckpt['epoch']}")
        if ckpt['training_reward'] != 'unknown':
            print(f"   Recompensa no treino: {ckpt['training_reward']:.4f}")
        if ckpt['training_utilization'] != 'unknown':
            print(f"   Utilização no treino: {ckpt['training_utilization']:.2%}")
        
        # 6. Salva resultado
        output_file = "/mnt/user-data/outputs/nesting_result.json"
        predictor.save_result(result, output_file)
        
        print("\n✅ Nesting concluído com sucesso!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


def exemplo_producao():
    #"""Exemplo de uso em produção com peças customizadas"""
    
    print("\n" + "=" * 80)
    print("EXEMPLO: Uso em Produção")
    print("=" * 80)
    
    print("""
    Para usar em produção com suas próprias peças:
    
    1. Carregue suas peças de arquivo (JSON, DXF, SVG, etc):
       
       from nesting_env import Polygon
       import json
       
       # Exemplo: Carregando de JSON
       with open('pecas.json', 'r') as f:
           data = json.load(f)
       
       pecas = [Polygon(p['vertices']) for p in data['pecas']]
    
    2. Inicialize o preditor:
       
       predictor = NestingPredictor()
    
    3. Execute o nesting:
       
       result = predictor.nest_pieces(
           pieces=pecas,
           sheet_width=1000,
           sheet_height=800,
           verbose=True
       )
    
    4. Salve os resultados:
       
       predictor.save_result(result, 'resultado.json')
    
    5. Use os resultados para corte:
       
       for placement in result['placements']:
           if placement['valid']:
               print(f"Peça {placement['piece_index']}: "
                     f"x={placement['x']}, y={placement['y']}, "
                     f"rotação={placement['rotation']}")
    """)


if __name__ == "__main__":
    # Executa demonstração completa
    resultado = demonstracao_completa()
    
    # Mostra exemplo de uso em produção
    if resultado:
        exemplo_producao()