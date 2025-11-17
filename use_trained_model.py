# """
# Script para usar modelo treinado - Carrega checkpoint e testa em peças
# """
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from checkpoint_manager import load_latest_checkpoint
from src.environment.nesting_env_fixed import NestingEnvironmentFixed, NestingConfig
from src.geometry.polygon import Polygon, create_rectangle, create_random_polygon


class NestingPredictor:
    # """
    # Classe para fazer predições usando modelo treinado
    # """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cpu'):
        # """
        # Inicializa o preditor
        
        # Args:
        #     checkpoint_path: Caminho específico do checkpoint (None = mais recente)
        #     device: 'cpu' ou 'cuda'
        # """
        self.device = device
        self.checkpoint = None
        self.actor = None
        self.env = None
        
        # Carrega checkpoint
        if checkpoint_path:
            self.checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            self.checkpoint = load_latest_checkpoint(device=device)
        
        if self.checkpoint is None:
            raise ValueError("Nenhum checkpoint encontrado!")
        
        print("✓ Checkpoint carregado com sucesso")
        self._setup_model()
    
    def _setup_model(self):
        #"""Configura o modelo a partir do checkpoint"""
        import torch.nn as nn
        
        # Modelo simples (deve corresponder ao usado no treinamento)
        class SimpleActor(nn.Module):
            def __init__(self, obs_channels=6, n_actions=3):
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
                    x = x['layout_image']
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        self.actor = SimpleActor()
        
        # Carregar pesos
        if 'actor_state_dict' in self.checkpoint:
            try:
                self.actor.load_state_dict(self.checkpoint['actor_state_dict'])
                print("✓ Pesos do modelo carregados")
            except Exception as e:
                print(f"⚠️  Aviso: Não foi possível carregar pesos exatos: {e}")
                print("   Usando modelo com pesos aleatórios")
        
        self.actor.eval()
        self.actor.to(self.device)
    
    def nest_pieces(self, 
                    pieces: List[Polygon],
                    container_width: float = 1000,
                    container_height: float = 600,
                    render: bool = False,
                    max_steps: Optional[int] = None) -> Dict:
        # """
        # Posiciona peças usando o modelo treinado
        
        # Args:
        #     pieces: Lista de polígonos para posicionar
        #     container_width: Largura do container
        #     container_height: Altura do container
        #     render: Se True, renderiza durante execução
        #     max_steps: Máximo de passos (None = len(pieces))
            
        # Returns:
        #     Dicionário com resultados
        # """
        # Configuração do ambiente
        config = NestingConfig(
            container_width=container_width,
            container_height=container_height,
            max_steps=max_steps or len(pieces) * 2
        )
        
        # CORREÇÃO: Criar ambiente sem passar pieces
        self.env = NestingEnvironmentFixed(
            config=config,
            render_mode='human' if render else None
        )
        
        # CORREÇÃO: Passar pieces através do reset
        obs, info = self.env.reset(options={'pieces': pieces})
        
        # Executar episódio
        total_reward = 0
        placements = []
        done = False
        step = 0
        
        print("\n" + "="*70)
        print("EXECUTANDO NESTING")
        print("="*70)
        
        while not done and step < config.max_steps:
            # Obter ação do modelo
            action = self._predict_action(obs)
            
            # Executar ação
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Registrar placement
            placement_info = {
                'step': step,
                'action': action,
                'reward': reward,
                'status': info.get('placement_status', 'unknown'),
                'utilization': info.get('utilization', 0)
            }
            placements.append(placement_info)
            
            # Log
            print(f"Passo {step}:")
            print(f"  Posição: ({action['position'][0]:.3f}, {action['position'][1]:.3f})")
            print(f"  Rotação: bin {action['rotation']}")
            print(f"  Recompensa: {reward:.4f}")
            print(f"  Status: {info.get('placement_status', 'N/A')}")
            print(f"  Utilização: {info.get('utilization', 0):.2%}")
            
            if render:
                self.env.render()
            
            if terminated:
                print(f"\n✓ Todas as peças posicionadas!")
                break
            elif truncated:
                print(f"\n⚠️  Máximo de passos atingido")
                break
        
        # Resultados finais
        results = {
            'total_reward': total_reward,
            'steps': step,
            'utilization': info.get('utilization', 0),
            'pieces_placed': info.get('n_placed', 0),
            'total_pieces': len(pieces),
            'placements': placements,
            'success': info.get('n_placed', 0) == len(pieces)
        }
        
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        print(f"Recompensa total: {total_reward:.4f}")
        print(f"Passos executados: {step}")
        print(f"Peças posicionadas: {results['pieces_placed']}/{results['total_pieces']}")
        print(f"Utilização final: {results['utilization']:.2%}")
        print(f"Sucesso: {'✓' if results['success'] else '✗'}")
        
        return results
    
    def _predict_action(self, obs: Dict) -> Dict:
        #"""Prediz ação a partir da observação"""
        with torch.no_grad():
            # Converter observação para tensor
            layout_tensor = torch.FloatTensor(obs['layout_image']).unsqueeze(0).to(self.device)
            
            # Obter ação do modelo
            action_values = self.actor({'layout_image': layout_tensor}).squeeze(0).cpu().numpy()
            
            # Converter para formato do ambiente
            # action_values está em [-1, 1]
            position = (action_values[:2] + 1) / 2  # Normalizar para [0, 1]
            rotation_normalized = (action_values[2] + 1) / 2
            rotation_bin = int(rotation_normalized * self.env.config.rotation_bins)
            rotation_bin = np.clip(rotation_bin, 0, self.env.config.rotation_bins - 1)
            
            return {
                'position': position,
                'rotation': rotation_bin
            }
    
    def visualize_result(self):
        #"""Visualiza o resultado final"""
        if self.env is None:
            print("⚠️  Execute nest_pieces() primeiro!")
            return
        
        self.env.render()
        if self.env.render_mode == 'human' and self.env.fig is not None:
            plt.show()
    
    def close(self):
        """Fecha o ambiente"""
        if self.env is not None:
            self.env.close()


def criar_pecas_teste(n_pieces: int = 10, seed: int = 42) -> List[Polygon]:
    #"""Cria conjunto de peças para teste"""
    np.random.seed(seed)
    pieces = []
    
    for i in range(n_pieces):
        if i % 3 == 0:
            # Retângulo
            width = np.random.uniform(50, 150)
            height = np.random.uniform(40, 120)
            piece = create_rectangle(width, height)
        else:
            # Polígono irregular
            n_vertices = np.random.randint(5, 8)
            radius = np.random.uniform(30, 60)
            piece = create_random_polygon(
                n_vertices=n_vertices,
                radius=radius,
                irregularity=0.5,
                spikeyness=0.3
            )
        
        piece.id = i
        pieces.append(piece)
    
    return pieces


def demonstracao_completa():
    """Demonstração completa do uso do modelo treinado"""
    
    print("="*70)
    print("DEMONSTRAÇÃO - USO DE MODELO TREINADO")
    print("="*70)
    
    try:
        # 1. Criar peças de teste
        print("\n📦 Criando peças de teste...")
        pieces = criar_pecas_teste(n_pieces=8, seed=42)
        print(f"   Criadas {len(pieces)} peças")
        
        # Calcular área total
        total_area = sum(p.area for p in pieces)
        print(f"   Área total das peças: {total_area:.2f}")
        
        # 2. Inicializar preditor
        print("\n🧠 Inicializando preditor...")
        predictor = NestingPredictor(device='cpu')
        
        # Mostrar info do checkpoint
        if 'epoch' in predictor.checkpoint:
            print(f"   Época: {predictor.checkpoint['epoch']}")
        if 'avg_reward' in predictor.checkpoint:
            print(f"   Recompensa média treino: {predictor.checkpoint['avg_reward']:.4f}")
        
        # 3. Executar nesting
        print("\n🎯 Executando nesting...")
        result = predictor.nest_pieces(
            pieces=pieces,
            container_width=1000,
            container_height=600,
            render=False  # Mude para True se quiser ver em tempo real
        )
        
        # 4. Visualizar resultado final
        print("\n📊 Visualizando resultado final...")
        predictor.visualize_result()
        
        # 5. Análise dos resultados
        print("\n" + "="*70)
        print("ANÁLISE DETALHADA")
        print("="*70)
        
        if result['success']:
            print("✅ SUCESSO - Todas as peças foram posicionadas!")
        else:
            print(f"⚠️  PARCIAL - {result['pieces_placed']}/{result['total_pieces']} peças posicionadas")
        
        print(f"\nEficiência:")
        print(f"  Utilização do container: {result['utilization']:.2%}")
        print(f"  Recompensa por passo: {result['total_reward']/max(result['steps'], 1):.4f}")
        
        # Análise de placements
        successful_placements = sum(1 for p in result['placements'] if p['status'] == 'valid')
        print(f"\nPlacements:")
        print(f"  Válidos: {successful_placements}/{len(result['placements'])}")
        print(f"  Taxa de sucesso: {successful_placements/max(len(result['placements']), 1):.2%}")
        
        # Limpeza
        predictor.close()
        
        return result
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


def teste_multiplos_casos():
    #"""Testa o modelo em múltiplos casos"""
    print("="*70)
    print("TESTE EM MÚLTIPLOS CASOS")
    print("="*70)
    
    predictor = NestingPredictor(device='cpu')
    
    casos = [
        {'n_pieces': 5, 'seed': 42, 'nome': 'Fácil (5 peças)'},
        {'n_pieces': 10, 'seed': 123, 'nome': 'Médio (10 peças)'},
        {'n_pieces': 15, 'seed': 456, 'nome': 'Difícil (15 peças)'},
    ]
    
    resultados = []
    
    for caso in casos:
        print(f"\n{'='*70}")
        print(f"Caso: {caso['nome']}")
        print(f"{'='*70}")
        
        pieces = criar_pecas_teste(n_pieces=caso['n_pieces'], seed=caso['seed'])
        
        result = predictor.nest_pieces(
            pieces=pieces,
            container_width=1000,
            container_height=600,
            render=False
        )
        
        resultados.append({
            'caso': caso['nome'],
            'result': result
        })
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    
    for r in resultados:
        print(f"\n{r['caso']}:")
        print(f"  Utilização: {r['result']['utilization']:.2%}")
        print(f"  Sucesso: {'✓' if r['result']['success'] else '✗'}")
        print(f"  Recompensa: {r['result']['total_reward']:.2f}")
    
    predictor.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--multiplos':
        teste_multiplos_casos()
    else:
        demonstracao_completa()
        
    print("\n✅ Demonstração concluída!")