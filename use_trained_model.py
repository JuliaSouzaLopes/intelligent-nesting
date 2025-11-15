# """
# use_trained_model.py

# Sistema completo para usar o modelo treinado em produção
# - Carrega peças de arquivos (DXF, SVG, JSON)
# - Executa nesting com modelo treinado
# - Exporta resultado para produção
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import time

from src.environment.nesting_env import NestingEnvironment, NestingConfig
from src.geometry.polygon import Polygon, create_rectangle, create_random_polygon


# =============================================================================
# Carregador de Peças
# =============================================================================

class PieceLoader:
    #"""Carrega peças de diferentes formatos"""
    
    @staticmethod
    def from_json(filepath: str) -> List[Polygon]:
        # """
        # Carrega peças de arquivo JSON.
        
        # Formato esperado:
        # {
        #     "pieces": [
        #         {
        #             "id": 0,
        #             "vertices": [[x1, y1], [x2, y2], ...]
        #         },
        #         ...
        #     ]
        # }
        # """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        pieces = []
        for piece_data in data['pieces']:
            vertices = piece_data['vertices']
            piece = Polygon(vertices, id=piece_data.get('id'))
            pieces.append(piece)
        
        print(f"✓ Carregadas {len(pieces)} peças de {filepath}")
        return pieces
    
    @staticmethod
    def from_dxf(filepath: str) -> List[Polygon]:
        # """
        # Carrega peças de arquivo DXF.
        
        # Requer: pip install ezdxf
        # """
        try:
            import ezdxf
        except ImportError:
            print("❌ ezdxf não instalado. Execute: pip install ezdxf")
            return []
        
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        
        pieces = []
        piece_id = 0
        
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = [(p[0], p[1]) for p in entity.get_points()]
                if len(vertices) >= 3:
                    piece = Polygon(vertices, id=piece_id)
                    pieces.append(piece)
                    piece_id += 1
        
        print(f"✓ Carregadas {len(pieces)} peças de {filepath}")
        return pieces
    
    @staticmethod
    def from_svg(filepath: str) -> List[Polygon]:
        # """
        # Carrega peças de arquivo SVG.
        
        # Requer: pip install svgpathtools
        # """
        try:
            from svgpathtools import svg2paths
        except ImportError:
            print("❌ svgpathtools não instalado. Execute: pip install svgpathtools")
            return []
        
        paths, attributes = svg2paths(filepath)
        
        pieces = []
        for i, path in enumerate(paths):
            vertices = []
            for segment in path:
                vertices.append((segment.start.real, segment.start.imag))
            
            if len(vertices) >= 3:
                piece = Polygon(vertices, id=i)
                pieces.append(piece)
        
        print(f"✓ Carregadas {len(pieces)} peças de {filepath}")
        return pieces
    
    @staticmethod
    def from_rectangles_list(rectangles: List[Tuple[float, float]]) -> List[Polygon]:
        # """
        # Cria retângulos a partir de lista de dimensões.
        
        # Args:
        #     rectangles: [(width, height), ...]
        # """
        pieces = []
        for i, (width, height) in enumerate(rectangles):
            piece = create_rectangle(width, height)
            piece.id = i
            pieces.append(piece)
        
        print(f"✓ Criados {len(pieces)} retângulos")
        return pieces
    
    @staticmethod
    def create_example_pieces() -> List[Polygon]:
        #"""Cria peças de exemplo para teste"""
        pieces = [
            create_rectangle(100, 60),
            create_rectangle(80, 50),
            create_rectangle(90, 70),
            create_rectangle(70, 40),
            create_rectangle(110, 55),
        ]
        
        for i, piece in enumerate(pieces):
            piece.id = i
        
        print(f"✓ Criadas {len(pieces)} peças de exemplo")
        return pieces


# =============================================================================
# Sistema de Nesting
# =============================================================================

class NestingSystem:
    #"""Sistema completo de nesting com modelo treinado"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 container_width: float = 1000.0,
                 container_height: float = 600.0,
                 device: str = 'cuda'):
        # """
        # Args:
        #     checkpoint_path: Caminho do checkpoint (.pt)
        #     container_width: Largura da chapa (mm)
        #     container_height: Altura da chapa (mm)
        #     device: 'cuda' ou 'cpu'
        # """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.container_width = container_width
        self.container_height = container_height
        
        # Carregar modelo
        print("="*70)
        print("CARREGANDO MODELO TREINADO")
        print("="*70)
        
        self.agent = self._load_model(checkpoint_path)
        
        print(f"✓ Modelo carregado de: {checkpoint_path}")
        print(f"✓ Device: {self.device}")
        print("="*70 + "\n")
    
    def _load_model(self, checkpoint_path: str):
        #"""Carrega modelo do checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Detectar qual tipo de modelo foi treinado
        state_dict = checkpoint['agent_state_dict']
        
        # Verificar tamanho para determinar tipo
        n_params = sum(p.numel() for p in state_dict.values())
        
        if n_params < 1e6:  # Modelo pequeno (TinyActorCritic)
            print(f"  Detectado: Modelo Leve (~{n_params/1e6:.1f}M params)")
            from train_2gb_gpu import TinyActorCritic
            agent = TinyActorCritic(
                embedding_dim=64,
                hidden_dim=128,
                rotation_bins=36
            )
        else:  # Modelo completo
            print(f"  Detectado: Modelo Completo (~{n_params/1e6:.1f}M params)")
            from train_complete_system_fixed import ActorCritic
            agent = ActorCritic(
                cnn_embedding_dim=256,
                hidden_dim=512,
                rotation_bins=36
            )
        
        agent.load_state_dict(state_dict)
        agent = agent.to(self.device)
        agent.eval()
        
        return agent
    
    def nest_pieces(self, 
                    pieces: List[Polygon],
                    max_attempts: int = 3,
                    visualize: bool = True) -> Dict:
        # """
        # Executa nesting nas peças fornecidas.
        
        # Args:
        #     pieces: Lista de peças (Polygon)
        #     max_attempts: Número de tentativas (retorna a melhor)
        #     visualize: Se True, mostra visualização
        
        # Returns:
        #     Dict com resultado:
        #     {
        #         'placed_pieces': List[Polygon],
        #         'utilization': float,
        #         'n_placed': int,
        #         'execution_time': float,
        #         'layout': imagem do layout
        #     }
        # """
        print(f"Executando nesting de {len(pieces)} peças...")
        print(f"Container: {self.container_width}mm × {self.container_height}mm")
        print(f"Tentativas: {max_attempts}\n")
        
        best_result = None
        best_utilization = 0
        
        start_time = time.time()
        
        for attempt in range(max_attempts):
            print(f"Tentativa {attempt + 1}/{max_attempts}...")
            
            result = self._single_nesting_run(pieces)
            
            utilization = result['utilization']
            print(f"  Utilização: {utilization*100:.2f}%")
            print(f"  Peças colocadas: {result['n_placed']}/{len(pieces)}")
            
            if utilization > best_utilization:
                best_utilization = utilization
                best_result = result
                print(f"  ✓ Nova melhor solução!")
            
            print()
        
        execution_time = time.time() - start_time
        best_result['execution_time'] = execution_time
        
        print("="*70)
        print("RESULTADO FINAL")
        print("="*70)
        print(f"Utilização: {best_result['utilization']*100:.2f}%")
        print(f"Peças colocadas: {best_result['n_placed']}/{len(pieces)}")
        print(f"Tempo de execução: {execution_time:.2f}s")
        print("="*70 + "\n")
        
        if visualize:
            self._visualize_result(best_result)
        
        return best_result
    
    def _single_nesting_run(self, pieces: List[Polygon]) -> Dict:
        #"""Executa uma tentativa de nesting"""
        # Criar environment
        env_config = NestingConfig(
            container_width=self.container_width,
            container_height=self.container_height,
            max_steps=len(pieces) * 3
        )
        env = NestingEnvironment(config=env_config)
        
        obs, _ = env.reset(options={'pieces': pieces.copy()})
        
        placed_pieces = []
        done = False
        
        while not done:
            # Converter obs para tensor
            obs_tensor = {
                'layout_image': torch.from_numpy(obs['layout_image']).unsqueeze(0).to(self.device),
                'current_piece': torch.from_numpy(obs['current_piece']).unsqueeze(0).to(self.device),
                'remaining_pieces': torch.from_numpy(obs['remaining_pieces']).unsqueeze(0).to(self.device),
                'stats': torch.from_numpy(obs['stats']).unsqueeze(0).to(self.device)
            }
            
            # Predição do modelo (determinístico)
            with torch.no_grad():
                action, _, _ = self.agent.get_action(obs_tensor, deterministic=True)
            
            # Executar ação
            action_dict = {
                'position': action['position'][0].cpu().numpy(),
                'rotation': int(action['rotation'][0].cpu().item())
            }
            
            obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
        
        # Pegar peças colocadas
        placed_pieces = env.placed_pieces.copy()
        
        env.close()
        
        return {
            'placed_pieces': placed_pieces,
            'utilization': info['utilization'],
            'n_placed': info['n_placed'],
            'total_pieces': len(pieces)
        }
    
    def _visualize_result(self, result: Dict):
        #"""Visualiza resultado do nesting"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Desenhar container
        container = plt.Rectangle(
            (0, 0),
            self.container_width,
            self.container_height,
            fill=False,
            edgecolor='black',
            linewidth=2,
            label='Container'
        )
        ax.add_patch(container)
        
        # Desenhar peças colocadas
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(result['placed_pieces'])))
        
        for i, piece in enumerate(result['placed_pieces']):
            try:
                piece.plot(ax, facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
            except:
                # Fallback: desenhar como círculo
                ax.plot(piece.position.x, piece.position.y, 'o', 
                       color=colors[i], markersize=10)
        
        # Configurar plot
        ax.set_xlim(-50, self.container_width + 50)
        ax.set_ylim(-50, self.container_height + 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Título com informações
        utilization = result['utilization']
        n_placed = result['n_placed']
        total = result['total_pieces']
        
        ax.set_title(
            f'Resultado do Nesting\n'
            f'Utilização: {utilization*100:.2f}% | '
            f'Peças: {n_placed}/{total}',
            fontsize=14,
            fontweight='bold'
        )
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('nesting_result.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualização salva: nesting_result.png")
        
        plt.show()


# =============================================================================
# Exportador de Resultados
# =============================================================================

class ResultExporter:
    """Exporta resultados para diferentes formatos"""
    
    @staticmethod
    def to_json(result: Dict, filepath: str):
        #"""Exporta para JSON"""
        data = {
            'container': {
                'width': result.get('container_width', 1000),
                'height': result.get('container_height', 600)
            },
            'utilization': float(result['utilization']),
            'n_placed': result['n_placed'],
            'total_pieces': result['total_pieces'],
            'execution_time': result.get('execution_time', 0),
            'pieces': []
        }
        
        for piece in result['placed_pieces']:
            piece_data = {
                'id': piece.id,
                'position': {
                    'x': float(piece.position.x),
                    'y': float(piece.position.y)
                },
                'rotation': float(piece.rotation),
                'vertices': [(float(v.x), float(v.y)) for v in piece.vertices],
                'area': float(piece.area)
            }
            data['pieces'].append(piece_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Resultado exportado: {filepath}")
    
    @staticmethod
    def to_dxf(result: Dict, filepath: str):
        #"""Exporta para DXF (CAD)"""
        try:
            import ezdxf
        except ImportError:
            print("❌ ezdxf não instalado. Execute: pip install ezdxf")
            return
        
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Desenhar container
        container_width = result.get('container_width', 1000)
        container_height = result.get('container_height', 600)
        
        msp.add_lwpolyline([
            (0, 0),
            (container_width, 0),
            (container_width, container_height),
            (0, container_height),
            (0, 0)
        ], dxfattribs={'layer': 'CONTAINER', 'color': 1})
        
        # Desenhar peças
        for piece in result['placed_pieces']:
            points = [(v.x, v.y) for v in piece.vertices]
            points.append(points[0])  # Fechar polígono
            
            msp.add_lwpolyline(
                points,
                dxfattribs={
                    'layer': f'PIECE_{piece.id}',
                    'color': 2 + (piece.id % 6)
                }
            )
        
        doc.saveas(filepath)
        print(f"✓ DXF exportado: {filepath}")
    
    @staticmethod
    def to_svg(result: Dict, filepath: str):
        """Exporta para SVG"""
        container_width = result.get('container_width', 1000)
        container_height = result.get('container_height', 600)
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{container_width}" height="{container_height}" 
     viewBox="0 0 {container_width} {container_height}"
     xmlns="http://www.w3.org/2000/svg">
    
    <!-- Container -->
    <rect x="0" y="0" width="{container_width}" height="{container_height}"
          fill="none" stroke="black" stroke-width="2"/>
    
    <!-- Peças -->
'''
        
        # Cores para as peças
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        for piece in result['placed_pieces']:
            points = ' '.join([f"{v.x},{v.y}" for v in piece.vertices])
            color = colors[piece.id % len(colors)]
            
            svg_content += f'''    <polygon points="{points}" 
            fill="{color}" fill-opacity="0.7" 
            stroke="black" stroke-width="1"/>\n'''
        
        svg_content += '''
    <!-- Info -->
    <text x="10" y="30" font-size="20" font-family="Arial">
        Utilização: {:.1f}%
    </text>
    <text x="10" y="55" font-size="16" font-family="Arial">
        Peças: {}/{} 
    </text>
</svg>'''.format(
            result['utilization'] * 100,
            result['n_placed'],
            result['total_pieces']
        )
        
        with open(filepath, 'w') as f:
            f.write(svg_content)
        
        print(f"✓ SVG exportado: {filepath}")


# =============================================================================
# EXEMPLOS DE USO
# =============================================================================

def example_1_from_rectangles():
    #"""Exemplo 1: Nesting de retângulos simples"""
    print("\n" + "="*70)
    print("EXEMPLO 1: RETÂNGULOS SIMPLES")
    print("="*70 + "\n")
    
    # Definir dimensões das peças (mm)
    rectangle_dimensions = [
        (150, 100),  # Peça 1: 150mm × 100mm
        (120, 80),   # Peça 2: 120mm × 80mm
        (180, 90),   # Peça 3: 180mm × 90mm
        (100, 70),   # Peça 4: 100mm × 70mm
        (140, 110),  # Peça 5: 140mm × 110mm
    ]
    
    # Carregar peças
    loader = PieceLoader()
    pieces = loader.from_rectangles_list(rectangle_dimensions)
    
    # Criar sistema de nesting
    system = NestingSystem(
        checkpoint_path='checkpoint_tiny_100.pt',  # ← Seu checkpoint
        container_width=1000,  # 1000mm = 1 metro
        container_height=600,  # 600mm
        device='cuda'  # ou 'cpu'
    )
    
    # Executar nesting
    result = system.nest_pieces(pieces, max_attempts=3, visualize=True)
    
    # Exportar resultados
    exporter = ResultExporter()
    exporter.to_json(result, 'nesting_result.json')
    exporter.to_svg(result, 'nesting_result.svg')
    
    return result


def example_2_from_json():
    #"""Exemplo 2: Carregar peças de arquivo JSON"""
    print("\n" + "="*70)
    print("EXEMPLO 2: CARREGAR DE JSON")
    print("="*70 + "\n")
    
    # Primeiro, criar arquivo JSON de exemplo
    example_json = {
        "pieces": [
            {
                "id": 0,
                "vertices": [[0, 0], [100, 0], [100, 60], [0, 60]]
            },
            {
                "id": 1,
                "vertices": [[0, 0], [80, 0], [80, 50], [0, 50]]
            }
        ]
    }
    
    with open('pieces_input.json', 'w') as f:
        json.dump(example_json, f, indent=2)
    
    print("✓ Arquivo pieces_input.json criado")
    
    # Carregar peças
    loader = PieceLoader()
    pieces = loader.from_json('pieces_input.json')
    
    # Executar nesting
    system = NestingSystem(
        checkpoint_path='checkpoint_tiny_100.pt',
        container_width=1000,
        container_height=600
    )
    
    result = system.nest_pieces(pieces)
    
    # Exportar
    exporter = ResultExporter()
    exporter.to_json(result, 'result.json')
    exporter.to_dxf(result, 'result.dxf')
    
    return result


def example_3_production():
    #"""Exemplo 3: Uso em produção - múltiplas chapas"""
    print("\n" + "="*70)
    print("EXEMPLO 3: PRODUÇÃO - MÚLTIPLAS CHAPAS")
    print("="*70 + "\n")
    
    # Muitas peças
    rectangles = [
        (150, 100), (120, 80), (180, 90), (100, 70),
        (140, 110), (160, 85), (130, 95), (110, 75),
        (170, 105), (125, 90), (155, 100), (135, 85)
    ]
    
    loader = PieceLoader()
    all_pieces = loader.from_rectangles_list(rectangles)
    
    system = NestingSystem(
        checkpoint_path='checkpoint_tiny_100.pt',
        container_width=1000,
        container_height=600
    )
    
    # Dividir em chapas
    results = []
    remaining_pieces = all_pieces.copy()
    sheet_number = 1
    
    while remaining_pieces:
        print(f"\n--- CHAPA {sheet_number} ---")
        print(f"Peças restantes: {len(remaining_pieces)}")
        
        result = system.nest_pieces(
            remaining_pieces,
            max_attempts=2,
            visualize=False
        )
        
        results.append(result)
        
        # Remover peças colocadas
        n_placed = result['n_placed']
        remaining_pieces = remaining_pieces[n_placed:]
        
        # Exportar chapa
        exporter = ResultExporter()
        exporter.to_json(result, f'sheet_{sheet_number}.json')
        exporter.to_dxf(result, f'sheet_{sheet_number}.dxf')
        
        sheet_number += 1
        
        if sheet_number > 10:  # Limite de segurança
            print("\n⚠️  Limite de chapas atingido")
            break
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO DA PRODUÇÃO")
    print("="*70)
    print(f"Total de peças: {len(all_pieces)}")
    print(f"Peças colocadas: {len(all_pieces) - len(remaining_pieces)}")
    print(f"Número de chapas: {len(results)}")
    
    avg_util = np.mean([r['utilization'] for r in results])
    print(f"Utilização média: {avg_util*100:.2f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Menu interativo"""
    print("\n" + "="*70)
    print("SISTEMA DE NESTING - USO EM PRODUÇÃO")
    print("="*70)
    print("""
Escolha uma opção:

1. Exemplo 1: Retângulos simples
2. Exemplo 2: Carregar de JSON
3. Exemplo 3: Produção (múltiplas chapas)
4. Criar seu próprio caso
5. Sair
    """)
    
    choice = input("Opção: ").strip()
    
    if choice == '1':
        example_1_from_rectangles()
    elif choice == '2':
        example_2_from_json()
    elif choice == '3':
        example_3_production()
    elif choice == '4':
        print("\n✓ Para criar seu caso:")
        print("1. Edite a função example_1_from_rectangles()")
        print("2. Modifique rectangle_dimensions")
        print("3. Execute novamente")
    else:
        print("\n✓ Até logo!")


if __name__ == "__main__":
    # Verificar se existe checkpoint
    if not Path('checkpoint_tiny_100.pt').exists():
        print("⚠️  Checkpoint não encontrado!")
        print("\nPrimeiro, treine o modelo:")
        print("  python train_2gb_gpu.py")
        print("\nDepois, use este script.")
    else:
        main()