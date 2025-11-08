import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Imports dos módulos anteriores (assumindo que estão implementados)
# from src.geometry.polygon import Polygon, Point, create_rectangle
# from src.geometry.nfp import NFPCalculator

# Gymnasium Environment para Nesting 2D
# Compatível com algoritmos de RL (PPO, DQN, etc.)

@dataclass
class NestingConfig:
    #"""Configuração do ambiente"""
    container_width: float = 1000.0
    container_height: float = 600.0
    max_pieces: int = 50
    max_steps: int = 100
    rotation_bins: int = 36  # 360/10 = 36 bins
    image_size: int = 256
    spacing: float = 2.0  # mm entre peças
    
    # Recompensas
    valid_placement_reward: float = 1.0
    invalid_placement_penalty: float = -5.0
    touching_bonus: float = 0.5
    corner_bonus: float = 0.3
    hole_penalty: float = -0.2
    progress_reward: float = 0.1
    time_penalty: float = -0.01
    final_multiplier: float = 100.0


class NestingEnvironment(gym.Env):
    # """
    # Ambiente Gymnasium para problema de Nesting 2D.
    
    # Observation Space:
    #     - Layout image: (6, 256, 256) float32
    #     - Piece features: (max_pieces, feature_dim) float32
    #     - Stats: (n_stats,) float32
    
    # Action Space:
    #     - position_x: [0, 1] continuous
    #     - position_y: [0, 1] continuous
    #     - rotation: discrete (0-35, representando 0-350°)
    # """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, 
                 config: Optional[NestingConfig] = None,
                 render_mode: Optional[str] = None):
        # """
        # Args:
        #     config: Configuração do ambiente
        #     render_mode: 'human' ou 'rgb_array'
        # """
        super().__init__()
        
        self.config = config or NestingConfig()
        self.render_mode = render_mode
        
        # Espaços
        self._setup_spaces()
        
        # Estado interno
        self.container = None
        self.pieces_to_place = []
        self.placed_pieces = []
        self.current_piece_idx = 0
        self.step_count = 0
        
        # NFP Calculator
        # self.nfp_calc = NFPCalculator(cache_enabled=True)
        
        # Visualização
        self.fig = None
        self.ax = None
    
    def _setup_spaces(self):
        #"""Define observation e action spaces"""
        cfg = self.config
        
        # =====================================================================
        # OBSERVATION SPACE
        # =====================================================================
        
        # Componentes da observação:
        # 1. Layout image (6 canais, 256×256)
        # 2. Features da próxima peça (tamanho fixo)
        # 3. Stats globais
        
        self.observation_space = spaces.Dict({
            # Imagem do layout atual
            'layout_image': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6, cfg.image_size, cfg.image_size),
                dtype=np.float32
            ),
            
            # Features da peça atual
            'current_piece': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),  # [area, perimeter, width, height, ...]
                dtype=np.float32
            ),
            
            # Features das peças restantes (agregado)
            'remaining_pieces': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            ),
            
            # Stats globais
            'stats': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5,),  # [utilization, progress, n_placed, ...]
                dtype=np.float32
            )
        })
        
        # =====================================================================
        # ACTION SPACE
        # =====================================================================
        
        # Ação = [position_x, position_y, rotation_bin]
        # position_x, position_y: [0, 1] continuous (normalizado)
        # rotation_bin: discrete 0-35 (bins de 10 graus)
        
        self.action_space = spaces.Dict({
            'position': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),  # (x, y)
                dtype=np.float32
            ),
            'rotation': spaces.Discrete(cfg.rotation_bins)
        })
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        # """
        # Reseta o ambiente.
        
        # Args:
        #     seed: Seed para reprodutibilidade
        #     options: Pode conter 'pieces' customizadas
            
        # Returns:
        #     observation, info
        # """
        super().reset(seed=seed)
        
        # Criar container (chapa)
        self.container = self._create_container()
        
        # Obter/gerar peças
        if options and 'pieces' in options:
            self.pieces_to_place = options['pieces'].copy()
        else:
            self.pieces_to_place = self._generate_random_pieces()
        
        # Resetar estado
        self.placed_pieces = []
        self.current_piece_idx = 0
        self.step_count = 0
        
        # Observação inicial
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        # """
        # Executa uma ação.
        
        # Args:
        #     action: Dict com 'position' e 'rotation'
            
        # Returns:
        #     observation, reward, terminated, truncated, info
        # """
        self.step_count += 1
        
        # Extrair ação
        position_normalized = action['position']  # [0, 1]
        rotation_bin = action['rotation']  # 0-35
        
        # Converter para coordenadas reais
        x = position_normalized[0] * self.config.container_width
        y = position_normalized[1] * self.config.container_height
        rotation_degrees = rotation_bin * (360.0 / self.config.rotation_bins)
        
        # Pegar peça atual
        current_piece = self.pieces_to_place[self.current_piece_idx]
        
        # Tentar colocar peça
        success, reward, info_placement = self._place_piece(
            current_piece, x, y, rotation_degrees
        )
        
        # Calcular recompensa total
        reward_total = reward
        
        # Penalidade de tempo
        reward_total += self.config.time_penalty
        
        # Verificar se terminou
        terminated = False
        truncated = False
        
        if success:
            self.current_piece_idx += 1
            
            # Todas as peças colocadas?
            if self.current_piece_idx >= len(self.pieces_to_place):
                terminated = True
                # Bônus final baseado na utilização
                utilization = self._calculate_utilization()
                final_bonus = utilization * self.config.final_multiplier
                reward_total += final_bonus
        else:
            # Colocação inválida - terminar episódio
            terminated = True
        
        # Timeout?
        if self.step_count >= self.config.max_steps:
            truncated = True
        
        # Nova observação
        observation = self._get_observation()
        info = self._get_info()
        info.update(info_placement)
        
        return observation, reward_total, terminated, truncated, info
    
    def _place_piece(self, piece, x: float, y: float, 
                    rotation: float) -> Tuple[bool, float, Dict]:
        # """
        # Tenta colocar uma peça.
        
        # Returns:
        #     success, reward, info
        # """
        # Rotacionar peça
        # rotated_piece = piece.rotate(rotation)
        # moved_piece = rotated_piece.set_position(x, y)
        
        # SIMULAÇÃO (sem geometria real por enquanto)
        # Na implementação real, usar NFP para verificar colisões
        
        # Simulação simplificada: checar bounds
        in_bounds = (0 <= x <= self.config.container_width and
                    0 <= y <= self.config.container_height)
        
        # Simular colisão (aleatório para teste)
        collides = np.random.random() < 0.1  # 10% chance de colisão
        
        if not in_bounds or collides:
            # Colocação inválida
            return False, self.config.invalid_placement_penalty, {
                'placement_status': 'invalid',
                'reason': 'out_of_bounds' if not in_bounds else 'collision'
            }
        
        # Colocação válida!
        # self.placed_pieces.append(moved_piece)
        
        # Simular colocação (placeholder)
        self.placed_pieces.append({
            'x': x,
            'y': y,
            'rotation': rotation,
            'piece_idx': self.current_piece_idx
        })
        
        # Calcular recompensa
        reward = self.config.valid_placement_reward
        
        # Bônus se toca outras peças
        if len(self.placed_pieces) > 1:
            # Simulação: 30% chance de tocar
            if np.random.random() < 0.3:
                reward += self.config.touching_bonus
        
        # Bônus se perto do canto
        if x < 100 and y < 100:
            reward += self.config.corner_bonus
        
        # Recompensa de progresso
        reward += self.config.progress_reward
        
        return True, reward, {
            'placement_status': 'valid',
            'position': (x, y),
            'rotation': rotation
        }
    
    def _get_observation(self) -> Dict:
       # """Constrói observação do estado atual"""
        
        # 1. Layout image (6 canais)
        layout_image = self._render_layout_as_image()
        
        # 2. Features da peça atual
        if self.current_piece_idx < len(self.pieces_to_place):
            current_piece_features = self._extract_piece_features(
                self.pieces_to_place[self.current_piece_idx]
            )
        else:
            current_piece_features = np.zeros(10, dtype=np.float32)
        
        # 3. Features agregadas das peças restantes
        remaining_features = self._extract_remaining_features()
        
        # 4. Stats globais
        stats = np.array([
            self._calculate_utilization(),
            self.current_piece_idx / len(self.pieces_to_place),
            len(self.placed_pieces) / len(self.pieces_to_place),
            self.step_count / self.config.max_steps,
            0.0  # placeholder
        ], dtype=np.float32)
        
        return {
            'layout_image': layout_image,
            'current_piece': current_piece_features,
            'remaining_pieces': remaining_features,
            'stats': stats
        }
    
    def _render_layout_as_image(self) -> np.ndarray:
        # """
        # Renderiza layout como imagem 6-channel.
        
        # Returns:
        #     array (6, 256, 256) float32
        # """
        size = self.config.image_size
        image = np.zeros((6, size, size), dtype=np.float32)
        
        # SIMULAÇÃO: preencher com padrão aleatório
        # Na implementação real, renderizar peças geometricamente
        
        # Canal 0: Ocupação
        for piece_info in self.placed_pieces:
            # Simular peça como círculo
            cx = int(piece_info['x'] / self.config.container_width * size)
            cy = int(piece_info['y'] / self.config.container_height * size)
            radius = 10
            
            y_grid, x_grid = np.ogrid[:size, :size]
            mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
            image[0][mask] = 1.0
        
        # Canal 1: Bordas (simplificado)
        # Canal 2: Distâncias (simplificado)
        # Canal 3: Próxima peça (simplificado)
        # Canal 4: Densidade (simplificado)
        # Canal 5: Acessibilidade (simplificado)
        
        # Para teste, preencher canais com dados sintéticos
        image[1] = np.random.rand(size, size) * 0.1
        image[2] = np.random.rand(size, size) * 0.2
        image[3] = np.random.rand(size, size) * 0.3 if self.current_piece_idx < len(self.pieces_to_place) else 0
        image[4] = np.random.rand(size, size) * 0.1
        image[5] = np.random.rand(size, size) * 0.1
        
        return image
    
    def _extract_piece_features(self, piece) -> np.ndarray:
        #"""Extrai features de uma peça"""
        # SIMULAÇÃO: features aleatórias
        # Na implementação real, extrair de piece.area, piece.perimeter, etc.
        return np.random.rand(10).astype(np.float32)
    
    def _extract_remaining_features(self) -> np.ndarray:
        #"""Features agregadas das peças restantes"""
        # SIMULAÇÃO
        n_remaining = len(self.pieces_to_place) - self.current_piece_idx
        return np.array([
            n_remaining,
            np.random.rand(),  # área total
            np.random.rand(),  # perímetro médio
            *np.random.rand(7)
        ], dtype=np.float32)
    
    def _calculate_utilization(self) -> float:
        #"""Calcula taxa de utilização"""
        # SIMULAÇÃO
        if len(self.placed_pieces) == 0:
            return 0.0
        
        # Na implementação real: (soma áreas peças) / (área container)
        utilization = len(self.placed_pieces) / len(self.pieces_to_place) * 0.85
        return min(utilization, 1.0)
    
    def _get_info(self) -> Dict:
        #"""Informações adicionais"""
        return {
            'step': self.step_count,
            'n_placed': len(self.placed_pieces),
            'n_remaining': len(self.pieces_to_place) - self.current_piece_idx,
            'utilization': self._calculate_utilization()
        }
    
    def _create_container(self):
        #"""Cria container (chapa)"""
        # from src.geometry.polygon import create_rectangle
        # return create_rectangle(
        #     self.config.container_width,
        #     self.config.container_height,
        #     center=(self.config.container_width/2, self.config.container_height/2)
        # )
        return {'width': self.config.container_width, 
                'height': self.config.container_height}
    
    def _generate_random_pieces(self) -> List:
        #"""Gera peças aleatórias para teste"""
        # from src.geometry.polygon import create_random_polygon
        n_pieces = np.random.randint(5, 15)
        pieces = []
        
        for i in range(n_pieces):
            # piece = create_random_polygon(
            #     n_vertices=np.random.randint(4, 10),
            #     radius=np.random.uniform(20, 50),
            #     irregularity=0.5,
            #     spikeyness=0.3
            # )
            # piece.id = i
            # pieces.append(piece)
            
            # PLACEHOLDER
            pieces.append({'id': i, 'area': np.random.uniform(100, 500)})
        
        return pieces
    
    def render(self):
        #"""Renderiza o estado atual"""
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        #"""Renderização para visualização humana"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.ax.clear()
        
        # Desenhar container
        rect = plt.Rectangle(
            (0, 0),
            self.config.container_width,
            self.config.container_height,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(rect)
        
        # Desenhar peças colocadas
        for piece_info in self.placed_pieces:
            circle = plt.Circle(
                (piece_info['x'], piece_info['y']),
                20,
                color='lightblue',
                ec='black'
            )
            self.ax.add_patch(circle)
        
        # Configurar eixos
        self.ax.set_xlim(-50, self.config.container_width + 50)
        self.ax.set_ylim(-50, self.config.container_height + 50)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Informações
        info_text = (f"Step: {self.step_count}\n"
                    f"Placed: {len(self.placed_pieces)}/{len(self.pieces_to_place)}\n"
                    f"Utilization: {self._calculate_utilization()*100:.1f}%")
        self.ax.text(0.02, 0.98, info_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.pause(0.01)
        return self.fig
    
    def _render_rgb_array(self) -> np.ndarray:
        #"""Renderização como array RGB"""
        # Renderizar para figura temporária
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Similar ao _render_human mas sem mostrar
        rect = plt.Rectangle(
            (0, 0),
            self.config.container_width,
            self.config.container_height,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        
        for piece_info in self.placed_pieces:
            circle = plt.Circle(
                (piece_info['x'], piece_info['y']),
                20,
                color='lightblue',
                ec='black'
            )
            ax.add_patch(circle)
        
        ax.set_xlim(-50, self.config.container_width + 50)
        ax.set_ylim(-50, self.config.container_height + 50)
        ax.set_aspect('equal')
        
        # Converter para array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return image
    
    def close(self):
        #"""Fecha o ambiente"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# =============================================================================
# Testes
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE DO AMBIENTE DE RL")
    print("="*70)
    
    # Criar ambiente
    config = NestingConfig(
        container_width=1000,
        container_height=600,
        max_steps=50
    )
    
    env = NestingEnvironment(config=config, render_mode='human')
    
    print(f"\n✓ Ambiente criado")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Resetar
    observation, info = env.reset(seed=42)
    print(f"\n✓ Ambiente resetado")
    print(f"  Info: {info}")
    
    # Executar alguns steps aleatórios
    print(f"\n" + "="*70)
    print("EXECUTANDO EPISÓDIO DE TESTE")
    print("="*70)
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 20:
        # Ação aleatória
        action = {
            'position': env.action_space['position'].sample(),
            'rotation': env.action_space['rotation'].sample()
        }
        
        # Step
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        print(f"\nStep {step}:")
        print(f"  Action: pos=({action['position'][0]:.2f}, {action['position'][1]:.2f}), "
              f"rot={action['rotation']*10}°")
        print(f"  Reward: {reward:.2f}")
        print(f"  Info: {info}")
        
        # Renderizar
        if step % 5 == 0:
            env.render()
    
    print(f"\n" + "="*70)
    print(f"EPISÓDIO TERMINADO")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps: {step}")
    print(f"  Utilization: {info['utilization']*100:.1f}%")
    print("="*70)
    
    env.close()
    
    print("\n✓ Ambiente de RL implementado e testado com sucesso!")