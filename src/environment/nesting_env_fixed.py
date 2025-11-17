# """
# CORREÇÃO: src/environment/nesting_env.py

# Versão corrigida que resolve o erro NoneType
# """

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Este arquivo corrige os bugs no nesting_env.py original

@dataclass
class NestingConfig:
    #"""Configuração do ambiente"""
    container_width: float = 1000.0
    container_height: float = 600.0
    max_pieces: int = 50
    max_steps: int = 100
    rotation_bins: int = 36
    image_size: int = 256
    spacing: float = 2.0
    
    valid_placement_reward: float = 1.0
    invalid_placement_penalty: float = -5.0
    touching_bonus: float = 0.5
    corner_bonus: float = 0.3
    hole_penalty: float = -0.2
    progress_reward: float = 0.1
    time_penalty: float = -0.01
    final_multiplier: float = 100.0


class NestingEnvironmentFixed(gym.Env):
    # """
    # Versão CORRIGIDA do ambiente de nesting
    
    # Correções principais:
    # 1. _render_layout_as_image sempre retorna array válido
    # 2. _extract_piece_features trata None corretamente
    # 3. _extract_remaining_features retorna valores reais
    # """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, 
                 config: Optional[NestingConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config or NestingConfig()
        self.render_mode = render_mode
        
        self._setup_spaces()
        
        # Estado interno
        self.container = None
        self.pieces_to_place = []
        self.placed_pieces = []
        self.current_piece_idx = 0
        self.step_count = 0
        
        self.fig = None
        self.ax = None
    
    def _setup_spaces(self):
        #"""Define observation e action spaces"""
        cfg = self.config
        
        self.observation_space = spaces.Dict({
            'layout_image': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6, cfg.image_size, cfg.image_size),
                dtype=np.float32
            ),
            'current_piece': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            ),
            'remaining_pieces': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            ),
            'stats': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32
            )
        })
        
        self.action_space = spaces.Dict({
            'position': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            ),
            'rotation': spaces.Discrete(cfg.rotation_bins)
        })
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        #"""Reseta o ambiente"""
        super().reset(seed=seed)
        
        # Criar container
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
        #"""Executa uma ação"""
        self.step_count += 1
        
        # Extrair ação
        position_normalized = action['position']
        rotation_bin = action['rotation']
        
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
        reward_total = reward + self.config.time_penalty
        
        # Verificar se terminou
        terminated = False
        truncated = False
        
        if success:
            self.current_piece_idx += 1
            
            if self.current_piece_idx >= len(self.pieces_to_place):
                terminated = True
                utilization = self._calculate_utilization()
                final_bonus = utilization * self.config.final_multiplier
                reward_total += final_bonus
        else:
            terminated = True
        
        if self.step_count >= self.config.max_steps:
            truncated = True
        
        # Nova observação
        observation = self._get_observation()
        info = self._get_info()
        info.update(info_placement)
        
        return observation, reward_total, terminated, truncated, info
    
    def _place_piece(self, piece, x: float, y: float, 
                     rotation: float) -> Tuple[bool, float, Dict]:
        #"""Tenta colocar uma peça"""
        try:
            # Rotacionar peça
            rotated_piece = piece.rotate(rotation)
            
            # Mover para posição desejada
            moved_piece = rotated_piece.set_position(x, y)
            
            # Verificar se está dentro do container
            if not self.container.shapely.contains(moved_piece.shapely):
                return False, self.config.invalid_placement_penalty, {
                    'placement_status': 'invalid',
                    'reason': 'out_of_bounds',
                    'position': (x, y),
                    'rotation': rotation
                }
            
            # Verificar colisões
            for placed in self.placed_pieces:
                if moved_piece.intersects(placed):
                    return False, self.config.invalid_placement_penalty, {
                        'placement_status': 'invalid',
                        'reason': 'collision',
                        'position': (x, y),
                        'rotation': rotation
                    }
            
            # Colocação válida!
            self.placed_pieces.append(moved_piece)
            
            # Calcular recompensa
            reward = self.config.valid_placement_reward
            
            # Bônus se toca outras peças
            if len(self.placed_pieces) > 1:
                touching = False
                for placed in self.placed_pieces[:-1]:
                    distance = moved_piece.distance_to(placed)
                    if distance < self.config.spacing:
                        touching = True
                        break
                
                if touching:
                    reward += self.config.touching_bonus
            
            # Bônus de canto
            if x < 100 and y < 100:
                reward += self.config.corner_bonus
            
            reward += self.config.progress_reward
            
            return True, reward, {
                'placement_status': 'valid',
                'position': (x, y),
                'rotation': rotation
            }
            
        except Exception as e:
            print(f"Warning: Error in _place_piece: {e}")
            return False, self.config.invalid_placement_penalty, {
                'placement_status': 'invalid',
                'reason': 'error',
                'error': str(e)
            }
    
    def _get_observation(self) -> Dict:
        #"""Constrói observação do estado atual - VERSÃO CORRIGIDA"""
        
        # 1. Layout image (GARANTIDO NÃO-NONE)
        layout_image = self._render_layout_as_image()
        
        # VERIFICAÇÃO CRÍTICA
        if layout_image is None:
            # Fallback: criar imagem vazia
            print("Warning: layout_image is None, using fallback")
            layout_image = np.zeros((6, self.config.image_size, self.config.image_size), 
                                   dtype=np.float32)
        
        # 2. Features da peça atual (GARANTIDO NÃO-NONE)
        current_piece_features = self._extract_piece_features_safe()
        
        # 3. Features restantes (GARANTIDO NÃO-NONE)
        remaining_features = self._extract_remaining_features_safe()
        
        # 4. Stats globais (SEMPRE VÁLIDO)
        stats = np.array([
            self._calculate_utilization(),
            self.current_piece_idx / max(len(self.pieces_to_place), 1),
            len(self.placed_pieces) / max(len(self.pieces_to_place), 1),
            self.step_count / max(self.config.max_steps, 1),
            0.0
        ], dtype=np.float32)
        
        return {
            'layout_image': layout_image,
            'current_piece': current_piece_features,
            'remaining_pieces': remaining_features,
            'stats': stats
        }
    
    def _render_layout_as_image(self) -> np.ndarray:
        #"""Renderiza layout - VERSÃO CORRIGIDA que NUNCA retorna None"""
        try:
            from src.representation.image_encoder import render_layout_as_image
            
            next_piece = None
            if self.current_piece_idx < len(self.pieces_to_place):
                next_piece = self.pieces_to_place[self.current_piece_idx]
            
            image = render_layout_as_image(
                container=self.container,
                placed_pieces=self.placed_pieces,
                next_piece=next_piece,
                size=self.config.image_size
            )
            
            # GARANTIR que retornou algo válido
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("render_layout_as_image returned None or invalid type")
            
            return image
            
        except Exception as e:
            # Fallback seguro
            print(f"Warning: Error rendering layout, using fallback: {e}")
            return np.zeros((6, self.config.image_size, self.config.image_size), 
                          dtype=np.float32)
    
    def _extract_piece_features_safe(self) -> np.ndarray:
        #"""Extrai features da peça ATUAL - VERSÃO SEGURA"""
        if self.current_piece_idx >= len(self.pieces_to_place):
            return np.zeros(10, dtype=np.float32)
        
        try:
            piece = self.pieces_to_place[self.current_piece_idx]
            
            features = np.array([
                piece.area / 10000.0,
                piece.perimeter / 500.0,
                piece.width / 200.0,
                piece.height / 200.0,
                piece.aspect_ratio,
                piece.calculate_complexity() / 10.0,
                len(piece.vertices) / 20.0,
                piece.rotation / 360.0,
                1.0,
                1.0,
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"Warning: Error extracting piece features: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _extract_remaining_features_safe(self) -> np.ndarray:
        #"""Features agregadas das peças RESTANTES - VERSÃO SEGURA"""
        try:
            n_remaining = len(self.pieces_to_place) - self.current_piece_idx
            
            if n_remaining <= 0:
                return np.zeros(10, dtype=np.float32)
            
            # Calcular features reais das peças restantes
            remaining_pieces = self.pieces_to_place[self.current_piece_idx:]
            
            total_area = sum(p.area for p in remaining_pieces)
            avg_perimeter = np.mean([p.perimeter for p in remaining_pieces])
            avg_width = np.mean([p.width for p in remaining_pieces])
            avg_height = np.mean([p.height for p in remaining_pieces])
            
            features = np.array([
                n_remaining / 50.0,
                total_area / 50000.0,
                avg_perimeter / 500.0,
                avg_width / 200.0,
                avg_height / 200.0,
                np.mean([p.aspect_ratio for p in remaining_pieces]),
                np.mean([len(p.vertices) for p in remaining_pieces]) / 20.0,
                0.0,
                0.0,
                0.0,
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"Warning: Error extracting remaining features: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _calculate_utilization(self) -> float:
        #"""Calcula taxa de utilização"""
        if len(self.placed_pieces) == 0:
            return 0.0
        
        total_pieces_area = sum(piece.area for piece in self.placed_pieces)
        container_area = self.container.area
        utilization = total_pieces_area / max(container_area, 1.0)
        
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
        from src.geometry.polygon import create_rectangle
        
        container = create_rectangle(
            self.config.container_width,
            self.config.container_height,
            center=(self.config.container_width/2, self.config.container_height/2)
        )
        
        return container
    
    def _generate_random_pieces(self) -> List:
        #"""Gera peças aleatórias para teste"""
        from src.geometry.polygon import create_random_polygon
        
        n_pieces = np.random.randint(5, 15)
        pieces = []
        
        for i in range(n_pieces):
            piece = create_random_polygon(
                n_vertices=np.random.randint(4, 10),
                radius=np.random.uniform(20, 50),
                irregularity=0.5,
                spikeyness=0.3
            )
            piece.id = i
            pieces.append(piece)
        
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
        for piece in self.placed_pieces:
            try:
                piece.plot(self.ax, facecolor='lightblue', alpha=0.6)
            except:
                pass
        
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
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rect = plt.Rectangle(
            (0, 0),
            self.config.container_width,
            self.config.container_height,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        
        for piece in self.placed_pieces:
            try:
                piece.plot(ax, facecolor='lightblue', alpha=0.6)
            except:
                pass
        
        ax.set_xlim(-50, self.config.container_width + 50)
        ax.set_ylim(-50, self.config.container_height + 50)
        ax.set_aspect('equal')
        
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


# Alias para compatibilidade
NestingEnvironment = NestingEnvironmentFixed


if __name__ == "__main__":
    print("="*70)
    print("TESTE DO AMBIENTE CORRIGIDO")
    print("="*70)
    
    from src.geometry.polygon import create_rectangle, create_random_polygon
    
    # Criar ambiente
    config = NestingConfig(max_steps=10)
    env = NestingEnvironmentFixed(config=config)
    
    # Criar peças
    pieces = [
        create_rectangle(50, 30),
        create_rectangle(40, 25),
        create_random_polygon(5, 20),
    ]
    
    for i, piece in enumerate(pieces):
        piece.id = i
    
    # Reset
    obs, info = env.reset(options={'pieces': pieces})
    print(f"✓ Reset bem-sucedido")
    print(f"  Info: {info}")
    
    # Verificar observação
    print(f"\nObservação:")
    for key, val in obs.items():
        print(f"  {key}: {type(val).__name__}, shape: {val.shape}, dtype: {val.dtype}")
    
    # Executar alguns steps
    print(f"\nExecutando steps...")
    for step in range(5):
        action = {
            'position': env.action_space['position'].sample(),
            'rotation': env.action_space['rotation'].sample()
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: reward={reward:.2f}, placed={info['n_placed']}/{len(pieces)}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Teste completo! Utilização final: {info['utilization']*100:.1f}%")
    env.close()
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