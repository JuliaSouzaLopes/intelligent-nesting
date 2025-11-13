# """
# src/training/curriculum.py

# Curriculum Learning: Aumenta gradualmente a dificuldade do problema.

# Progressão:
# - Número de peças: 3 → 50
# - Complexidade: retângulos → polígonos irregulares
# - Tamanho do container: grande → realista
# """

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.geometry.polygon import (
    Polygon, create_rectangle, create_regular_polygon, create_random_polygon
)


@dataclass
class CurriculumStage:
    #"""Define um estágio do curriculum"""
    stage_id: int
    name: str
    n_pieces_min: int
    n_pieces_max: int
    complexity: str  # 'rectangles', 'regular', 'mixed', 'irregular'
    container_scale: float  # Multiplica tamanho base
    piece_size_range: Tuple[float, float]  # (min, max) para dimensões
    rotation_enabled: bool
    min_utilization_target: float  # Utilização mínima para avançar


class CurriculumScheduler:
    # """
    # Gerencia progressão do curriculum learning.
    
    # Aumenta dificuldade baseado em performance.
    # """
    
    def __init__(self,
                 base_container_size: Tuple[float, float] = (1000, 600),
                 auto_advance: bool = True,
                 advancement_threshold: float = 0.6):
        # """
        # Args:
        #     base_container_size: Tamanho base do container (width, height)
        #     auto_advance: Se True, avança automaticamente ao atingir threshold
        #     advancement_threshold: Utilização mínima para avançar de estágio
        # """
        self.base_container_size = base_container_size
        self.auto_advance = auto_advance
        self.advancement_threshold = advancement_threshold
        
        # Definir estágios
        self.stages = self._define_stages()
        
        # Estado atual
        self.current_stage_idx = 0
        self.episode_count = 0
        self.stage_episode_count = 0
        
        # Estatísticas
        self.stage_utilizations: List[float] = []
        self.stage_success_rate: List[float] = []
    
    def _define_stages(self) -> List[CurriculumStage]:
        #"""Define os estágios do curriculum"""
        stages = [
            # Estágio 1: Muito Fácil - Retângulos pequenos
            CurriculumStage(
                stage_id=1,
                name="Warm-up: Poucos Retângulos",
                n_pieces_min=3,
                n_pieces_max=5,
                complexity='rectangles',
                container_scale=1.5,  # Container maior
                piece_size_range=(30, 60),
                rotation_enabled=False,
                min_utilization_target=0.50
            ),
            
            # Estágio 2: Fácil - Mais retângulos
            CurriculumStage(
                stage_id=2,
                name="Basic: Retângulos com Rotação",
                n_pieces_min=5,
                n_pieces_max=8,
                complexity='rectangles',
                container_scale=1.3,
                piece_size_range=(25, 55),
                rotation_enabled=True,
                min_utilization_target=0.55
            ),
            
            # Estágio 3: Médio - Polígonos regulares
            CurriculumStage(
                stage_id=3,
                name="Intermediate: Polígonos Regulares",
                n_pieces_min=8,
                n_pieces_max=12,
                complexity='regular',
                container_scale=1.2,
                piece_size_range=(20, 50),
                rotation_enabled=True,
                min_utilization_target=0.60
            ),
            
            # Estágio 4: Médio-Difícil - Mix
            CurriculumStage(
                stage_id=4,
                name="Advanced: Mix de Formas",
                n_pieces_min=12,
                n_pieces_max=18,
                complexity='mixed',
                container_scale=1.1,
                piece_size_range=(20, 45),
                rotation_enabled=True,
                min_utilization_target=0.65
            ),
            
            # Estágio 5: Difícil - Irregulares
            CurriculumStage(
                stage_id=5,
                name="Expert: Polígonos Irregulares",
                n_pieces_min=18,
                n_pieces_max=25,
                complexity='irregular',
                container_scale=1.0,
                piece_size_range=(15, 40),
                rotation_enabled=True,
                min_utilization_target=0.70
            ),
            
            # Estágio 6: Muito Difícil - Produção
            CurriculumStage(
                stage_id=6,
                name="Production: Problema Realista",
                n_pieces_min=25,
                n_pieces_max=50,
                complexity='irregular',
                container_scale=1.0,
                piece_size_range=(10, 35),
                rotation_enabled=True,
                min_utilization_target=0.75
            ),
        ]
        return stages
    
    @property
    def current_stage(self) -> CurriculumStage:
        #"""Retorna estágio atual"""
        return self.stages[self.current_stage_idx]
    
    def generate_problem(self) -> Dict:
        # """
        # Gera um problema de acordo com o estágio atual.
        
        # Returns:
        #     Dict contendo:
        #         - pieces: Lista de Polygon
        #         - container_size: (width, height)
        #         - stage_info: Informações do estágio
        # """
        stage = self.current_stage
        
        # Número de peças
        n_pieces = np.random.randint(stage.n_pieces_min, stage.n_pieces_max + 1)
        
        # Gerar peças
        pieces = self._generate_pieces(
            n_pieces=n_pieces,
            complexity=stage.complexity,
            size_range=stage.piece_size_range,
            rotation_enabled=stage.rotation_enabled
        )
        
        # Tamanho do container
        base_w, base_h = self.base_container_size
        container_size = (
            base_w * stage.container_scale,
            base_h * stage.container_scale
        )
        
        return {
            'pieces': pieces,
            'container_size': container_size,
            'stage_info': {
                'stage_id': stage.stage_id,
                'stage_name': stage.name,
                'n_pieces': n_pieces,
                'complexity': stage.complexity
            }
        }
    
    def _generate_pieces(self,
                        n_pieces: int,
                        complexity: str,
                        size_range: Tuple[float, float],
                        rotation_enabled: bool) -> List[Polygon]:
        #"""Gera peças de acordo com complexidade"""
        pieces = []
        min_size, max_size = size_range
        
        for i in range(n_pieces):
            # Tamanho aleatório
            size = np.random.uniform(min_size, max_size)
            
            # Criar peça baseado na complexidade
            if complexity == 'rectangles':
                # Retângulos com aspect ratio variado
                aspect = np.random.uniform(0.5, 2.0)
                width = size
                height = size / aspect
                piece = create_rectangle(width, height)
                
            elif complexity == 'regular':
                # Polígonos regulares (triângulo a octógono)
                n_sides = np.random.choice([3, 4, 5, 6, 8])
                piece = create_regular_polygon(n_sides, radius=size/2)
                
            elif complexity == 'mixed':
                # 50% retângulos, 50% regulares
                if np.random.rand() < 0.5:
                    aspect = np.random.uniform(0.5, 2.0)
                    width = size
                    height = size / aspect
                    piece = create_rectangle(width, height)
                else:
                    n_sides = np.random.choice([3, 4, 5, 6])
                    piece = create_regular_polygon(n_sides, radius=size/2)
                    
            elif complexity == 'irregular':
                # Polígonos irregulares
                n_vertices = np.random.randint(5, 10)
                irregularity = np.random.uniform(0.3, 0.7)
                spikeyness = np.random.uniform(0.2, 0.5)
                
                piece = create_random_polygon(
                    n_vertices=n_vertices,
                    radius=size/2,
                    irregularity=irregularity,
                    spikeyness=spikeyness
                )
            else:
                raise ValueError(f"Unknown complexity: {complexity}")
            
            # Aplicar rotação inicial se habilitada
            if rotation_enabled:
                initial_rotation = np.random.uniform(0, 360)
                piece = piece.rotate(initial_rotation)
            
            piece.id = i
            pieces.append(piece)
        
        return pieces
    
    def record_episode(self, utilization: float, success: bool):
        # """
        # Registra resultado de um episódio.
        
        # Args:
        #     utilization: Taxa de utilização alcançada
        #     success: Se conseguiu colocar todas as peças
        # """
        self.episode_count += 1
        self.stage_episode_count += 1
        
        self.stage_utilizations.append(utilization)
        self.stage_success_rate.append(1.0 if success else 0.0)
        
        # Limitar histórico
        max_history = 100
        if len(self.stage_utilizations) > max_history:
            self.stage_utilizations = self.stage_utilizations[-max_history:]
            self.stage_success_rate = self.stage_success_rate[-max_history:]
    
    def should_advance(self, min_episodes: int = 50) -> bool:
        # """
        # Verifica se deve avançar para próximo estágio.
        
        # Args:
        #     min_episodes: Mínimo de episódios antes de poder avançar
            
        # Returns:
        #     True se deve avançar
        # """
        if not self.auto_advance:
            return False
        
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Já no último estágio
        
        if self.stage_episode_count < min_episodes:
            return False  # Precisa mais episódios
        
        # Calcular performance recente
        recent_window = 50
        if len(self.stage_utilizations) < recent_window:
            return False
        
        recent_util = np.mean(self.stage_utilizations[-recent_window:])
        recent_success = np.mean(self.stage_success_rate[-recent_window:])
        
        # Critérios para avanço
        target_util = self.current_stage.min_utilization_target
        
        should_advance = (
            recent_util >= target_util and
            recent_success >= 0.8  # 80% de sucesso
        )
        
        return should_advance
    
    def advance_stage(self):
        #"""Avança para próximo estágio"""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.stage_episode_count = 0
            self.stage_utilizations = []
            self.stage_success_rate = []
            
            print(f"\n{'='*70}")
            print(f"🎓 CURRICULUM ADVANCE!")
            print(f"{'='*70}")
            print(f"Novo estágio: {self.current_stage.name}")
            print(f"Stage {self.current_stage.stage_id}/{len(self.stages)}")
            print(f"{'='*70}\n")
    
    def get_stats(self) -> Dict:
        #"""Retorna estatísticas do curriculum"""
        recent_window = min(50, len(self.stage_utilizations))
        
        if recent_window > 0:
            recent_util = np.mean(self.stage_utilizations[-recent_window:])
            recent_success = np.mean(self.stage_success_rate[-recent_window:])
        else:
            recent_util = 0.0
            recent_success = 0.0
        
        return {
            'current_stage': self.current_stage.stage_id,
            'stage_name': self.current_stage.name,
            'total_episodes': self.episode_count,
            'stage_episodes': self.stage_episode_count,
            'recent_utilization': recent_util,
            'recent_success_rate': recent_success,
            'target_utilization': self.current_stage.min_utilization_target,
            'can_advance': self.should_advance()
        }
    
    def reset(self):
        #"""Reseta curriculum para estágio inicial"""
        self.current_stage_idx = 0
        self.episode_count = 0
        self.stage_episode_count = 0
        self.stage_utilizations = []
        self.stage_success_rate = []


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE: CURRICULUM LEARNING")
    print("="*70)
    
    # Criar scheduler
    curriculum = CurriculumScheduler(
        base_container_size=(1000, 600),
        auto_advance=True,
        advancement_threshold=0.6
    )
    
    print(f"\n✓ Curriculum criado com {len(curriculum.stages)} estágios")
    print(f"Estágio inicial: {curriculum.current_stage.name}")
    
    # Gerar alguns problemas
    print("\n" + "="*70)
    print("GERANDO PROBLEMAS")
    print("="*70)
    
    for i in range(5):
        problem = curriculum.generate_problem()
        
        print(f"\nProblema {i+1}:")
        print(f"  Stage: {problem['stage_info']['stage_name']}")
        print(f"  Peças: {problem['stage_info']['n_pieces']}")
        print(f"  Complexidade: {problem['stage_info']['complexity']}")
        print(f"  Container: {problem['container_size']}")
        print(f"  Exemplo peça 0: {problem['pieces'][0]}")
    
    # Simular progressão
    print("\n" + "="*70)
    print("SIMULANDO PROGRESSÃO")
    print("="*70)
    
    for episode in range(200):
        # Gerar problema
        problem = curriculum.generate_problem()
        
        # Simular resultado (performance melhora com tempo)
        base_util = 0.4 + (episode / 200) * 0.3
        noise = np.random.uniform(-0.1, 0.1)
        utilization = np.clip(base_util + noise, 0, 1)
        success = utilization > 0.5
        
        # Registrar
        curriculum.record_episode(utilization, success)
        
        # Verificar avanço
        if curriculum.should_advance():
            curriculum.advance_stage()
        
        # Log a cada 25 episódios
        if (episode + 1) % 25 == 0:
            stats = curriculum.get_stats()
            print(f"\nEpisode {episode + 1}:")
            print(f"  Stage: {stats['stage_name']}")
            print(f"  Recent Util: {stats['recent_utilization']:.2%}")
            print(f"  Recent Success: {stats['recent_success_rate']:.2%}")
            print(f"  Target: {stats['target_utilization']:.2%}")
            print(f"  Can Advance: {stats['can_advance']}")
    
    print("\n" + "="*70)
    print("✓ CURRICULUM LEARNING IMPLEMENTADO!")
    print("="*70)