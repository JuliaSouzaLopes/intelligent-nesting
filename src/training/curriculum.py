# """
# src/training/curriculum.py

# Curriculum Learning: Aumenta progressivamente a dificuldade do treinamento.

# Estratégias:
# 1. Número de peças: 3 → 50
# 2. Complexidade das peças: retângulos → polígonos irregulares
# 3. Tamanho do container: pequeno → grande
# """

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    #"""Define uma etapa do curriculum"""
    name: str
    n_pieces_range: Tuple[int, int]
    piece_complexity: str  # 'rectangles', 'regular', 'mixed', 'irregular'
    container_size: float  # Multiplicador do tamanho base
    rotation_difficulty: str  # 'none', 'discrete', 'continuous'
    success_threshold: float  # Utilização mínima para avançar


class CurriculumScheduler:
    # """
    # Gerencia o curriculum de treinamento.
    
    # Aumenta dificuldade baseado no desempenho do agente.
    # """
    
    def __init__(self, config: dict):
        # """
        # Args:
        #     config: Configuração do curriculum
        # """
        self.config = config
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_successes = 0
        
        # Definir estágios
        self.stages = self._create_stages()
        
        # Histórico
        self.history = []
    
    def _create_stages(self) -> List[CurriculumStage]:
        #"""Cria estágios do curriculum"""
        stages = [
            # Estágio 1: Muito fácil
            CurriculumStage(
                name="Stage 1: Retângulos simples",
                n_pieces_range=(3, 5),
                piece_complexity='rectangles',
                container_size=1.5,
                rotation_difficulty='none',
                success_threshold=0.60
            ),
            
            # Estágio 2: Adicionar rotação
            CurriculumStage(
                name="Stage 2: Retângulos com rotação",
                n_pieces_range=(4, 7),
                piece_complexity='rectangles',
                container_size=1.3,
                rotation_difficulty='discrete',
                success_threshold=0.65
            ),
            
            # Estágio 3: Mais peças
            CurriculumStage(
                name="Stage 3: Mais retângulos",
                n_pieces_range=(7, 12),
                piece_complexity='rectangles',
                container_size=1.2,
                rotation_difficulty='discrete',
                success_threshold=0.70
            ),
            
            # Estágio 4: Polígonos regulares
            CurriculumStage(
                name="Stage 4: Polígonos regulares",
                n_pieces_range=(5, 10),
                piece_complexity='regular',
                container_size=1.2,
                rotation_difficulty='discrete',
                success_threshold=0.65
            ),
            
            # Estágio 5: Mix
            CurriculumStage(
                name="Stage 5: Mix de peças",
                n_pieces_range=(8, 15),
                piece_complexity='mixed',
                container_size=1.1,
                rotation_difficulty='discrete',
                success_threshold=0.70
            ),
            
            # Estágio 6: Irregular
            CurriculumStage(
                name="Stage 6: Polígonos irregulares",
                n_pieces_range=(10, 20),
                piece_complexity='irregular',
                container_size=1.0,
                rotation_difficulty='discrete',
                success_threshold=0.75
            ),
            
            # Estágio 7: Difícil
            CurriculumStage(
                name="Stage 7: Muitas peças irregulares",
                n_pieces_range=(20, 35),
                piece_complexity='irregular',
                container_size=1.0,
                rotation_difficulty='discrete',
                success_threshold=0.75
            ),
            
            # Estágio 8: Muito difícil
            CurriculumStage(
                name="Stage 8: Máximo desafio",
                n_pieces_range=(30, 50),
                piece_complexity='irregular',
                container_size=1.0,
                rotation_difficulty='continuous',
                success_threshold=0.80
            ),
        ]
        
        return stages
    
    def get_current_stage(self) -> CurriculumStage:
        #"""Retorna estágio atual"""
        return self.stages[self.current_stage]
    
    def should_advance(self) -> bool:
        # """
        # Verifica se deve avançar para próximo estágio.
        
        # Critério: Taxa de sucesso nas últimas N tentativas
        # """
        min_episodes = self.config.get('min_episodes_per_stage', 100)
        
        if self.stage_episodes < min_episodes:
            return False
        
        success_rate = self.stage_successes / max(self.stage_episodes, 1)
        threshold = self.stages[self.current_stage].success_threshold
        
        return success_rate >= threshold
    
    def advance_stage(self):
        #"""Avança para próximo estágio"""
        if self.current_stage < len(self.stages) - 1:
            old_stage = self.current_stage
            self.current_stage += 1
            
            # Reset contadores
            self.stage_episodes = 0
            self.stage_successes = 0
            
            print("="*70)
            print(f"🎓 CURRICULUM ADVANCEMENT!")
            print(f"   {self.stages[old_stage].name}")
            print(f"   ↓")
            print(f"   {self.stages[self.current_stage].name}")
            print("="*70)
    
    def update(self, utilization: float):
        # """
        # Atualiza curriculum com resultado de episódio.
        
        # Args:
        #     utilization: Utilização alcançada (0-1)
        # """
        self.stage_episodes += 1
        
        # Considerar sucesso se atingiu threshold
        if utilization >= self.stages[self.current_stage].success_threshold:
            self.stage_successes += 1
        
        # Salvar histórico
        self.history.append({
            'stage': self.current_stage,
            'episode': self.stage_episodes,
            'utilization': utilization
        })
        
        # Verificar se deve avançar
        if self.should_advance():
            self.advance_stage()
    
    def get_problem_config(self) -> Dict:
        # """
        # Gera configuração do problema para o estágio atual.
        
        # Returns:
        #     Dict com configuração para gerar problema
        # """
        stage = self.get_current_stage()
        
        # Número de peças (aleatório no range)
        n_pieces = np.random.randint(
            stage.n_pieces_range[0],
            stage.n_pieces_range[1] + 1
        )
        
        return {
            'n_pieces': n_pieces,
            'piece_complexity': stage.piece_complexity,
            'container_multiplier': stage.container_size,
            'rotation_difficulty': stage.rotation_difficulty
        }
    
    def generate_pieces(self, config: Dict) -> List:
        # """
        # Gera peças baseado na configuração do curriculum.
        
        # Args:
        #     config: Configuração retornada por get_problem_config()
        
        # Returns:
        #     Lista de peças (Polygons)
        # """
        from src.geometry.polygon import (
            create_rectangle, 
            create_regular_polygon,
            create_random_polygon
        )
        
        n_pieces = config['n_pieces']
        complexity = config['piece_complexity']
        
        pieces = []
        
        for i in range(n_pieces):
            if complexity == 'rectangles':
                # Apenas retângulos
                width = np.random.uniform(30, 80)
                height = np.random.uniform(20, 60)
                piece = create_rectangle(width, height)
                
            elif complexity == 'regular':
                # Polígonos regulares
                n_sides = np.random.choice([4, 5, 6, 8])
                radius = np.random.uniform(20, 40)
                piece = create_regular_polygon(n_sides, radius)
                
            elif complexity == 'mixed':
                # Mix: 50% retângulos, 50% regulares
                if np.random.rand() < 0.5:
                    width = np.random.uniform(30, 70)
                    height = np.random.uniform(20, 50)
                    piece = create_rectangle(width, height)
                else:
                    n_sides = np.random.choice([5, 6, 8])
                    radius = np.random.uniform(20, 35)
                    piece = create_regular_polygon(n_sides, radius)
                    
            elif complexity == 'irregular':
                # Polígonos irregulares
                n_vertices = np.random.randint(5, 10)
                radius = np.random.uniform(20, 40)
                irregularity = np.random.uniform(0.4, 0.8)
                spikeyness = np.random.uniform(0.3, 0.6)
                
                piece = create_random_polygon(
                    n_vertices=n_vertices,
                    radius=radius,
                    irregularity=irregularity,
                    spikeyness=spikeyness
                )
            else:
                # Fallback: retângulo
                piece = create_rectangle(50, 30)
            
            piece.id = i
            pieces.append(piece)
        
        return pieces
    
    def get_stats(self) -> Dict:
        #"""Retorna estatísticas do curriculum"""
        stage = self.get_current_stage()
        
        return {
            'current_stage': self.current_stage,
            'stage_name': stage.name,
            'stage_episodes': self.stage_episodes,
            'stage_successes': self.stage_successes,
            'success_rate': self.stage_successes / max(self.stage_episodes, 1),
            'total_stages': len(self.stages)
        }
    
    def save_state(self, path: str):
        """Salva estado do curriculum"""
        import pickle
        
        state = {
            'current_stage': self.current_stage,
            'stage_episodes': self.stage_episodes,
            'stage_successes': self.stage_successes,
            'history': self.history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str):
        """Carrega estado do curriculum"""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.current_stage = state['current_stage']
        self.stage_episodes = state['stage_episodes']
        self.stage_successes = state['stage_successes']
        self.history = state['history']


# =============================================================================
# Exemplo de Uso
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE: CURRICULUM SCHEDULER")
    print("="*70)
    
    # Criar scheduler
    config = {
        'min_episodes_per_stage': 10
    }
    
    curriculum = CurriculumScheduler(config)
    
    print(f"\n✓ Curriculum criado com {len(curriculum.stages)} estágios")
    
    # Mostrar todos os estágios
    print("\n" + "="*70)
    print("ESTÁGIOS DO CURRICULUM")
    print("="*70)
    
    for i, stage in enumerate(curriculum.stages):
        print(f"\n{i+1}. {stage.name}")
        print(f"   Peças: {stage.n_pieces_range[0]}-{stage.n_pieces_range[1]}")
        print(f"   Complexidade: {stage.piece_complexity}")
        print(f"   Threshold: {stage.success_threshold*100:.0f}%")
    
    # Simular progresso
    print("\n" + "="*70)
    print("SIMULAÇÃO DE PROGRESSO")
    print("="*70)
    
    for episode in range(50):
        # Simular utilização (melhora gradualmente)
        utilization = 0.5 + 0.01 * episode + np.random.uniform(-0.05, 0.05)
        utilization = np.clip(utilization, 0, 1)
        
        curriculum.update(utilization)
        
        if episode % 10 == 0:
            stats = curriculum.get_stats()
            print(f"\nEpisode {episode}:")
            print(f"  Stage: {stats['stage_name']}")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            print(f"  Utilization: {utilization*100:.1f}%")
    
    # Gerar problemas
    print("\n" + "="*70)
    print("GERAÇÃO DE PROBLEMAS")
    print("="*70)
    
    for i in range(3):
        config = curriculum.get_problem_config()
        pieces = curriculum.generate_pieces(config)
        
        print(f"\nProblema {i+1}:")
        print(f"  Peças: {len(pieces)}")
        print(f"  Complexidade: {config['piece_complexity']}")
        print(f"  Tipos: {[type(p).__name__ for p in pieces[:3]]}...")
    
    print("\n" + "="*70)
    print("✓ CURRICULUM SCHEDULER IMPLEMENTADO!")
    print("="*70)