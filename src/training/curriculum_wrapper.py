# """
# curriculum_wrapper.py
# Wrapper seguro que sempre funciona
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.curriculum import CurriculumScheduler as _OriginalScheduler
import numpy as np


class SafeCurriculumScheduler(_OriginalScheduler):
    # """
    # Wrapper seguro do CurriculumScheduler que garante inicialização correta
    # """
    
    def __init__(self, config: dict):
        """Inicializa com verificações extras"""
        try:
            # Chamar __init__ original
            super().__init__(config)
            
            # Verificar e corrigir atributos faltando
            if not hasattr(self, 'current_stage'):
                print("⚠️  Corrigindo: current_stage faltando")
                self.current_stage = 0
            
            if not hasattr(self, 'stage_episodes'):
                print("⚠️  Corrigindo: stage_episodes faltando")
                self.stage_episodes = 0
            
            if not hasattr(self, 'stage_successes'):
                print("⚠️  Corrigindo: stage_successes faltando")
                self.stage_successes = 0
            
            if not hasattr(self, 'stages'):
                print("⚠️  Corrigindo: stages faltando")
                self.stages = self._create_stages()
            
            if not hasattr(self, 'history'):
                print("⚠️  Corrigindo: history faltando")
                self.history = []
                
        except Exception as e:
            print(f"❌ ERRO no __init__ original: {e}")
            print("   Inicializando manualmente...")
            
            # Inicializar manualmente
            self.config = config
            self.current_stage = 0
            self.stage_episodes = 0
            self.stage_successes = 0
            self.stages = self._create_stages()
            self.history = []
    
    def get_current_stage_safe(self):
        """Versão segura de get_current_stage"""
        if not hasattr(self, 'stages') or not hasattr(self, 'current_stage'):
            raise RuntimeError("CurriculumScheduler não inicializado corretamente")
        
        return self.stages[self.current_stage]
    
    def get_problem_config_safe(self):
        """Versão segura de get_problem_config"""
        stage = self.get_current_stage_safe()
        
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


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE: SafeCurriculumScheduler")
    print("="*70)
    
    # Criar curriculum seguro
    curriculum = SafeCurriculumScheduler({'min_episodes_per_stage': 10})
    
    print(f"\n✅ Curriculum criado")
    print(f"   current_stage: {curriculum.current_stage}")
    print(f"   n_stages: {len(curriculum.stages)}")
    
    # Obter estágio
    stage = curriculum.get_current_stage_safe()
    print(f"\n✅ Stage atual: {stage.name}")
    
    # Obter config
    problem_config = curriculum.get_problem_config_safe()
    print(f"\n✅ Problem config: {problem_config}")
    
    # Gerar peças
    pieces = curriculum.generate_pieces(problem_config)
    print(f"\n✅ Peças geradas: {len(pieces)}")
    
    print("\n" + "="*70)
    print("✅ TUDO FUNCIONOU!")
    print("="*70)