# """
# test_curriculum_init.py
# Teste detalhado do __init__ do CurriculumScheduler
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("TESTE DETALHADO: CurriculumScheduler.__init__")
print("="*70)

# 1. Import
print("\n1. Importando...")
try:
    from src.training.curriculum import CurriculumScheduler, CurriculumStage
    print("   ✅ Import OK")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Criar instância COM debug
print("\n2. Criando CurriculumScheduler COM debug...")

class CurriculumSchedulerDebug(CurriculumScheduler):
    """Versão com debug do __init__"""
    
    def __init__(self, config: dict):
        print("   [DEBUG] Entrando em __init__")
        print(f"   [DEBUG] config = {config}")
        
        try:
            print("   [DEBUG] Setando self.config...")
            self.config = config
            print("   [DEBUG] ✓ self.config OK")
            
            print("   [DEBUG] Setando self.current_stage = 0...")
            self.current_stage = 0
            print("   [DEBUG] ✓ self.current_stage OK")
            
            print("   [DEBUG] Setando self.stage_episodes = 0...")
            self.stage_episodes = 0
            print("   [DEBUG] ✓ self.stage_episodes OK")
            
            print("   [DEBUG] Setando self.stage_successes = 0...")
            self.stage_successes = 0
            print("   [DEBUG] ✓ self.stage_successes OK")
            
            print("   [DEBUG] Chamando self._create_stages()...")
            self.stages = self._create_stages()
            print(f"   [DEBUG] ✓ self.stages OK, len={len(self.stages)}")
            
            print("   [DEBUG] Setando self.history = []...")
            self.history = []
            print("   [DEBUG] ✓ self.history OK")
            
            print("   [DEBUG] __init__ completo com sucesso!")
            
        except Exception as e:
            print(f"   [DEBUG] ❌ ERRO durante __init__: {e}")
            import traceback
            traceback.print_exc()
            raise

try:
    print("\nCriando instância com debug...")
    curriculum_debug = CurriculumSchedulerDebug({'min_episodes_per_stage': 10})
    print("   ✅ Instância criada com sucesso!")
    
except Exception as e:
    print(f"   ❌ ERRO ao criar: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Verificar atributos
print("\n3. Verificando atributos...")
attrs_to_check = ['config', 'current_stage', 'stage_episodes', 'stage_successes', 'stages', 'history']

for attr in attrs_to_check:
    has_it = hasattr(curriculum_debug, attr)
    if has_it:
        value = getattr(curriculum_debug, attr)
        print(f"   ✅ {attr:20s} = {type(value).__name__:15s} {str(value)[:50]}")
    else:
        print(f"   ❌ {attr:20s} = MISSING!")

# 4. Criar instância normal (sem debug)
print("\n4. Criando CurriculumScheduler normal (sem debug)...")
try:
    curriculum_normal = CurriculumScheduler({'min_episodes_per_stage': 10})
    print("   ✅ Instância normal criada")
    
    # Verificar atributos
    print("\n   Verificando atributos da instância normal:")
    for attr in attrs_to_check:
        has_it = hasattr(curriculum_normal, attr)
        if has_it:
            value = getattr(curriculum_normal, attr)
            print(f"   ✅ {attr:20s} = {type(value).__name__}")
        else:
            print(f"   ❌ {attr:20s} = MISSING!")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Testar acesso a stages
print("\n5. Testando acesso a stages...")
try:
    print(f"   current_stage = {curriculum_normal.current_stage}")
    print(f"   type(stages) = {type(curriculum_normal.stages)}")
    print(f"   len(stages) = {len(curriculum_normal.stages)}")
    
    print(f"\n   Acessando stages[current_stage]...")
    stage = curriculum_normal.stages[curriculum_normal.current_stage]
    print(f"   ✅ Stage acessado: {stage.name}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Diagnóstico completo
print("\n" + "="*70)
print("DIAGNÓSTICO COMPLETO")
print("="*70)

print("\n__dict__ do curriculum:")
for key, value in curriculum_normal.__dict__.items():
    print(f"   {key:20s} = {type(value).__name__:15s}")

print("\n" + "="*70)
print("✅ TESTE COMPLETO!")
print("="*70)
print("\nSe você viu '❌ current_stage = MISSING!' em qualquer lugar,")
print("significa que há um problema no __init__ do CurriculumScheduler.")
print("\nPor favor, me mostre a saída completa deste teste!")