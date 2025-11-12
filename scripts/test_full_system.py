# """
# scripts/test_full_system.py

# Teste do sistema completo: Environment + Geometria + Image Encoder
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.polygon import create_rectangle, create_random_polygon
from src.environment.nesting_env import NestingEnvironment, NestingConfig

def test_full_system():
    print("="*70)
    print("TESTE: SISTEMA COMPLETO")
    print("="*70)
    
    # Criar peças reais
    pieces = [
        create_rectangle(50, 30),
        create_rectangle(40, 25),
        create_random_polygon(5, radius=20),
        create_random_polygon(6, radius=25),
    ]
    
    for i, piece in enumerate(pieces):
        piece.id = i
    
    print(f"✓ Criadas {len(pieces)} peças")
    
    # Criar environment com peças reais
    config = NestingConfig(max_steps=20)
    env = NestingEnvironment(config=config)
    
    obs, info = env.reset(options={'pieces': pieces})
    print(f"✓ Environment resetado com peças reais")
    print(f"  Layout image shape: {obs['layout_image'].shape}")
    
    # Executar alguns steps
    for step in range(5):
        action = {
            'position': env.action_space['position'].sample(),
            'rotation': env.action_space['rotation'].sample()
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: reward={reward:.2f}, "
              f"placed={info['n_placed']}/{len(pieces)}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Sistema completo funciona!")
    print(f"  Utilização final: {info['utilization']*100:.1f}%")
    
    env.close()

if __name__ == "__main__":
    test_full_system()