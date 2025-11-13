# """
# scripts/quick_test.py
# Teste rápido do sistema completo antes de treinar
# """

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

print("\n" + "="*70)
print("🧪 TESTE RÁPIDO DO SISTEMA COMPLETO")
print("="*70)

# =============================================================================
# 1. TESTAR IMPORTS
# =============================================================================

print("\n1. Testando imports...")
try:
    from src.geometry.polygon import create_rectangle, create_random_polygon
    from src.environment.nesting_env import NestingEnvironment, NestingConfig
    from src.models.cnn.encoder import LayoutCNNEncoder
    from src.representation.image_encoder import render_layout_as_image
    from src.training.curriculum import CurriculumScheduler
    print("   ✅ Todos os imports OK!")
except Exception as e:
    print(f"   ❌ Erro nos imports: {e}")
    sys.exit(1)

# =============================================================================
# 2. TESTAR GEOMETRIA
# =============================================================================

print("\n2. Testando geometria...")
try:
    piece1 = create_rectangle(50, 30)
    piece2 = create_random_polygon(6, 25)
    
    # Transformações
    piece1_rotated = piece1.rotate(45)
    piece2_translated = piece2.translate(100, 50)
    
    print(f"   ✅ Geometria OK!")
    print(f"      - Piece 1: área={piece1.area:.1f}, perímetro={piece1.perimeter:.1f}")
    print(f"      - Piece 2: {len(piece2.vertices)} vértices")
except Exception as e:
    print(f"   ❌ Erro na geometria: {e}")
    sys.exit(1)

# =============================================================================
# 3. TESTAR IMAGE ENCODER
# =============================================================================

print("\n3. Testando image encoder...")
try:
    container = create_rectangle(1000, 600, center=(500, 300))
    placed = [
        create_rectangle(50, 30).set_position(100, 100),
        create_rectangle(40, 25).set_position(200, 150)
    ]
    next_piece = create_rectangle(45, 28)
    
    image = render_layout_as_image(
        container=container,
        placed_pieces=placed,
        next_piece=next_piece,
        size=256
    )
    
    print(f"   ✅ Image encoder OK!")
    print(f"      - Shape: {image.shape}")
    print(f"      - Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"      - Dtype: {image.dtype}")
except Exception as e:
    print(f"   ❌ Erro no image encoder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 4. TESTAR ENVIRONMENT
# =============================================================================

print("\n4. Testando environment...")
try:
    config = NestingConfig(max_steps=20)
    env = NestingEnvironment(config=config)
    
    # Criar peças
    pieces = [
        create_rectangle(50, 30),
        create_rectangle(40, 25),
        create_random_polygon(5, 20)
    ]
    for i, p in enumerate(pieces):
        p.id = i
    
    # Reset
    obs, info = env.reset(options={'pieces': pieces})
    
    print(f"   ✅ Environment OK!")
    print(f"      - Observation keys: {list(obs.keys())}")
    print(f"      - Layout image shape: {obs['layout_image'].shape}")
    print(f"      - Action space: position={env.action_space['position'].shape}, "
          f"rotation={env.action_space['rotation'].n}")
    
    # Teste de step
    action = {
        'position': np.array([0.5, 0.5]),
        'rotation': 0
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"      - Step OK: reward={reward:.2f}, done={terminated or truncated}")
    
    env.close()
    
except Exception as e:
    print(f"   ❌ Erro no environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 5. TESTAR CNN
# =============================================================================

print("\n5. Testando CNN encoder...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cnn = LayoutCNNEncoder(
        input_channels=6,
        embedding_dim=256
    ).to(device)
    
    # Input dummy
    x = torch.randn(2, 6, 256, 256).to(device)
    
    # Forward
    with torch.no_grad():
        embedding, heatmap = cnn(x)
    
    print(f"   ✅ CNN OK!")
    print(f"      - Device: {device}")
    print(f"      - Parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"      - Embedding shape: {embedding.shape}")
    print(f"      - Heatmap shape: {heatmap.shape}")
    
except Exception as e:
    print(f"   ❌ Erro na CNN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 6. TESTAR ACTOR-CRITIC
# =============================================================================

print("\n6. Testando Actor-Critic...")
try:
    import torch.nn as nn
    
    class ActorCriticTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = LayoutCNNEncoder(6, 256)
            
            total_input = 256 + 10 + 10 + 5
            self.shared = nn.Linear(total_input, 512)
            self.actor_pos = nn.Linear(512, 2)
            self.actor_rot = nn.Linear(512, 36)
            self.critic = nn.Linear(512, 1)
        
        def forward(self, obs):
            cnn_emb, _ = self.cnn(obs['layout_image'])
            combined = torch.cat([
                cnn_emb,
                obs['current_piece'],
                obs['remaining_pieces'],
                obs['stats']
            ], dim=1)
            shared = torch.relu(self.shared(combined))
            return {
                'position': torch.sigmoid(self.actor_pos(shared)),
                'rotation': self.actor_rot(shared),
                'value': self.critic(shared)
            }
    
    agent = ActorCriticTest().to(device)
    
    # Criar observação dummy
    obs_tensor = {
        'layout_image': torch.randn(1, 6, 256, 256).to(device),
        'current_piece': torch.randn(1, 10).to(device),
        'remaining_pieces': torch.randn(1, 10).to(device),
        'stats': torch.randn(1, 5).to(device)
    }
    
    # Forward
    with torch.no_grad():
        output = agent(obs_tensor)
    
    print(f"   ✅ Actor-Critic OK!")
    print(f"      - Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"      - Position output: {output['position'].shape}")
    print(f"      - Rotation output: {output['rotation'].shape}")
    print(f"      - Value output: {output['value'].shape}")
    
except Exception as e:
    print(f"   ❌ Erro no Actor-Critic: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 7. TESTAR CURRICULUM
# =============================================================================

print("\n7. Testando curriculum...")
try:
    curriculum = CurriculumScheduler({'min_episodes_per_stage': 10})
    
    stage = curriculum.get_current_stage()
    print(f"   ✅ Curriculum OK!")
    print(f"      - Total stages: {len(curriculum.stages)}")
    print(f"      - Current: {stage.name}")
    print(f"      - Pieces range: {stage.n_pieces_range}")
    
    # Gerar peças
    problem_config = curriculum.get_problem_config()
    pieces = curriculum.generate_pieces(problem_config)
    
    print(f"      - Generated {len(pieces)} pieces")
    
    # Simular progresso
    for i in range(15):
        curriculum.update(utilization=0.7)
    
    stats = curriculum.get_stats()
    print(f"      - After 15 episodes: stage={stats['current_stage']}, "
          f"success_rate={stats['success_rate']:.2f}")
    
except Exception as e:
    print(f"   ❌ Erro no curriculum: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 8. TESTE INTEGRADO
# =============================================================================

print("\n8. Teste integrado completo...")
try:
    # Criar sistema completo
    env = NestingEnvironment(config=NestingConfig(max_steps=10))
    agent = ActorCriticTest().to(device)
    curriculum = CurriculumScheduler({'min_episodes_per_stage': 5})
    
    # Gerar peças
    problem_config = curriculum.get_problem_config()
    pieces = curriculum.generate_pieces(problem_config)
    
    # Reset env
    obs, _ = env.reset(options={'pieces': pieces})
    
    # Executar alguns steps
    total_reward = 0
    for step in range(5):
        # Converter observação
        obs_tensor = {
            'layout_image': torch.from_numpy(obs['layout_image']).unsqueeze(0).to(device),
            'current_piece': torch.from_numpy(obs['current_piece']).unsqueeze(0).to(device),
            'remaining_pieces': torch.from_numpy(obs['remaining_pieces']).unsqueeze(0).to(device),
            'stats': torch.from_numpy(obs['stats']).unsqueeze(0).to(device)
        }
        
        # Gerar ação
        with torch.no_grad():
            output = agent(obs_tensor)
            position = output['position'][0].cpu().numpy()
            rotation = torch.argmax(output['rotation'], dim=-1)[0].cpu().item()
        
        # Step
        action = {'position': position, 'rotation': rotation}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"   ✅ Teste integrado OK!")
    print(f"      - Steps executados: {step+1}")
    print(f"      - Reward total: {total_reward:.2f}")
    print(f"      - Utilização: {info['utilization']*100:.1f}%")
    
    env.close()
    
except Exception as e:
    print(f"   ❌ Erro no teste integrado: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# RESUMO
# =============================================================================

print("\n" + "="*70)
print("✅ TODOS OS TESTES PASSARAM!")
print("="*70)
print("\n🚀 Sistema pronto para treinamento!")
print("\nPara treinar, execute:")
print("   python scripts/train_complete_system.py --iterations 1000")
print("\n" + "="*70 + "\n")