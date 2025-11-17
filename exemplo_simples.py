# """
# Exemplo Simples - Usa automaticamente o checkpoint mais recente
# """
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path se necessário
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from checkpoint_manager import load_latest_checkpoint


def criar_pecas_exemplo():
    #"""Cria peças de exemplo para teste"""
    from src.geometry.polygon import Polygon
    
    # Peças simples para teste
    pecas = [
        Polygon([(0, 0), (100, 0), (100, 50), (0, 50)]),  # Retângulo 100x50
        Polygon([(0, 0), (80, 0), (80, 60), (0, 60)]),    # Retângulo 80x60
        Polygon([(0, 0), (60, 0), (60, 40), (0, 40)]),    # Retângulo 60x40
    ]
    
    return pecas


def testar_com_checkpoint():
    #"""Testa o sistema usando o checkpoint mais recente"""
    
    print("=" * 80)
    print("EXEMPLO SIMPLES - Teste com Checkpoint Mais Recente")
    print("=" * 80)
    
    # 1. Carrega checkpoint mais recente automaticamente
    print("\n🔍 Buscando checkpoint mais recente...")
    checkpoint = load_latest_checkpoint(base_dir="scripts", device='cpu')
    
    if checkpoint is None:
        print("\n❌ Nenhum checkpoint encontrado!")
        print("   Execute o treinamento primeiro:")
        print("   python train_ppo.py")
        return
    
    # 2. Cria as peças de exemplo
    print("\n📦 Criando peças de exemplo...")
    pecas = criar_pecas_exemplo()
    print(f"   Criadas {len(pecas)} peças")
    
    # 3. Configura o ambiente
    print("\n🎯 Configurando ambiente...")
    try:
        from src.environment.nesting_env_fixed import NestingEnv
        
        env = NestingEnv(
            pieces=pecas,
            sheet_width=500,
            sheet_height=400,
            render_mode=None
        )
        print("   ✓ Ambiente criado com sucesso")
        
    except Exception as e:
        print(f"   ❌ Erro ao criar ambiente: {e}")
        return
    
    # 4. Cria o modelo (simplificado - sem CNN completo)
    print("\n🧠 Criando modelo...")
    try:
        # Determina dimensões
        obs_shape = env.observation_space['visual'].shape
        n_actions = env.action_space.shape[0]
        
        print(f"   Observação: {obs_shape}")
        print(f"   Ações: {n_actions}")
        
        # Cria modelo simples para demonstração
        from collections import OrderedDict
        import torch.nn as nn
        
        class SimpleActor(nn.Module):
            def __init__(self, obs_channels, n_actions):
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
                    x = x['visual']
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        actor = SimpleActor(obs_shape[0], n_actions)
        
        # Tenta carregar pesos do checkpoint
        if 'actor_state_dict' in checkpoint:
            try:
                actor.load_state_dict(checkpoint['actor_state_dict'])
                print("   ✓ Pesos do actor carregados do checkpoint")
            except Exception as e:
                print(f"   ⚠️  Não foi possível carregar pesos: {e}")
                print("   ℹ️  Usando modelo aleatório para demonstração")
        
        actor.eval()
        
    except Exception as e:
        print(f"   ❌ Erro ao criar modelo: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Executa alguns passos de teste
    print("\n🎮 Executando teste...")
    print("-" * 80)
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(min(len(pecas), 5)):  # Máximo 5 passos ou número de peças
        # Converte observação para tensor
        visual_obs = torch.FloatTensor(obs['visual']).unsqueeze(0)
        
        # Obtém ação do modelo
        with torch.no_grad():
            action = actor({'visual': visual_obs}).squeeze(0).numpy()
        
        # Executa ação
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Passo {step + 1}:")
        print(f"  Ação: x={action[0]:.3f}, y={action[1]:.3f}, rot={action[2]:.3f}")
        print(f"  Recompensa: {reward:.4f}")
        print(f"  Utilização: {info.get('utilization', 0):.2%}")
        
        if terminated or truncated:
            print(f"\n🏁 Episódio finalizado no passo {step + 1}")
            break
    
    print("-" * 80)
    print(f"\n📊 Resultado Final:")
    print(f"   Recompensa total: {total_reward:.4f}")
    print(f"   Utilização final: {info.get('utilization', 0):.2%}")
    print(f"   Peças posicionadas: {info.get('pieces_placed', 0)}/{len(pecas)}")
    
    # 6. Informações do checkpoint
    print("\n" + "=" * 80)
    print("📈 INFORMAÇÕES DO CHECKPOINT USADO")
    print("=" * 80)
    if 'epoch' in checkpoint:
        print(f"Época de treinamento: {checkpoint['epoch']}")
    if 'iteration' in checkpoint:
        print(f"Iteração: {checkpoint['iteration']}")
    if 'avg_reward' in checkpoint:
        print(f"Recompensa média no treino: {checkpoint['avg_reward']:.4f}")
    if 'avg_utilization' in checkpoint:
        print(f"Utilização média no treino: {checkpoint['avg_utilization']:.2%}")
    
    print("\n✅ Teste concluído!")


if __name__ == "__main__":
    testar_com_checkpoint()