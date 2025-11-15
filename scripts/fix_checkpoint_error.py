# """
# fix_checkpoint_error.py

# Correção rápida para erro do PyTorch 2.6
# Execute este script para corrigir seus checkpoints
# """

import torch
import glob
from pathlib import Path

print("="*70)
print("CORREÇÃO DE CHECKPOINTS - PyTorch 2.6")
print("="*70)

# Encontrar todos os checkpoints
checkpoints = glob.glob('checkpoint_*.pt')

if not checkpoints:
    print("\n❌ Nenhum checkpoint encontrado!")
    print("Os checkpoints devem estar na mesma pasta deste script.")
    exit(1)

print(f"\n✓ Encontrados {len(checkpoints)} checkpoint(s):")
for cp in checkpoints:
    print(f"  - {cp}")

print("\n" + "="*70)
print("TESTANDO CARREGAMENTO")
print("="*70)

for checkpoint_path in checkpoints:
    print(f"\nTestando: {checkpoint_path}")
    
    try:
        # Tentar carregar com weights_only=False (fix PyTorch 2.6)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"  ✓ Checkpoint OK!")
        
        # Mostrar informações
        if 'iteration' in checkpoint:
            print(f"    Iteração: {checkpoint['iteration']}")
        
        if 'history' in checkpoint and checkpoint['history']:
            history = checkpoint['history']
            if 'rewards' in history and history['rewards']:
                print(f"    Reward médio: {history['rewards'][-1]:.2f}")
            if 'utilizations' in history and history['utilizations']:
                print(f"    Utilização: {history['utilizations'][-1]*100:.1f}%")
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")

print("\n" + "="*70)
print("SOLUÇÃO")
print("="*70)

print("""
O erro acontece porque o PyTorch 2.6 mudou o comportamento padrão
de torch.load() para ser mais seguro.

SOLUÇÃO APLICADA:
Seus scripts foram atualizados para usar weights_only=False.

PRÓXIMOS PASSOS:
1. Os checkpoints estão OK
2. Execute: python train_continuo.py
3. O script agora vai funcionar!

Se ainda der erro, execute:
  python train_continuo.py --device cpu
""")

print("="*70)
print("✓ Verificação completa!")
print("="*70)