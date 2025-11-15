# """
# test_performance.py

# Compara performance das duas abordagens
# """

import numpy as np
import torch
import time

print("="*70)
print("TESTE DE PERFORMANCE: Conversões NumPy → PyTorch")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

n_steps = 2048
batch_size = 64

# =============================================================================
# TESTE 1: Conversão de Lista (LENTO)
# =============================================================================

print("TESTE 1: Conversão Lenta (lista de arrays)")
print("-"*70)

# Simula coleta de dados
positions_list = []
for i in range(n_steps):
    positions_list.append(np.random.rand(2).astype(np.float32))

# Medir tempo de conversão
start = time.time()
tensor_slow = torch.tensor(positions_list, device=device)  # ❌ LENTO
elapsed_slow = time.time() - start

print(f"Tempo: {elapsed_slow*1000:.2f} ms")
print(f"Shape: {tensor_slow.shape}")

# =============================================================================
# TESTE 2: Conversão Pre-alocada (RÁPIDO)
# =============================================================================

print("\nTESTE 2: Conversão Rápida (array pre-alocado)")
print("-"*70)

# Pre-alocar
positions_array = np.zeros((n_steps, 2), dtype=np.float32)

# Simula coleta
for i in range(n_steps):
    positions_array[i] = np.random.rand(2).astype(np.float32)

# Medir tempo de conversão
start = time.time()
tensor_fast = torch.from_numpy(positions_array).to(device)  # ✓ RÁPIDO
elapsed_fast = time.time() - start

print(f"Tempo: {elapsed_fast*1000:.2f} ms")
print(f"Shape: {tensor_fast.shape}")

# =============================================================================
# COMPARAÇÃO
# =============================================================================

print("\n" + "="*70)
print("COMPARAÇÃO")
print("="*70)

speedup = elapsed_slow / elapsed_fast

print(f"Método Lento:  {elapsed_slow*1000:8.2f} ms")
print(f"Método Rápido: {elapsed_fast*1000:8.2f} ms")
print(f"\n⚡ SPEEDUP: {speedup:.1f}x mais rápido!")

if speedup > 5:
    print("✓✓✓ EXCELENTE otimização! ✓✓✓")
elif speedup > 2:
    print("✓✓ BOA otimização!")
else:
    print("✓ Pequena melhoria")

# =============================================================================
# TESTE 3: Batch Indexing
# =============================================================================

print("\n" + "="*70)
print("TESTE 3: Batch Indexing")
print("="*70)

# Criar tensor grande
big_tensor = torch.randn(n_steps, 6, 256, 256, device=device)
batch_indices = np.random.choice(n_steps, batch_size, replace=False)

# Método 1: Stack (lento)
print("\nMétodo 1: torch.stack (LENTO)")
start = time.time()
batch_slow = torch.stack([big_tensor[i] for i in batch_indices])
elapsed_idx_slow = time.time() - start
print(f"Tempo: {elapsed_idx_slow*1000:.2f} ms")

# Método 2: Indexação direta (rápido)
print("\nMétodo 2: Indexação direta (RÁPIDO)")
start = time.time()
batch_fast = big_tensor[batch_indices]
elapsed_idx_fast = time.time() - start
print(f"Tempo: {elapsed_idx_fast*1000:.2f} ms")

speedup_idx = elapsed_idx_slow / elapsed_idx_fast
print(f"\n⚡ SPEEDUP: {speedup_idx:.1f}x mais rápido!")

# =============================================================================
# RESUMO FINAL
# =============================================================================

print("\n" + "="*70)
print("RESUMO DAS OTIMIZAÇÕES")
print("="*70)

total_speedup = (elapsed_slow + elapsed_idx_slow) / (elapsed_fast + elapsed_idx_fast)

print(f"""
Conversão NumPy→Tensor: {speedup:.1f}x mais rápido
Batch Indexing:         {speedup_idx:.1f}x mais rápido
SPEEDUP TOTAL:          {total_speedup:.1f}x

Economia de tempo por iteração: {(elapsed_slow + elapsed_idx_slow - elapsed_fast - elapsed_idx_fast)*1000:.1f} ms

Para 1000 iterações:
  Antes:  {(elapsed_slow + elapsed_idx_slow)*1000:.1f} segundos
  Depois: {(elapsed_fast + elapsed_idx_fast)*1000:.1f} segundos
  Economiza: {((elapsed_slow + elapsed_idx_slow)*1000 - (elapsed_fast + elapsed_idx_fast)*1000)/60:.1f} minutos!
""")

print("="*70)
print("✓ Teste de performance concluído!")
print("="*70)