# """
# aplicar_correcao.py

# Aplica correção do PyTorch 2.6 em TODOS os arquivos locais
# """

import os
import re
from pathlib import Path

print("="*70)
print("APLICANDO CORREÇÃO PyTorch 2.6")
print("="*70)

# Arquivos para corrigir
files_to_fix = [
    'use_trained_model.py',
    'train_continuo.py',
    'exemplo_simples.py'
]

# Padrão a corrigir
old_pattern = r'torch\.load\(([^,]+),\s*map_location=([^)]+)\)'
new_pattern = r'torch.load(\1, map_location=\2, weights_only=False)'

fixed_count = 0

for filename in files_to_fix:
    if not Path(filename).exists():
        print(f"\n⚠️  {filename} não encontrado, pulando...")
        continue
    
    print(f"\n📝 Processando: {filename}")
    
    # Ler arquivo
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se já tem a correção
    if 'weights_only=False' in content:
        print(f"   ✓ Já corrigido")
        continue
    
    # Fazer backup
    backup_file = filename + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   ✓ Backup: {backup_file}")
    
    # Aplicar correção
    original_content = content
    content = re.sub(old_pattern, new_pattern, content)
    
    if content != original_content:
        # Salvar corrigido
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ CORRIGIDO!")
        fixed_count += 1
    else:
        print(f"   ⚠️  Nenhuma alteração necessária")

print("\n" + "="*70)
print("RESUMO")
print("="*70)
print(f"Arquivos corrigidos: {fixed_count}")

if fixed_count > 0:
    print("\n✅ Correção aplicada com sucesso!")
    print("\nAgora execute:")
    print("  python exemplo_simples.py")
else:
    print("\n⚠️  Nenhum arquivo precisou ser corrigido")
    print("\nVerifique se:")
    print("1. Os arquivos estão na pasta correta")
    print("2. Você está executando da pasta do projeto")

print("\n" + "="*70)