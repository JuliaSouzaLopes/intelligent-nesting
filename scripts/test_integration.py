# """
# scripts/test_integration.py
# Teste de integração - VERSÃO CORRIGIDA
# """

import sys
import os
from pathlib import Path

# ============================================================================
# CORREÇÃO DE PATH - Adiciona a raiz do projeto ao sys.path
# ============================================================================

# Pega o diretório deste script
script_dir = Path(__file__).resolve().parent  # .../intelligent-nesting/scripts/

# Pega a raiz do projeto (um nível acima)
project_root = script_dir.parent  # .../intelligent-nesting/

# Adiciona ao sys.path ANTES de importar qualquer coisa
sys.path.insert(0, str(project_root))

print(f"✓ Project root adicionado ao path: {project_root}")

# ============================================================================
# AGORA PODE IMPORTAR
# ============================================================================

try:
    from src.geometry.polygon import Polygon, create_rectangle, create_random_polygon
    print("✓ Import src.geometry.polygon OK")
except ImportError as e:
    print(f"✗ ERRO ao importar: {e}")
    print("\nVerifique:")
    print(f"1. Existe o arquivo? {project_root / 'src' / 'geometry' / 'polygon.py'}")
    print(f"2. Existe __init__.py? {project_root / 'src' / '__init__.py'}")
    sys.exit(1)

try:
    from src.geometry.nfp import NFPCalculator
    print("✓ Import src.geometry.nfp OK")
except ImportError as e:
    print(f"✗ ERRO ao importar NFP: {e}")

# ============================================================================
# TESTES
# ============================================================================

def test_geometry():
    #"""Testa módulo de geometria"""
    print("\n" + "="*70)
    print("TESTE 1: GEOMETRIA")
    print("="*70)
    
    # Criar retângulo
    rect = create_rectangle(100, 50, center=(0, 0))
    print(f"✓ Retângulo criado: {rect}")
    print(f"  - Área: {rect.area:.2f}")
    print(f"  - Perímetro: {rect.perimeter:.2f}")
    
    # Rotacionar
    rotated = rect.rotate(45)
    print(f"✓ Rotação 45°: rotação={rotated.rotation}°")
    
    # Transladar
    moved = rect.translate(100, 50)
    print(f"✓ Translação: posição=({moved.position.x:.1f}, {moved.position.y:.1f})")
    
    # Polígono aleatório
    poly = create_random_polygon(6, radius=40, irregularity=0.5)
    print(f"✓ Polígono aleatório: {len(poly.vertices)} vértices")
    
    return True


def test_nfp():
    #"""Testa NFP calculator"""
    print("\n" + "="*70)
    print("TESTE 2: NFP (No-Fit Polygon)")
    print("="*70)
    
    try:
        piece_a = create_rectangle(50, 30)
        piece_b = create_rectangle(60, 40)
        
        calc = NFPCalculator(cache_enabled=True)
        nfp = calc.calculate_nfp(piece_a, piece_b)
        
        print(f"✓ NFP calculado:")
        print(f"  - Área NFP: {nfp.area:.2f}")
        print(f"  - Vértices: {len(nfp.vertices)}")
        
        stats = calc.get_cache_stats()
        print(f"✓ Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("TESTE DE INTEGRAÇÃO SIMPLIFICADO")
    print("="*70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working dir: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print("="*70)
    
    # Executar testes
    results = {
        'geometry': test_geometry(),
        'nfp': test_nfp(),
    }
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    print("="*70)
    
    if all(results.values()):
        print("\n✓✓✓ TODOS OS TESTES PASSARAM! ✓✓✓\n")
        return 0
    else:
        print("\n✗ ALGUNS TESTES FALHARAM\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)