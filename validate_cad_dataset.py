#!/usr/bin/env python3
"""
validate_cad_dataset.py - Valida e visualiza dataset CAD

Verifica:
- Quais arquivos foram carregados
- Complexidade de cada conjunto
- Visualização das peças
- Estatísticas do dataset

USO: python validate_cad_dataset.py
"""

import sys
from pathlib import Path

# Adicionar ao path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("="*80)
    print("VALIDAÇÃO DE DATASET CAD")
    print("="*80)
    
    # Importar
    try:
        from train_with_real_cad import CADDatasetLoader
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")
        print("\nCertifique-se de que train_with_real_cad.py está no diretório.")
        return 1
    
    # Pedir diretório
    print("\nDigite o caminho do dataset:")
    print("(ou Enter para usar: datasets/cad_pieces)")
    dataset_dir = input("> ").strip() or "datasets/cad_pieces"
    
    # Carregar
    loader = CADDatasetLoader(dataset_dir)
    piece_sets = loader.load_all()
    
    if not piece_sets:
        print("\n" + "="*80)
        print("📝 COMO ADICIONAR ARQUIVOS CAD")
        print("="*80)
        print(f"""
1. Crie a pasta: mkdir -p {dataset_dir}

2. Adicione arquivos:
   - DXF (AutoCAD): use LWPOLYLINE
   - SVG: use polygon fechado
   - JSON: formato simples (veja exemplo)

3. Execute este script novamente
        """)
        return 0
    
    # Estatísticas gerais
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS DO DATASET")
    print("="*80)
    
    total_pieces = sum(ps.n_pieces for ps in piece_sets)
    avg_complexity = sum(ps.complexity for ps in piece_sets) / len(piece_sets)
    
    print(f"Total de conjuntos: {len(piece_sets)}")
    print(f"Total de peças: {total_pieces}")
    print(f"Complexidade média: {avg_complexity:.2f}")
    print(f"Range de complexidade: {piece_sets[0].complexity:.2f} → {piece_sets[-1].complexity:.2f}")
    
    # Detalhes de cada conjunto
    print("\n" + "="*80)
    print("📋 CONJUNTOS DE PEÇAS")
    print("="*80)
    
    for i, ps in enumerate(piece_sets, 1):
        print(f"\n{i}. {ps.name}")
        print(f"   Arquivo: {Path(ps.source_file).name}")
        print(f"   Peças: {ps.n_pieces}")
        print(f"   Complexidade: {ps.complexity:.2f}")
        
        # Detalhes das peças
        if ps.pieces:
            vertices_counts = [len(p.vertices) for p in ps.pieces]
            print(f"   Vértices por peça: {min(vertices_counts)}-{max(vertices_counts)}")
    
    # Sugestão de ordem de treinamento
    print("\n" + "="*80)
    print("🎓 ORDEM SUGERIDA PARA TREINAMENTO")
    print("="*80)
    
    for i, ps in enumerate(piece_sets, 1):
        print(f"{i}. {ps.name} (complexidade: {ps.complexity:.2f})")
    
    # Salvar relatório
    report_path = Path(dataset_dir) / "dataset_report.txt"
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DO DATASET CAD\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total de conjuntos: {len(piece_sets)}\n")
        f.write(f"Total de peças: {total_pieces}\n")
        f.write(f"Complexidade média: {avg_complexity:.2f}\n\n")
        
        f.write("CONJUNTOS:\n")
        for i, ps in enumerate(piece_sets, 1):
            f.write(f"{i}. {ps.name}: {ps.n_pieces} peças, complexity: {ps.complexity:.2f}\n")
    
    print(f"\n💾 Relatório salvo em: {report_path}")
    
    print("\n" + "="*80)
    print("✅ VALIDAÇÃO CONCLUÍDA")
    print("="*80)
    print(f"""
Dataset pronto para treinamento!

Próximo passo:
  python train_with_real_cad.py
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())