# """
# exemplo_simples.py

# Exemplo SUPER SIMPLES de como usar o sistema
# (com detecção automática de checkpoint)
# """

import sys
import glob
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from use_trained_model import NestingSystem, PieceLoader, ResultExporter

print("="*70)
print("EXEMPLO SIMPLES DE USO")
print("="*70)

# 0. VERIFICAR SE TEM CHECKPOINT
print("\n0. Verificando checkpoints...")
checkpoints = glob.glob('checkpoint_*.pt')

if not checkpoints:
    print("❌ ERRO: Nenhum checkpoint encontrado!")
    print("\n" + "="*70)
    print("VOCÊ PRECISA TREINAR PRIMEIRO!")
    print("="*70)
    print("""
Execute um destes comandos para treinar:

Opção 1 (GPU 2GB):
  python train_2gb_gpu.py

Opção 2 (Treinamento continuado):
  python train_continuo.py --iterations 100

Opção 3 (Ultra-rápido para teste):
  python train_ultra_fast.py

Após treinar, execute este script novamente.
    """)
    exit(1)

# Usar checkpoint mais recente
checkpoint_path = max(checkpoints, key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))
print(f"   ✓ Checkpoint encontrado: {checkpoint_path}")

# 1. DEFINIR SUAS PEÇAS (em milímetros)
print("\n1. Definindo peças...")
pecas = [
    (150, 100),  # Peça 1: 150mm × 100mm
    (120, 80),   # Peça 2: 120mm × 80mm
    (180, 90),   # Peça 3: 180mm × 90mm
    (100, 70),   # Peça 4: 100mm × 70mm
    (140, 110),  # Peça 5: 140mm × 110mm
]
print(f"   ✓ {len(pecas)} peças definidas")

# 2. CARREGAR PEÇAS
print("\n2. Carregando peças...")
loader = PieceLoader()
pieces = loader.from_rectangles_list(pecas)

# 3. CRIAR SISTEMA
print("\n3. Criando sistema de nesting...")
try:
    system = NestingSystem(
        checkpoint_path=checkpoint_path,
        container_width=1000,    # Chapa: 1000mm × 600mm
        container_height=600,
        device='cuda'  # ou 'cpu'
    )
except Exception as e:
    print(f"\n❌ Erro ao carregar modelo: {e}")
    print("\nPossíveis soluções:")
    print("1. Treine um novo modelo: python train_continuo.py")
    print("2. Use CPU: mude device='cuda' para device='cpu'")
    exit(1)

# 4. EXECUTAR NESTING
print("\n4. Executando nesting...")
try:
    result = system.nest_pieces(
        pieces,
        max_attempts=3,      # Tenta 3 vezes
        visualize=True       # Mostra imagem
    )
except Exception as e:
    print(f"\n❌ Erro durante nesting: {e}")
    print("\nVerifique se o environment está funcionando corretamente.")
    exit(1)

# 5. EXPORTAR RESULTADOS
print("\n5. Exportando resultados...")
exporter = ResultExporter()
exporter.to_json(result, 'resultado.json')
exporter.to_svg(result, 'resultado.svg')

try:
    exporter.to_dxf(result, 'resultado.dxf')
except ImportError:
    print("   ⚠️  DXF não exportado (pip install ezdxf)")

print("\n" + "="*70)
print("✓✓✓ PRONTO! ✓✓✓")
print("="*70)
print(f"""
Resultados:
• Utilização: {result['utilization']*100:.2f}%
• Peças colocadas: {result['n_placed']}/{len(pecas)}
• Tempo: {result['execution_time']:.2f}s

Arquivos gerados:
• nesting_result.png  ← Visualização
• resultado.json      ← Dados completos
• resultado.svg       ← Para CAD
• resultado.dxf       ← Para AutoCAD (se ezdxf instalado)

Agora você pode:
1. Abrir nesting_result.png para ver
2. Usar resultado.json no seu sistema
3. Importar resultado.svg no CAD
4. Importar resultado.dxf no AutoCAD

Para treinar mais e melhorar:
  python train_continuo.py --iterations 100
""")

print("="*70)