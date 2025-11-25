#!/usr/bin/env python3
"""
GUIA COMPLETO: Como Usar o Sistema com Arquivos CAD (DXF/SVG/JSON)

Este script mostra EXATAMENTE como rodar o sistema de nesting
com peças vindas de arquivos CAD reais.
"""

import json
from pathlib import Path

# =============================================================================
# INSTALAÇÃO DE DEPENDÊNCIAS
# =============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          🎯 GUIA: NESTING COM ARQUIVOS CAD (DXF/SVG/JSON) 🎯              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
📦 PASSO 0: INSTALAR DEPENDÊNCIAS
═══════════════════════════════════════════════════════════════════════════

Execute estes comandos:

# Para arquivos DXF (AutoCAD):
pip install ezdxf --break-system-packages

# Para arquivos SVG:
pip install svgpathtools --break-system-packages

# Se não tiver PyTorch:
pip install torch torchvision --break-system-packages

═══════════════════════════════════════════════════════════════════════════
""")


# =============================================================================
# CARREGADORES DE ARQUIVO
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
📁 PASSO 1: CARREGAR PEÇAS DE ARQUIVO CAD
═══════════════════════════════════════════════════════════════════════════

OPÇÃO A: Arquivo DXF (AutoCAD)
------------------------------
""")

code_dxf = '''
from use_trained_model import PieceLoader

# Carregar peças de arquivo DXF
loader = PieceLoader()
pieces = loader.from_dxf("minhas_pecas.dxf")

# O loader busca todas as LWPOLYLINE no arquivo
# e converte para polígonos
'''
print(code_dxf)


print("""
OPÇÃO B: Arquivo SVG
------------------------------
""")

code_svg = '''
from use_trained_model import PieceLoader

# Carregar peças de arquivo SVG
loader = PieceLoader()
pieces = loader.from_svg("minhas_pecas.svg")
'''
print(code_svg)


print("""
OPÇÃO C: Arquivo JSON
------------------------------

1. Crie um arquivo JSON assim:
""")

json_example = {
    "pieces": [
        {
            "id": 0,
            "name": "Retângulo Grande",
            "vertices": [[0, 0], [150, 0], [150, 100], [0, 100]]
        },
        {
            "id": 1,
            "name": "Forma L",
            "vertices": [[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]
        },
        {
            "id": 2,
            "name": "Trapézio",
            "vertices": [[0, 0], [120, 0], [100, 60], [20, 60]]
        }
    ]
}

print(f"Exemplo (pieces.json):\n{json.dumps(json_example, indent=2)}")

code_json = '''

2. Carregue no Python:

from use_trained_model import PieceLoader

loader = PieceLoader()
pieces = loader.from_json("pieces.json")
'''
print(code_json)


print("""
OPÇÃO D: Lista de Retângulos (mais simples)
------------------------------
""")

code_rect = '''
from use_trained_model import PieceLoader

# Dimensões em mm: (largura, altura)
rectangles = [
    (150, 100),   # 150mm × 100mm
    (120, 80),    # 120mm × 80mm
    (180, 90),    # etc.
    (100, 70),
]

loader = PieceLoader()
pieces = loader.from_rectangles_list(rectangles)
'''
print(code_rect)


# =============================================================================
# EXECUÇÃO DO NESTING
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
🎯 PASSO 2: EXECUTAR O NESTING
═══════════════════════════════════════════════════════════════════════════
""")

code_nesting = '''
from use_trained_model import NestingSystem

# Criar sistema (usa checkpoint mais recente automaticamente)
system = NestingSystem(
    checkpoint_path="scripts/checkpoint_epoch_50.pt",  # Seu checkpoint
    container_width=1000,    # Largura da chapa em mm
    container_height=600,    # Altura da chapa em mm
    device='cuda'            # 'cuda' para GPU ou 'cpu'
)

# Executar nesting
result = system.nest_pieces(
    pieces,
    max_attempts=3,      # Tenta 3 vezes, retorna melhor
    visualize=True       # Mostra imagem do resultado
)

# Ver resultado
print(f"Utilização: {result['utilization']*100:.2f}%")
print(f"Peças colocadas: {result['n_placed']}/{len(pieces)}")
'''
print(code_nesting)


# =============================================================================
# EXPORTAÇÃO DOS RESULTADOS
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
💾 PASSO 3: EXPORTAR RESULTADO PARA CAD
═══════════════════════════════════════════════════════════════════════════
""")

code_export = '''
from use_trained_model import ResultExporter

exporter = ResultExporter()

# Exportar para JSON (dados completos)
exporter.to_json(result, "nesting_result.json")

# Exportar para SVG (para importar em CAD)
exporter.to_svg(result, "nesting_result.svg")

# Exportar para DXF (para AutoCAD)
exporter.to_dxf(result, "nesting_result.dxf")
# Agora abra nesting_result.dxf no AutoCAD!
'''
print(code_export)


# =============================================================================
# SCRIPT COMPLETO
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
📋 SCRIPT COMPLETO - COPIE E USE
═══════════════════════════════════════════════════════════════════════════

Salve como: run_nesting_cad.py
""")

complete_script = '''
#!/usr/bin/env python3
"""
run_nesting_cad.py - Script completo para nesting com arquivos CAD
"""

import sys
import glob
from pathlib import Path

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

from use_trained_model import NestingSystem, PieceLoader, ResultExporter


def main():
    print("=" * 70)
    print("NESTING COM ARQUIVOS CAD")
    print("=" * 70)
    
    # =================================================================
    # CONFIGURAÇÃO - EDITE AQUI!
    # =================================================================
    
    # Arquivo de entrada (escolha um):
    INPUT_FILE = "pecas.dxf"           # Arquivo DXF
    # INPUT_FILE = "pecas.svg"         # Arquivo SVG
    # INPUT_FILE = "pecas.json"        # Arquivo JSON
    
    # Dimensões da chapa (em mm)
    CONTAINER_WIDTH = 1000    # Largura
    CONTAINER_HEIGHT = 600    # Altura
    
    # Número de tentativas (mais = melhor resultado, mais tempo)
    MAX_ATTEMPTS = 3
    
    # =================================================================
    # CARREGAR PEÇAS
    # =================================================================
    
    print(f"\\n1. Carregando peças de {INPUT_FILE}...")
    loader = PieceLoader()
    
    # Detectar formato pelo extensão
    ext = Path(INPUT_FILE).suffix.lower()
    
    if ext == '.dxf':
        pieces = loader.from_dxf(INPUT_FILE)
    elif ext == '.svg':
        pieces = loader.from_svg(INPUT_FILE)
    elif ext == '.json':
        pieces = loader.from_json(INPUT_FILE)
    else:
        print(f"❌ Formato não suportado: {ext}")
        print("   Use: .dxf, .svg, ou .json")
        return
    
    if not pieces:
        print("❌ Nenhuma peça carregada!")
        return
    
    print(f"   ✓ Carregadas {len(pieces)} peças")
    
    # =================================================================
    # ENCONTRAR CHECKPOINT
    # =================================================================
    
    print("\\n2. Buscando checkpoint...")
    
    # Procura checkpoints na pasta scripts
    checkpoints = glob.glob("scripts/*.pt")
    
    if not checkpoints:
        # Tenta na pasta atual
        checkpoints = glob.glob("*.pt")
    
    if not checkpoints:
        print("❌ Nenhum checkpoint encontrado!")
        print("   Execute o treinamento primeiro.")
        return
    
    # Usa o mais recente
    checkpoint_path = max(checkpoints, key=lambda x: Path(x).stat().st_mtime)
    print(f"   ✓ Usando: {checkpoint_path}")
    
    # =================================================================
    # CRIAR SISTEMA
    # =================================================================
    
    print("\\n3. Criando sistema de nesting...")
    
    try:
        system = NestingSystem(
            checkpoint_path=checkpoint_path,
            container_width=CONTAINER_WIDTH,
            container_height=CONTAINER_HEIGHT,
            device='cuda'  # Mude para 'cpu' se não tiver GPU
        )
    except Exception as e:
        print(f"❌ Erro ao criar sistema: {e}")
        print("\\nTente: device='cpu'")
        return
    
    # =================================================================
    # EXECUTAR NESTING
    # =================================================================
    
    print(f"\\n4. Executando nesting ({MAX_ATTEMPTS} tentativas)...")
    
    result = system.nest_pieces(
        pieces,
        max_attempts=MAX_ATTEMPTS,
        visualize=True
    )
    
    # =================================================================
    # MOSTRAR RESULTADO
    # =================================================================
    
    print("\\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Utilização: {result['utilization']*100:.2f}%")
    print(f"Peças colocadas: {result['n_placed']}/{len(pieces)}")
    print(f"Tempo: {result.get('execution_time', 0):.2f}s")
    
    # =================================================================
    # EXPORTAR RESULTADOS
    # =================================================================
    
    print("\\n5. Exportando resultados...")
    exporter = ResultExporter()
    
    # JSON (dados completos)
    exporter.to_json(result, "nesting_result.json")
    print("   ✓ nesting_result.json")
    
    # SVG (para CAD)
    exporter.to_svg(result, "nesting_result.svg")
    print("   ✓ nesting_result.svg")
    
    # DXF (para AutoCAD)
    try:
        exporter.to_dxf(result, "nesting_result.dxf")
        print("   ✓ nesting_result.dxf")
    except ImportError:
        print("   ⚠️  DXF não exportado (pip install ezdxf)")
    
    # PNG (visualização)
    print("   ✓ nesting_result.png")
    
    print("\\n" + "=" * 70)
    print("✅ PRONTO!")
    print("=" * 70)
    print("""
Agora você pode:

1. Abrir nesting_result.png para visualizar
2. Importar nesting_result.svg no seu CAD
3. Abrir nesting_result.dxf no AutoCAD
4. Usar nesting_result.json para integração

""")


if __name__ == "__main__":
    main()
'''

print(complete_script)


# =============================================================================
# FORMATOS DE ARQUIVO
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
📐 FORMATOS DE ARQUIVO SUPORTADOS
═══════════════════════════════════════════════════════════════════════════

┌────────┬───────────────────┬────────────────────────────────────────┐
│ Formato│ Extensão          │ Notas                                  │
├────────┼───────────────────┼────────────────────────────────────────┤
│ DXF    │ .dxf              │ AutoCAD - busca LWPOLYLINE             │
│ SVG    │ .svg              │ Paths fechados (polygon, polyline)     │
│ JSON   │ .json             │ Lista de vértices (veja exemplo acima) │
├────────┼───────────────────┼────────────────────────────────────────┤
│ Saída  │ .dxf, .svg, .json │ Resultado com posições calculadas      │
└────────┴───────────────────┴────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
""")


# =============================================================================
# FLUXO DE TRABALHO TÍPICO
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
🔄 FLUXO DE TRABALHO TÍPICO
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CAD (AutoCAD/SolidWorks)                                              │
│  └─→ Exportar peças como DXF                                           │
│       └─→ pecas.dxf                                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Python (Sistema de Nesting)                                           │
│  └─→ Carregar: loader.from_dxf("pecas.dxf")                           │
│       └─→ Executar nesting: system.nest_pieces(pieces)                │
│            └─→ Exportar: exporter.to_dxf("resultado.dxf")             │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CAD (AutoCAD)                                                         │
│  └─→ Importar resultado.dxf                                            │
│       └─→ Peças já posicionadas na chapa!                             │
│            └─→ Enviar para corte                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
""")


# =============================================================================
# EXEMPLO DE ARQUIVO JSON
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
📝 CRIAR ARQUIVO pieces_exemplo.json
═══════════════════════════════════════════════════════════════════════════
""")

# Criar arquivo de exemplo
example_pieces = {
    "description": "Exemplo de peças para nesting",
    "units": "millimeters",
    "pieces": [
        {
            "id": 0,
            "name": "Retângulo Grande",
            "vertices": [[0, 0], [150, 0], [150, 100], [0, 100]]
        },
        {
            "id": 1,
            "name": "Retângulo Médio",
            "vertices": [[0, 0], [120, 0], [120, 80], [0, 80]]
        },
        {
            "id": 2,
            "name": "Forma em L",
            "vertices": [[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]
        },
        {
            "id": 3,
            "name": "Trapézio",
            "vertices": [[0, 0], [120, 0], [100, 60], [20, 60]]
        },
        {
            "id": 4,
            "name": "Hexágono",
            "vertices": [[50, 0], [100, 25], [100, 75], [50, 100], [0, 75], [0, 25]]
        }
    ]
}

# Salvar arquivo
json_path = "/mnt/user-data/outputs/pieces_exemplo.json"
with open(json_path, 'w') as f:
    json.dump(example_pieces, f, indent=2)

print(f"✓ Arquivo criado: {json_path}")
print(f"\nConteúdo:\n{json.dumps(example_pieces, indent=2)}")


# =============================================================================
# DICAS IMPORTANTES
# =============================================================================

print("""

═══════════════════════════════════════════════════════════════════════════
💡 DICAS IMPORTANTES
═══════════════════════════════════════════════════════════════════════════

1. PREPARAÇÃO DO DXF
   ─────────────────
   • Use LWPOLYLINE para desenhar as peças
   • Feche todos os polígonos
   • Use unidades em milímetros
   • Cada peça deve ser um polígono separado

2. ESCALAS
   ────────
   • O sistema espera valores em milímetros
   • Se suas peças estão em metros, multiplique por 1000
   • Se estão em polegadas, multiplique por 25.4

3. PERFORMANCE
   ────────────
   • Use GPU (device='cuda') se disponível
   • Mais tentativas (max_attempts) = melhor resultado
   • 3-5 tentativas geralmente é suficiente

4. PROBLEMAS COMUNS
   ─────────────────
   • "Nenhuma peça carregada": Verifique se o DXF tem LWPOLYLINE
   • "CUDA out of memory": Use device='cpu'
   • "Checkpoint não encontrado": Verifique pasta scripts/

═══════════════════════════════════════════════════════════════════════════
""")


# =============================================================================
# CHECKLIST
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════
✅ CHECKLIST ANTES DE EXECUTAR
═══════════════════════════════════════════════════════════════════════════

□ Instalei ezdxf (pip install ezdxf)
□ Instalei svgpathtools (pip install svgpathtools)
□ Tenho um checkpoint treinado (.pt)
□ Meu arquivo CAD usa LWPOLYLINE
□ As peças estão em milímetros
□ Defini o tamanho da chapa correto

═══════════════════════════════════════════════════════════════════════════
🚀 PRÓXIMOS PASSOS
═══════════════════════════════════════════════════════════════════════════

1. Copie o script completo acima para: run_nesting_cad.py

2. Edite a seção CONFIGURAÇÃO:
   - INPUT_FILE = "seu_arquivo.dxf"
   - CONTAINER_WIDTH = sua_largura
   - CONTAINER_HEIGHT = sua_altura

3. Execute:
   python run_nesting_cad.py

4. Abra nesting_result.dxf no AutoCAD

═══════════════════════════════════════════════════════════════════════════

Pronto! Agora você pode fazer nesting de peças CAD reais! 🎉
""")