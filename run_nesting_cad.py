#!/usr/bin/env python3
# """
# run_nesting_cad.py - Execute nesting com arquivos CAD

# COMO USAR:
# 1. Edite a seção CONFIGURAÇÃO abaixo
# 2. Execute: python run_nesting_cad.py
# 3. Abra nesting_result.dxf no AutoCAD

# Formatos suportados: DXF, SVG, JSON
# """

import sys
import glob
from pathlib import Path


# =============================================================================
# CONFIGURAÇÃO - EDITE AQUI!
# =============================================================================

# Arquivo de entrada (escolha um):
INPUT_FILE = "pecas.dxf"           # Arquivo DXF (AutoCAD)
# INPUT_FILE = "pecas.svg"         # Arquivo SVG
# INPUT_FILE = "pecas.json"        # Arquivo JSON (veja pieces_exemplo.json)

# Dimensões da chapa (em milímetros)
CONTAINER_WIDTH = 1000    # Largura da chapa
CONTAINER_HEIGHT = 600    # Altura da chapa

# Número de tentativas (mais = melhor resultado, mais tempo)
MAX_ATTEMPTS = 3

# Device: 'cuda' para GPU ou 'cpu' se não tiver GPU
DEVICE = 'cuda'

# Pasta de checkpoints
CHECKPOINT_DIR = "scripts"


# =============================================================================
# NÃO EDITE ABAIXO (a menos que saiba o que está fazendo)
# =============================================================================

def verificar_dependencias():
    #"""Verifica se dependências estão instaladas"""
    print("Verificando dependências...")
    
    # PyTorch
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        
        if DEVICE == 'cuda' and not torch.cuda.is_available():
            print("   ⚠️  CUDA não disponível, usando CPU")
            return 'cpu'
        elif DEVICE == 'cuda':
            print(f"   ✓ CUDA disponível")
            
    except ImportError:
        print("   ❌ PyTorch não instalado!")
        print("      Execute: pip install torch --break-system-packages")
        return None
    
    # ezdxf (para DXF)
    ext = Path(INPUT_FILE).suffix.lower()
    if ext == '.dxf':
        try:
            import ezdxf
            print(f"   ✓ ezdxf {ezdxf.__version__}")
        except ImportError:
            print("   ❌ ezdxf não instalado!")
            print("      Execute: pip install ezdxf --break-system-packages")
            return None
    
    # svgpathtools (para SVG)
    if ext == '.svg':
        try:
            import svgpathtools
            print("   ✓ svgpathtools")
        except ImportError:
            print("   ❌ svgpathtools não instalado!")
            print("      Execute: pip install svgpathtools --break-system-packages")
            return None
    
    return DEVICE


def encontrar_checkpoint():
    #"""Encontra checkpoint mais recente"""
    print(f"\nBuscando checkpoint em '{CHECKPOINT_DIR}/'...")
    
    # Procura na pasta configurada
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/*.pt")
    
    if not checkpoints:
        # Tenta na pasta atual
        checkpoints = glob.glob("*.pt")
    
    if not checkpoints:
        print("   ❌ Nenhum checkpoint encontrado!")
        print(f"\n   Verifique se há arquivos .pt em '{CHECKPOINT_DIR}/'")
        print("   Ou execute o treinamento primeiro.")
        return None
    
    # Usa o mais recente por data de modificação
    checkpoint_path = max(checkpoints, key=lambda x: Path(x).stat().st_mtime)
    print(f"   ✓ Usando: {checkpoint_path}")
    
    return checkpoint_path


def carregar_pecas():
    #"""Carrega peças do arquivo"""
    # Importa aqui para evitar erro se dependências não estão instaladas
    try:
        from use_trained_model import PieceLoader
    except ImportError:
        print("   ❌ use_trained_model.py não encontrado!")
        print("   Certifique-se de que está no diretório correto.")
        return None
    
    print(f"\nCarregando peças de '{INPUT_FILE}'...")
    
    # Verifica se arquivo existe
    if not Path(INPUT_FILE).exists():
        print(f"   ❌ Arquivo não encontrado: {INPUT_FILE}")
        print(f"\n   Verifique o caminho do arquivo.")
        return None
    
    loader = PieceLoader()
    
    # Detecta formato pela extensão
    ext = Path(INPUT_FILE).suffix.lower()
    
    try:
        if ext == '.dxf':
            pieces = loader.from_dxf(INPUT_FILE)
        elif ext == '.svg':
            pieces = loader.from_svg(INPUT_FILE)
        elif ext == '.json':
            pieces = loader.from_json(INPUT_FILE)
        else:
            print(f"   ❌ Formato não suportado: {ext}")
            print("   Use: .dxf, .svg, ou .json")
            return None
            
    except Exception as e:
        print(f"   ❌ Erro ao carregar arquivo: {e}")
        return None
    
    if not pieces:
        print("   ❌ Nenhuma peça encontrada no arquivo!")
        print("\n   Para arquivos DXF:")
        print("   - Use LWPOLYLINE para desenhar peças")
        print("   - Feche todos os polígonos")
        return None
    
    print(f"   ✓ Carregadas {len(pieces)} peças")
    
    return pieces


def executar_nesting(pieces, checkpoint_path, device):
    #"""Executa o nesting"""
    from use_trained_model import NestingSystem
    
    print(f"\nCriando sistema de nesting...")
    print(f"   Container: {CONTAINER_WIDTH}mm × {CONTAINER_HEIGHT}mm")
    
    try:
        system = NestingSystem(
            checkpoint_path=checkpoint_path,
            container_width=CONTAINER_WIDTH,
            container_height=CONTAINER_HEIGHT,
            device=device
        )
    except Exception as e:
        print(f"   ❌ Erro ao criar sistema: {e}")
        if 'CUDA' in str(e):
            print("\n   Tente mudar DEVICE = 'cpu' no início do script")
        return None
    
    print(f"\nExecutando nesting ({MAX_ATTEMPTS} tentativas)...")
    
    try:
        result = system.nest_pieces(
            pieces,
            max_attempts=MAX_ATTEMPTS,
            visualize=True
        )
    except Exception as e:
        print(f"   ❌ Erro durante nesting: {e}")
        return None
    
    return result


def exportar_resultados(result):
    #"""Exporta resultados"""
    from use_trained_model import ResultExporter
    
    print("\nExportando resultados...")
    exporter = ResultExporter()
    
    # JSON (dados completos)
    try:
        exporter.to_json(result, "nesting_result.json")
        print("   ✓ nesting_result.json")
    except Exception as e:
        print(f"   ⚠️ JSON: {e}")
    
    # SVG (para CAD)
    try:
        exporter.to_svg(result, "nesting_result.svg")
        print("   ✓ nesting_result.svg")
    except Exception as e:
        print(f"   ⚠️ SVG: {e}")
    
    # DXF (para AutoCAD)
    try:
        exporter.to_dxf(result, "nesting_result.dxf")
        print("   ✓ nesting_result.dxf")
    except ImportError:
        print("   ⚠️ DXF não exportado (pip install ezdxf)")
    except Exception as e:
        print(f"   ⚠️ DXF: {e}")


def main():
    #"""Função principal"""
    
    print("=" * 70)
    print("🎯 NESTING COM ARQUIVOS CAD")
    print("=" * 70)
    
    # 1. Verificar dependências
    device = verificar_dependencias()
    if device is None:
        return 1
    
    # 2. Encontrar checkpoint
    checkpoint_path = encontrar_checkpoint()
    if checkpoint_path is None:
        return 1
    
    # 3. Carregar peças
    pieces = carregar_pecas()
    if pieces is None:
        return 1
    
    # 4. Executar nesting
    result = executar_nesting(pieces, checkpoint_path, device)
    if result is None:
        return 1
    
    # 5. Mostrar resultado
    print("\n" + "=" * 70)
    print("📊 RESULTADO")
    print("=" * 70)
    print(f"Utilização: {result['utilization']*100:.2f}%")
    print(f"Peças colocadas: {result['n_placed']}/{len(pieces)}")
    if 'execution_time' in result:
        print(f"Tempo: {result['execution_time']:.2f}s")
    
    # 6. Exportar
    exportar_resultados(result)
    
    # 7. Finalizar
    print("\n" + "=" * 70)
    print("✅ PRONTO!")
    print("=" * 70)
    print(f"""
Arquivos gerados:
  • nesting_result.png   ← Visualização
  • nesting_result.json  ← Dados completos
  • nesting_result.svg   ← Para importar no CAD
  • nesting_result.dxf   ← Para abrir no AutoCAD

Agora você pode:
1. Visualizar: abra nesting_result.png
2. AutoCAD: abra nesting_result.dxf
3. Outros CAD: importe nesting_result.svg

Utilização alcançada: {result['utilization']*100:.2f}%
""")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelado pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)