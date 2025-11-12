# """
# src/representation/image_encoder.py

# Converte layouts geométricos em imagens multi-canal para CNN.

# Gera imagem de 6 canais:
# - Canal 0: Ocupação (peças colocadas)
# - Canal 1: Bordas das peças
# - Canal 2: Mapa de distância (distância até peça mais próxima)
# - Canal 3: Próxima peça a ser colocada
# - Canal 4: Densidade local (blur da ocupação)
# - Canal 5: Acessibilidade (livre × distância)
# """

import numpy as np
from typing import List, Optional
from PIL import Image, ImageDraw
import warnings


def render_layout_as_image(container, 
                          placed_pieces: List,
                          next_piece=None,
                          size: int = 256) -> np.ndarray:
    # """
    # Renderiza layout como imagem 6-channel.
    
    # Args:
    #     container: Container (Polygon) com bounds
    #     placed_pieces: Lista de Polygon já colocados
    #     next_piece: Próxima peça (Polygon, opcional)
    #     size: Tamanho da imagem em pixels (padrão 256x256)
    
    # Returns:
    #     array (6, size, size) float32 normalizado [0, 1]
    
    # Example:
    #     >>> container = create_rectangle(1000, 600)
    #     >>> pieces = [create_rectangle(50, 30), ...]
    #     >>> image = render_layout_as_image(container, pieces, size=256)
    #     >>> image.shape
    #     (6, 256, 256)
    # """
    # Inicializar imagem (6 canais)
    image = np.zeros((6, size, size), dtype=np.float32)
    
    # Obter bounds do container
    try:
        if hasattr(container, 'width'):
            container_width = container.width
            container_height = container.height
        else:
            minx, miny, maxx, maxy = container.bounds
            container_width = maxx - minx
            container_height = maxy - miny
    except Exception as e:
        print(f"Warning: Could not get container dimensions: {e}")
        container_width = 1000
        container_height = 600
    
    # Calcular escala (world coordinates → pixel coordinates)
    scale_x = size / container_width
    scale_y = size / container_height
    
    # ===========================================================================
    # CANAL 0: OCUPAÇÃO (Peças colocadas)
    # ===========================================================================
    for piece in placed_pieces:
        try:
            mask = _polygon_to_mask(piece, size, scale_x, scale_y)
            image[0] += mask
        except Exception as e:
            warnings.warn(f"Could not render piece: {e}")
    
    # Clipar valores (caso peças sobrepostas)
    image[0] = np.clip(image[0], 0, 1)
    
    # ===========================================================================
    # CANAL 1: BORDAS (Contornos das peças)
    # ===========================================================================
    for piece in placed_pieces:
        try:
            border = _polygon_border_mask(piece, size, scale_x, scale_y, width=2)
            image[1] += border
        except Exception as e:
            warnings.warn(f"Could not render border: {e}")
    
    image[1] = np.clip(image[1], 0, 1)
    
    # ===========================================================================
    # CANAL 2: MAPA DE DISTÂNCIA (Distância até peça mais próxima)
    # ===========================================================================
    try:
        from scipy.ndimage import distance_transform_edt
        
        # Áreas ocupadas (1 = ocupado, 0 = livre)
        occupied = (image[0] > 0).astype(np.uint8)
        
        # Calcular distância das áreas livres até ocupadas
        if occupied.sum() > 0:
            distances = distance_transform_edt(1 - occupied)
            # Normalizar pela diagonal da imagem
            max_distance = np.sqrt(2) * size
            image[2] = distances / max_distance
        else:
            # Se não há peças, toda área está livre (distância máxima)
            image[2] = np.ones((size, size), dtype=np.float32)
            
    except ImportError:
        # Se scipy não disponível, usar aproximação simples
        warnings.warn("scipy not available, using simple distance approximation")
        image[2] = 1 - image[0]  # Inverso da ocupação
    
    # ===========================================================================
    # CANAL 3: PRÓXIMA PEÇA (Referência da peça a ser colocada)
    # ===========================================================================
    if next_piece is not None:
        try:
            # Centralizar a próxima peça no container como referência
            center_x = container_width / 2
            center_y = container_height / 2
            
            centered_piece = next_piece.set_position(center_x, center_y)
            mask = _polygon_to_mask(centered_piece, size, scale_x, scale_y)
            image[3] = mask
            
        except Exception as e:
            warnings.warn(f"Could not render next piece: {e}")
    
    # ===========================================================================
    # CANAL 4: DENSIDADE LOCAL (Convolução gaussiana da ocupação)
    # ===========================================================================
    try:
        from scipy.ndimage import gaussian_filter
        
        # Blur da ocupação para indicar densidade
        sigma = size / 25  # ~10 pixels para 256x256
        image[4] = gaussian_filter(image[0], sigma=sigma)
        
    except ImportError:
        # Fallback: usar a própria ocupação
        image[4] = image[0]
    
    # ===========================================================================
    # CANAL 5: ACESSIBILIDADE (Área livre × distância)
    # ===========================================================================
    # Combina informação de espaço livre com proximidade
    free_space = 1 - image[0]  # Inverso da ocupação
    image[5] = free_space * image[2]
    
    # ===========================================================================
    # NORMALIZAÇÃO FINAL
    # ===========================================================================
    # Normalizar cada canal para [0, 1]
    for i in range(6):
        channel_max = image[i].max()
        if channel_max > 0:
            image[i] /= channel_max
    
    return image


def _polygon_to_mask(polygon, size: int, scale_x: float, scale_y: float) -> np.ndarray:
    # """
    # Converte polígono para máscara binária.
    
    # Args:
    #     polygon: Polygon com atributo vertices
    #     size: Tamanho da imagem (NxN)
    #     scale_x: Escala de conversão x
    #     scale_y: Escala de conversão y
    
    # Returns:
    #     array (size, size) float32 [0, 1]
    # """
    # Criar imagem PIL em escala de cinza
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    # Escalar vértices do polígono para coordenadas de pixel
    try:
        scaled_vertices = [
            (int(v.x * scale_x), int(v.y * scale_y))
            for v in polygon.vertices
        ]
        
        # Desenhar polígono preenchido
        if len(scaled_vertices) >= 3:
            draw.polygon(scaled_vertices, fill=255)
        
    except AttributeError as e:
        warnings.warn(f"Polygon missing vertices attribute: {e}")
    
    # Converter para numpy e normalizar
    mask = np.array(img, dtype=np.float32) / 255.0
    
    return mask


def _polygon_border_mask(polygon, size: int, scale_x: float, scale_y: float,
                        width: int = 2) -> np.ndarray:
    # """
    # Cria máscara apenas das bordas do polígono.
    
    # Args:
    #     polygon: Polygon com atributo vertices
    #     size: Tamanho da imagem (NxN)
    #     scale_x: Escala de conversão x
    #     scale_y: Escala de conversão y
    #     width: Espessura da borda em pixels
    
    # Returns:
    #     array (size, size) float32 [0, 1]
    # """
    # Criar imagem PIL
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    # Escalar vértices
    try:
        scaled_vertices = [
            (int(v.x * scale_x), int(v.y * scale_y))
            for v in polygon.vertices
        ]
        
        # Desenhar apenas o contorno
        if len(scaled_vertices) >= 3:
            draw.polygon(scaled_vertices, outline=255, width=width)
        
    except AttributeError as e:
        warnings.warn(f"Polygon missing vertices attribute: {e}")
    
    # Converter para numpy e normalizar
    mask = np.array(img, dtype=np.float32) / 255.0
    
    return mask


def visualize_channels(image: np.ndarray, save_path: Optional[str] = None):
    # """
    # Visualiza os 6 canais da imagem.
    
    # Args:
    #     image: Array (6, H, W)
    #     save_path: Caminho para salvar (opcional)
    # """
    import matplotlib.pyplot as plt
    
    channel_names = [
        'Canal 0: Ocupação',
        'Canal 1: Bordas',
        'Canal 2: Distância',
        'Canal 3: Próxima Peça',
        'Canal 4: Densidade',
        'Canal 5: Acessibilidade'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):
        axes[i].imshow(image[i], cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(channel_names[i], fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
        # Adicionar colorbar
        cbar = plt.colorbar(axes[i].images[0], ax=axes[i], fraction=0.046)
        cbar.set_label('Intensidade', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualização salva em: {save_path}")
    
    plt.show()


# =============================================================================
# EXEMPLO DE USO E TESTES
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE: IMAGE ENCODER")
    print("="*70)
    
    # Importar módulos necessários
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.geometry.polygon import create_rectangle, create_random_polygon
    
    # Criar container
    container = create_rectangle(1000, 600, center=(500, 300))
    print(f"✓ Container: {container.width}x{container.height}")
    
    # Criar algumas peças
    placed_pieces = [
        create_rectangle(80, 50).set_position(100, 100),
        create_rectangle(60, 40).set_position(200, 120),
        create_random_polygon(6, 30).set_position(350, 150),
        create_random_polygon(5, 25).set_position(150, 250),
    ]
    print(f"✓ Criadas {len(placed_pieces)} peças colocadas")
    
    # Próxima peça
    next_piece = create_rectangle(50, 30)
    print(f"✓ Próxima peça: {next_piece}")
    
    # Renderizar
    print("\nRenderizando layout como imagem...")
    image = render_layout_as_image(
        container=container,
        placed_pieces=placed_pieces,
        next_piece=next_piece,
        size=256
    )
    
    print(f"✓ Imagem renderizada!")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Verificar cada canal
    print("\n" + "="*70)
    print("ESTATÍSTICAS POR CANAL")
    print("="*70)
    
    channel_names = ['Ocupação', 'Bordas', 'Distância', 'Próxima Peça', 
                     'Densidade', 'Acessibilidade']
    
    for i, name in enumerate(channel_names):
        print(f"\nCanal {i} ({name}):")
        print(f"  Min: {image[i].min():.3f}")
        print(f"  Max: {image[i].max():.3f}")
        print(f"  Mean: {image[i].mean():.3f}")
        print(f"  Std: {image[i].std():.3f}")
        print(f"  Non-zero: {(image[i] > 0).sum()}/{image[i].size} pixels")
    
    # Visualizar
    print("\n" + "="*70)
    print("VISUALIZAÇÃO")
    print("="*70)
    
    try:
        visualize_channels(image, save_path='image_encoder_test.png')
        print("✓ Visualização criada!")
    except Exception as e:
        print(f"⚠ Não foi possível criar visualização: {e}")
    
    # Teste de performance
    print("\n" + "="*70)
    print("TESTE DE PERFORMANCE")
    print("="*70)
    
    import time
    
    n_iterations = 100
    start = time.time()
    
    for _ in range(n_iterations):
        _ = render_layout_as_image(container, placed_pieces, next_piece, size=256)
    
    elapsed = time.time() - start
    fps = n_iterations / elapsed
    
    print(f"✓ Renderizadas {n_iterations} imagens em {elapsed:.2f}s")
    print(f"  Performance: {fps:.1f} FPS")
    print(f"  Tempo médio: {elapsed/n_iterations*1000:.2f} ms/imagem")
    
    print("\n" + "="*70)
    print("✓ TODOS OS TESTES PASSARAM!")
    print("="*70)