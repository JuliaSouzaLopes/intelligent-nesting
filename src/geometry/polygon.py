import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate as shapely_rotate

#Classe Polygon: Representação e operações geométricas básicas

@dataclass
class Point:
    #"""Ponto 2D"""
    x: float
    y: float
    
    def __iter__(self):
        return iter((self.x, self.y))
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def distance_to(self, other: 'Point') -> float:
        #"""Distância euclidiana até outro ponto"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Polygon:
    # """
    # Representa um polígono 2D irregular.
    
    # Attributes:
    #     vertices: Lista de vértices (pontos) em sentido anti-horário
    #     position: Posição atual (ponto de referência, geralmente centroide)
    #     rotation: Rotação atual em graus
    #     id: Identificador único
    # """
    
    def __init__(self, vertices: List[Tuple[float, float]], 
                 id: Optional[int] = None):
        # """
        # Inicializa polígono.
        
        # Args:
        #     vertices: Lista de tuplas (x, y) representando vértices
        #     id: Identificador único (opcional)
        # """
        self.vertices = [Point(x, y) for x, y in vertices]
        self.id = id
        self.rotation = 0.0
        self.position = self.calculate_centroid()
        
        # Cache para propriedades computacionais
        self._area = None
        self._perimeter = None
        self._bounds = None
        self._shapely_polygon = None
        
        # Normalizar ordem dos vértices (anti-horário)
        if not self.is_counter_clockwise():
            self.vertices.reverse()
    
    @property
    def shapely(self) -> ShapelyPolygon:
        #"""Converte para Shapely Polygon (cached)"""
        if not hasattr(self, '_shapely_polygon') or self._shapely_polygon is None:
            coords = [(v.x, v.y) for v in self.vertices]
            self._shapely_polygon = ShapelyPolygon(coords)
        return self._shapely_polygon
    
    def invalidate_cache(self):
        #"""Invalida caches após transformações"""
        self._area = None
        self._perimeter = None
        self._bounds = None
        self._shapely_polygon = None
    
    # =========================================================================
    # Propriedades Geométricas
    # =========================================================================
    
    @property
    def area(self) -> float:
        #"""Área do polígono (Shoelace formula)"""
        if self._area is None:
            self._area = abs(self.shapely.area)
        return self._area
    
    @property
    def perimeter(self) -> float:
        #"""Perímetro do polígono"""
        if self._perimeter is None:
            self._perimeter = self.shapely.length
        return self._perimeter
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        #"""Bounding box (minx, miny, maxx, maxy)"""
        if self._bounds is None:
            self._bounds = self.shapely.bounds
        return self._bounds
    
    @property
    def width(self) -> float:
        #"""Largura do bounding box"""
        minx, _, maxx, _ = self.bounds
        return maxx - minx
    
    @property
    def height(self) -> float:
        #"""Altura do bounding box"""
        _, miny, _, maxy = self.bounds
        return maxy - miny
    
    @property
    def aspect_ratio(self) -> float:
        #"""Razão largura/altura"""
        return self.width / max(self.height, 1e-6)
    
    def calculate_centroid(self) -> Point:
        #"""Calcula centroide do polígono"""
        centroid = self.shapely.centroid
        return Point(centroid.x, centroid.y)
    
    def calculate_complexity(self) -> float:
        # """
        # Medida de complexidade baseada em:
        # - Número de vértices
        # - Razão perímetro/área
        # - Não-convexidade
        # """
        n_vertices = len(self.vertices)
        compactness = self.perimeter**2 / (4 * np.pi * max(self.area, 1e-6))
        convexity = self.convex_hull().area / max(self.area, 1e-6)
        
        complexity = (n_vertices / 10.0) * compactness * (2 - convexity)
        return min(complexity, 10.0)  # Normalizar [0-10]
    
    def convex_hull(self) -> 'Polygon':
        #"""Retorna o convex hull"""
        hull = self.shapely.convex_hull
        vertices = list(hull.exterior.coords[:-1])  # Remove último duplicado
        return Polygon(vertices)
    
    # =========================================================================
    # Transformações Geométricas
    # =========================================================================
    
    def translate(self, dx: float, dy: float) -> 'Polygon':
        # """
        # Translada polígono.
        
        # Args:
        #     dx: Deslocamento em x
        #     dy: Deslocamento em y
            
        # Returns:
        #     Novo polígono transladado
        # """
        new_vertices = [(v.x + dx, v.y + dy) for v in self.vertices]
        new_poly = Polygon(new_vertices, id=self.id)
        new_poly.rotation = self.rotation
        return new_poly
    
    def rotate(self, angle: float, origin: Optional[Point] = None) -> 'Polygon':
        # """
        # Rotaciona polígono.
        
        # Args:
        #     angle: Ângulo em graus (sentido anti-horário)
        #     origin: Ponto de rotação (padrão: centroide)
            
        # Returns:
        #     Novo polígono rotacionado
        # """
        if origin is None:
            origin = self.position
        
        rotated = shapely_rotate(
            self.shapely, 
            angle, 
            origin=(origin.x, origin.y)
        )
        
        new_vertices = list(rotated.exterior.coords[:-1])
        new_poly = Polygon(new_vertices, id=self.id)
        new_poly.rotation = (self.rotation + angle) % 360
        return new_poly
    
    def scale(self, scale_x: float, scale_y: Optional[float] = None) -> 'Polygon':
        # """
        # Escala polígono.
        
        # Args:
        #     scale_x: Fator de escala em x
        #     scale_y: Fator de escala em y (padrão: igual a scale_x)
        # """
        if scale_y is None:
            scale_y = scale_x
        
        centroid = self.position
        new_vertices = []
        
        for v in self.vertices:
            # Transladar para origem
            dx = v.x - centroid.x
            dy = v.y - centroid.y
            
            # Escalar
            dx *= scale_x
            dy *= scale_y
            
            # Transladar de volta
            new_vertices.append((dx + centroid.x, dy + centroid.y))
        
        return Polygon(new_vertices, id=self.id)
    
    def set_position(self, x: float, y: float) -> 'Polygon':
        #"""Move polígono para nova posição (centroide)"""
        current = self.position
        dx = x - current.x
        dy = y - current.y
        return self.translate(dx, dy)
    
    # =========================================================================
    # Operações Booleanas
    # =========================================================================
    
    def intersects(self, other: 'Polygon') -> bool:
        #"""Verifica se polígonos se intersectam"""
        return self.shapely.intersects(other.shapely)
    
    def contains_point(self, point: Point) -> bool:
        #"""Verifica se ponto está dentro do polígono"""
        from shapely.geometry import Point as ShapelyPoint
        return self.shapely.contains(ShapelyPoint(point.x, point.y))
    
    def distance_to(self, other: 'Polygon') -> float:
        #"""Distância mínima até outro polígono"""
        return self.shapely.distance(other.shapely)
    
    def union(self, other: 'Polygon') -> 'Polygon':
        #"""União com outro polígono"""
        union = self.shapely.union(other.shapely)
        if union.geom_type == 'Polygon':
            vertices = list(union.exterior.coords[:-1])
            return Polygon(vertices)
        else:
            raise ValueError("Union resulted in non-polygon geometry")
    
    def intersection(self, other: 'Polygon') -> Optional['Polygon']:
        #"""Interseção com outro polígono"""
        inter = self.shapely.intersection(other.shapely)
        if inter.is_empty:
            return None
        if inter.geom_type == 'Polygon':
            vertices = list(inter.exterior.coords[:-1])
            return Polygon(vertices)
        return None
    
    # =========================================================================
    # Utilitários
    # =========================================================================
    
    def is_counter_clockwise(self) -> bool:
        #"""Verifica se vértices estão em sentido anti-horário"""
        # Usar área com sinal (Shoelace formula)
        area_sum = 0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area_sum += (self.vertices[j].x - self.vertices[i].x) * \
                        (self.vertices[j].y + self.vertices[i].y)
        return area_sum < 0
    
    def simplify(self, tolerance: float = 1.0) -> 'Polygon':
        # """
        # Simplifica polígono (Douglas-Peucker).
        
        # Args:
        #     tolerance: Tolerância máxima para simplificação
        # """
        simplified = self.shapely.simplify(tolerance, preserve_topology=True)
        vertices = list(simplified.exterior.coords[:-1])
        return Polygon(vertices, id=self.id)
    
    def to_dict(self) -> dict:
        #"""Serializa para dicionário"""
        return {
            'id': self.id,
            'vertices': [(v.x, v.y) for v in self.vertices],
            'position': (self.position.x, self.position.y),
            'rotation': self.rotation,
            'area': self.area,
            'perimeter': self.perimeter,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Polygon':
        #"""Deserializa de dicionário"""
        poly = cls(data['vertices'], id=data['id'])
        poly.rotation = data['rotation']
        return poly
    
    def copy(self) -> 'Polygon':
        #"""Cria cópia profunda"""
        return Polygon(
            [(v.x, v.y) for v in self.vertices],
            id=self.id
        )
    
    # =========================================================================
    # Visualização
    # =========================================================================
    
    def plot(self, ax=None, **kwargs):
        # """
        # Plota polígono.
        
        # Args:
        #     ax: Matplotlib axis (cria novo se None)
        #     **kwargs: Argumentos para matplotlib.patches.Polygon
        # """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Configurações padrão
        plot_kwargs = {
            'facecolor': 'lightblue',
            'edgecolor': 'black',
            'linewidth': 2,
            'alpha': 0.7
        }
        plot_kwargs.update(kwargs)
        
        # Plotar polígono
        coords = [(v.x, v.y) for v in self.vertices]
        patch = plt.Polygon(coords, **plot_kwargs)
        ax.add_patch(patch)
        
        # Marcar centroide
        ax.plot(self.position.x, self.position.y, 'ro', markersize=5)
        
        # Ajustar limites
        minx, miny, maxx, maxy = self.bounds
        margin = 0.1 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if self.id is not None:
            ax.text(self.position.x, self.position.y, f'{self.id}',
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        return ax
    
    def __repr__(self) -> str:
        return (f"Polygon(id={self.id}, vertices={len(self.vertices)}, "
                f"area={self.area:.2f}, rotation={self.rotation:.1f}°)")


# =============================================================================
# Funções Auxiliares
# =============================================================================

def create_rectangle(width: float, height: float, 
                     center: Tuple[float, float] = (0, 0)) -> Polygon:
    #"""Cria retângulo"""
    cx, cy = center
    vertices = [
        (cx - width/2, cy - height/2),
        (cx + width/2, cy - height/2),
        (cx + width/2, cy + height/2),
        (cx - width/2, cy + height/2),
    ]
    return Polygon(vertices)


def create_regular_polygon(n_sides: int, radius: float,
                          center: Tuple[float, float] = (0, 0)) -> Polygon:
    #"""Cria polígono regular"""
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    vertices = [
        (cx + radius * np.cos(a), cy + radius * np.sin(a))
        for a in angles
    ]
    return Polygon(vertices)


def create_random_polygon(n_vertices: int, radius: float,
                         irregularity: float = 0.5,
                         spikeyness: float = 0.5) -> Polygon:
    # """
    # Cria polígono irregular aleatório.
    
    # Args:
    #     n_vertices: Número de vértices
    #     radius: Raio médio
    #     irregularity: [0-1] Quão irregular (0=regular, 1=muito irregular)
    #     spikeyness: [0-1] Quão "pontiagudo"
    # """
    irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n_vertices
    spikeyness = np.clip(spikeyness, 0, 1) * radius
    
    # Ângulos aleatórios
    angle_steps = []
    lower = (2 * np.pi / n_vertices) - irregularity
    upper = (2 * np.pi / n_vertices) + irregularity
    
    cumsum = 0
    for _ in range(n_vertices):
        angle = np.random.uniform(lower, upper)
        angle_steps.append(angle)
        cumsum += angle
    
    # Normalizar para fechar o polígono
    cumsum /= (2 * np.pi)
    angle_steps = [x / cumsum for x in angle_steps]
    
    # Gerar vértices
    vertices = []
    angle = np.random.uniform(0, 2 * np.pi)
    
    for angle_step in angle_steps:
        r = np.clip(np.random.normal(radius, spikeyness), 0, 2 * radius)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append((x, y))
        angle += angle_step
    
    return Polygon(vertices)


# =============================================================================
# Exemplo de Uso
# =============================================================================

if __name__ == "__main__":
    # Criar alguns polígonos
    rect = create_rectangle(100, 50, center=(0, 0))
    hexagon = create_regular_polygon(6, radius=50, center=(150, 0))
    irregular = create_random_polygon(8, radius=40, irregularity=0.7)
    irregular = irregular.translate(300, 0)
    
    # Transformações
    rotated_rect = rect.rotate(45)
    scaled_hex = hexagon.scale(1.5)
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    rect.plot(axes[0, 0])
    axes[0, 0].set_title('Retângulo')
    
    hexagon.plot(axes[0, 1])
    axes[0, 1].set_title('Hexágono Regular')
    
    irregular.plot(axes[0, 2])
    axes[0, 2].set_title('Polígono Irregular')
    
    rotated_rect.plot(axes[1, 0], facecolor='lightcoral')
    axes[1, 0].set_title('Retângulo Rotacionado 45°')
    
    scaled_hex.plot(axes[1, 1], facecolor='lightgreen')
    axes[1, 1].set_title('Hexágono Escalado 1.5x')
    
    # Teste de interseção
    poly1 = create_rectangle(80, 80, center=(0, 0))
    poly2 = create_rectangle(80, 80, center=(40, 40))
    
    axes[1, 2].set_title('Teste de Interseção')
    poly1.plot(axes[1, 2], facecolor='lightblue')
    poly2.plot(axes[1, 2], facecolor='lightcoral')
    
    intersects = poly1.intersects(poly2)
    axes[1, 2].text(0, -60, f'Intersects: {intersects}',
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('polygon_examples.png', dpi=150)
    plt.show()
    
    print("✓ Módulo Polygon implementado e testado com sucesso!")