# """
# src/evaluation/metrics.py

# Métricas para avaliar qualidade de soluções de nesting.
# """

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.geometry.polygon import Polygon


@dataclass
class NestingMetrics:
    #"""Métricas de uma solução de nesting"""
    
    # Métricas básicas
    utilization: float  # Taxa de utilização [0,1]
    n_placed: int  # Número de peças colocadas
    n_total: int  # Número total de peças
    placement_rate: float  # Taxa de colocação [0,1]
    
    # Métricas de qualidade
    compactness: float  # Quão compacta é a solução
    balance: float  # Distribuição de peso
    stability: float  # Estabilidade da solução
    
    # Métricas de espaço
    wasted_area: float  # Área desperdiçada
    holes_penalty: float  # Penalidade por buracos
    perimeter_utilization: float  # Uso de perímetro
    
    # Métricas de tempo
    solving_time: float  # Tempo de solução (segundos)
    n_attempts: int  # Número de tentativas
    
    def to_dict(self) -> Dict:
        #"""Converte para dicionário"""
        return {
            'utilization': self.utilization,
            'n_placed': self.n_placed,
            'n_total': self.n_total,
            'placement_rate': self.placement_rate,
            'compactness': self.compactness,
            'balance': self.balance,
            'stability': self.stability,
            'wasted_area': self.wasted_area,
            'holes_penalty': self.holes_penalty,
            'perimeter_utilization': self.perimeter_utilization,
            'solving_time': self.solving_time,
            'n_attempts': self.n_attempts
        }
    
    def __repr__(self) -> str:
        return (f"NestingMetrics(util={self.utilization:.2%}, "
                f"placed={self.n_placed}/{self.n_total}, "
                f"time={self.solving_time:.2f}s)")


class MetricsCalculator:
    #"""Calcula métricas de qualidade de nesting"""
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self,
                             container: Polygon,
                             placed_pieces: List[Polygon],
                             total_pieces: List[Polygon],
                             solving_time: float = 0.0,
                             n_attempts: int = 1) -> NestingMetrics:
        # """
        # Calcula todas as métricas de uma solução.
        
        # Args:
        #     container: Container (chapa)
        #     placed_pieces: Peças colocadas
        #     total_pieces: Todas as peças (incluindo não colocadas)
        #     solving_time: Tempo de solução
        #     n_attempts: Número de tentativas
            
        # Returns:
        #     NestingMetrics com todas as métricas
        # """
        n_placed = len(placed_pieces)
        n_total = len(total_pieces)
        
        # Métricas básicas
        utilization = self.calculate_utilization(container, placed_pieces)
        placement_rate = n_placed / max(n_total, 1)
        
        # Métricas de qualidade
        compactness = self.calculate_compactness(container, placed_pieces)
        balance = self.calculate_balance(container, placed_pieces)
        stability = self.calculate_stability(placed_pieces)
        
        # Métricas de espaço
        wasted_area = container.area * (1 - utilization)
        holes_penalty = self.calculate_holes_penalty(container, placed_pieces)
        perimeter_util = self.calculate_perimeter_utilization(
            container, placed_pieces
        )
        
        return NestingMetrics(
            utilization=utilization,
            n_placed=n_placed,
            n_total=n_total,
            placement_rate=placement_rate,
            compactness=compactness,
            balance=balance,
            stability=stability,
            wasted_area=wasted_area,
            holes_penalty=holes_penalty,
            perimeter_utilization=perimeter_util,
            solving_time=solving_time,
            n_attempts=n_attempts
        )
    
    def calculate_utilization(self,
                             container: Polygon,
                             placed_pieces: List[Polygon]) -> float:
        # """
        # Calcula taxa de utilização.
        
        # Taxa = área_peças / área_container
        # """
        if len(placed_pieces) == 0:
            return 0.0
        
        total_pieces_area = sum(piece.area for piece in placed_pieces)
        container_area = container.area
        
        return min(total_pieces_area / container_area, 1.0)
    
    def calculate_compactness(self,
                             container: Polygon,
                             placed_pieces: List[Polygon]) -> float:
        # """
        # Calcula compactação da solução.
        
        # Compactação = 1 - (bounding_box_area / container_area)
        # """
        if len(placed_pieces) == 0:
            return 0.0
        
        # Calcular bounding box de todas as peças
        all_x = []
        all_y = []
        
        for piece in placed_pieces:
            for vertex in piece.vertices:
                all_x.append(vertex.x)
                all_y.append(vertex.y)
        
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        
        bbox_area = (maxx - minx) * (maxy - miny)
        container_area = container.area
        
        # Compactness: menor bounding box é melhor
        compactness = 1 - (bbox_area / container_area)
        
        return max(compactness, 0.0)
    
    def calculate_balance(self,
                         container: Polygon,
                         placed_pieces: List[Polygon]) -> float:
        # """
        # Calcula balanço da distribuição de peso.
        
        # Balance = 1 - distância_do_centro_de_massa_ao_centro_do_container
        # """
        if len(placed_pieces) == 0:
            return 1.0
        
        # Centro do container
        container_center = container.position
        
        # Centro de massa das peças (ponderado por área)
        total_area = sum(p.area for p in placed_pieces)
        
        com_x = sum(p.position.x * p.area for p in placed_pieces) / total_area
        com_y = sum(p.position.y * p.area for p in placed_pieces) / total_area
        
        # Distância do COM ao centro do container
        dx = com_x - container_center.x
        dy = com_y - container_center.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalizar pela diagonal do container
        diagonal = np.sqrt(container.width**2 + container.height**2)
        normalized_distance = distance / diagonal
        
        balance = 1 - normalized_distance
        
        return max(balance, 0.0)
    
    def calculate_stability(self, placed_pieces: List[Polygon]) -> float:
        # """
        # Calcula estabilidade baseado em contatos entre peças.
        
        # Mais contatos = mais estável
        # """
        if len(placed_pieces) < 2:
            return 1.0
        
        n_contacts = 0
        contact_threshold = 5.0  # mm
        
        for i, piece_a in enumerate(placed_pieces):
            for piece_b in placed_pieces[i+1:]:
                distance = piece_a.distance_to(piece_b)
                if distance < contact_threshold:
                    n_contacts += 1
        
        # Máximo de contatos possível
        max_contacts = len(placed_pieces) * (len(placed_pieces) - 1) / 2
        
        stability = n_contacts / max_contacts if max_contacts > 0 else 0.0
        
        return stability
    
    def calculate_holes_penalty(self,
                               container: Polygon,
                               placed_pieces: List[Polygon]) -> float:
        # """
        # Calcula penalidade por buracos (áreas inacessíveis).
        
        # Aproximação: conta número de regiões isoladas.
        # """
        # Simplificação: retornar 0 por enquanto
        # Implementação completa requer análise topológica complexa
        return 0.0
    
    def calculate_perimeter_utilization(self,
                                       container: Polygon,
                                       placed_pieces: List[Polygon]) -> float:
        # """
        # Calcula quanto do perímetro do container é usado.
        
        # Peças nas bordas = melhor utilização
        # """
        if len(placed_pieces) == 0:
            return 0.0
        
        minx, miny, maxx, maxy = container.bounds
        edge_threshold = 10.0  # mm da borda
        
        n_on_edge = 0
        
        for piece in placed_pieces:
            piece_minx, piece_miny, piece_maxx, piece_maxy = piece.bounds
            
            # Verificar se toca alguma borda
            on_edge = (
                abs(piece_minx - minx) < edge_threshold or
                abs(piece_maxx - maxx) < edge_threshold or
                abs(piece_miny - miny) < edge_threshold or
                abs(piece_maxy - maxy) < edge_threshold
            )
            
            if on_edge:
                n_on_edge += 1
        
        perimeter_util = n_on_edge / len(placed_pieces)
        
        return perimeter_util


class BenchmarkComparator:
    #"""Compara soluções com benchmarks"""
    
    def __init__(self):
        self.baseline_utilizations = {
            'random': 0.45,
            'greedy': 0.65,
            'genetic': 0.75,
            'optimal': 0.85
        }
    
    def compare_to_baseline(self,
                           metrics: NestingMetrics,
                           baseline: str = 'greedy') -> Dict:
        # """
        # Compara com baseline.
        
        # Args:
        #     metrics: Métricas da solução
        #     baseline: Nome do baseline
            
        # Returns:
        #     Dict com comparação
        # """
        if baseline not in self.baseline_utilizations:
            raise ValueError(f"Unknown baseline: {baseline}")
        
        baseline_util = self.baseline_utilizations[baseline]
        improvement = (metrics.utilization - baseline_util) / baseline_util
        
        return {
            'solution_utilization': metrics.utilization,
            'baseline_utilization': baseline_util,
            'absolute_improvement': metrics.utilization - baseline_util,
            'relative_improvement': improvement,
            'better_than_baseline': metrics.utilization > baseline_util
        }
    
    def calculate_score(self, metrics: NestingMetrics) -> float:
        # """
        # Calcula score global ponderado.
        
        # Score = weighted average de várias métricas
        # """
        weights = {
            'utilization': 0.40,
            'placement_rate': 0.25,
            'compactness': 0.15,
            'balance': 0.10,
            'stability': 0.10
        }
        
        score = (
            weights['utilization'] * metrics.utilization +
            weights['placement_rate'] * metrics.placement_rate +
            weights['compactness'] * metrics.compactness +
            weights['balance'] * metrics.balance +
            weights['stability'] * metrics.stability
        )
        
        return score


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE: METRICS")
    print("="*70)
    
    from src.geometry.polygon import create_rectangle, create_random_polygon
    
    # Criar container e peças
    container = create_rectangle(1000, 600, center=(500, 300))
    
    placed_pieces = [
        create_rectangle(100, 60).set_position(100, 100),
        create_rectangle(80, 50).set_position(220, 120),
        create_rectangle(90, 55).set_position(340, 110),
        create_random_polygon(6, 30).set_position(150, 250),
        create_random_polygon(5, 25).set_position(280, 270),
    ]
    
    total_pieces = placed_pieces + [
        create_rectangle(70, 40),
        create_rectangle(60, 35),
    ]
    
    print(f"Container: {container.width}x{container.height}")
    print(f"Peças colocadas: {len(placed_pieces)}/{len(total_pieces)}")
    
    # Calcular métricas
    calculator = MetricsCalculator()
    
    metrics = calculator.calculate_all_metrics(
        container=container,
        placed_pieces=placed_pieces,
        total_pieces=total_pieces,
        solving_time=12.5,
        n_attempts=3
    )
    
    print("\n" + "="*70)
    print("MÉTRICAS CALCULADAS")
    print("="*70)
    print(f"Utilização: {metrics.utilization:.2%}")
    print(f"Taxa de Colocação: {metrics.placement_rate:.2%}")
    print(f"Compactação: {metrics.compactness:.2%}")
    print(f"Balanço: {metrics.balance:.2%}")
    print(f"Estabilidade: {metrics.stability:.2%}")
    print(f"Área Desperdiçada: {metrics.wasted_area:.2f} mm²")
    print(f"Uso de Perímetro: {metrics.perimeter_utilization:.2%}")
    print(f"Tempo: {metrics.solving_time:.2f}s")
    
    # Comparar com baseline
    print("\n" + "="*70)
    print("COMPARAÇÃO COM BASELINE")
    print("="*70)
    
    comparator = BenchmarkComparator()
    
    for baseline in ['random', 'greedy', 'genetic']:
        comparison = comparator.compare_to_baseline(metrics, baseline)
        
        print(f"\nBaseline: {baseline}")
        print(f"  Baseline util: {comparison['baseline_utilization']:.2%}")
        print(f"  Nossa util: {comparison['solution_utilization']:.2%}")
        print(f"  Melhoria: {comparison['relative_improvement']:+.1%}")
        print(f"  Melhor? {comparison['better_than_baseline']}")
    
    # Score global
    score = comparator.calculate_score(metrics)
    print(f"\n{'='*70}")
    print(f"SCORE GLOBAL: {score:.3f}")
    print(f"{'='*70}")
    
    print("\n✓ MÉTRICAS IMPLEMENTADAS COM SUCESSO!")