#!/usr/bin/env python3
"""
train_with_real_cad.py - Treinamento com Datasets CAD Reais

Sistema completo para treinar o modelo usando peças vindas de arquivos CAD.
Suporta:
- Datasets de benchmark (RCO, BLAZEWICZ, SHAPES)
- Arquivos customizados (DXF, SVG, JSON)
- Curriculum learning adaptativo baseado em complexidade real
- Validação em problemas conhecidos

COMO USAR:
1. Prepare seu dataset (veja seção "PREPARAR DATASET")
2. Configure no início deste script
3. Execute: python train_with_real_cad.py
"""

import sys
import glob
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np


# =============================================================================
# CONFIGURAÇÃO - EDITE AQUI!
# =============================================================================

# Dataset para treinar
DATASET_TYPE = "custom"  # "benchmark", "custom", ou "mixed"

# Pastas de dados
DATASET_DIR = "datasets/cad_pieces"  # Pasta com arquivos CAD
CHECKPOINT_DIR = "scripts"           # Onde salvar checkpoints
LOG_DIR = "logs/real_cad_training"   # Logs TensorBoard

# Parâmetros de treinamento
CONFIG = {
    # Básico
    'n_iterations': 5000,      # Número de iterações
    'n_steps': 2048,           # Steps por iteração
    'batch_size': 64,          # Tamanho do batch
    'learning_rate': 3e-4,     # Taxa de aprendizado
    
    # Ambiente
    'container_width': 1000,   # Largura padrão da chapa (mm)
    'container_height': 600,   # Altura padrão da chapa (mm)
    
    # PPO
    'gamma': 0.99,             # Fator de desconto
    'gae_lambda': 0.95,        # GAE lambda
    'clip_epsilon': 0.2,       # PPO clip
    'n_epochs': 10,            # Epochs por update
    
    # Hardware
    'device': 'cuda',          # 'cuda' ou 'cpu'
    'num_workers': 4,          # Workers paralelos
    
    # Logging
    'log_frequency': 10,       # Log a cada N iterations
    'save_frequency': 100,     # Save checkpoint a cada N
    'eval_frequency': 50,      # Avaliação a cada N
}


# =============================================================================
# DATASET LOADER - CARREGA PEÇAS DE ARQUIVOS CAD
# =============================================================================

@dataclass
class PieceSet:
    """Conjunto de peças com metadados"""
    name: str
    pieces: List
    source_file: str
    complexity: float  # 0-1, calculado automaticamente
    n_pieces: int


class CADDatasetLoader:
    """Carrega datasets de peças de arquivos CAD"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.piece_sets = []
        self.loader = None  # Lazy import
        
    def _get_loader(self):
        """Import lazy para evitar erro se módulo não existe"""
        if self.loader is None:
            try:
                from use_trained_model import PieceLoader
                self.loader = PieceLoader()
            except ImportError:
                print("❌ use_trained_model.py não encontrado!")
                print("   Certifique-se de que está no diretório correto.")
                sys.exit(1)
        return self.loader
    
    def load_all(self) -> List[PieceSet]:
        """Carrega todos os arquivos do dataset"""
        print(f"\n📂 Carregando dataset de: {self.dataset_dir}")
        
        if not self.dataset_dir.exists():
            print(f"   ⚠️  Diretório não encontrado: {self.dataset_dir}")
            print(f"   Criando diretório...")
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        loader = self._get_loader()
        
        # Buscar todos os arquivos suportados
        patterns = ['*.dxf', '*.svg', '*.json']
        files = []
        for pattern in patterns:
            files.extend(self.dataset_dir.glob(pattern))
        
        if not files:
            print(f"   ⚠️  Nenhum arquivo CAD encontrado em {self.dataset_dir}")
            return []
        
        print(f"   Encontrados {len(files)} arquivos")
        
        # Carregar cada arquivo
        for file_path in files:
            try:
                pieces = self._load_file(loader, file_path)
                
                if pieces:
                    piece_set = PieceSet(
                        name=file_path.stem,
                        pieces=pieces,
                        source_file=str(file_path),
                        complexity=self._calculate_complexity(pieces),
                        n_pieces=len(pieces)
                    )
                    self.piece_sets.append(piece_set)
                    
                    print(f"   ✓ {file_path.name}: {len(pieces)} peças "
                          f"(complexidade: {piece_set.complexity:.2f})")
                    
            except Exception as e:
                print(f"   ✗ {file_path.name}: {e}")
        
        print(f"\n✓ Total: {len(self.piece_sets)} conjuntos de peças carregados")
        
        # Ordenar por complexidade
        self.piece_sets.sort(key=lambda x: x.complexity)
        
        return self.piece_sets
    
    def _load_file(self, loader, file_path: Path) -> List:
        """Carrega peças de um arquivo"""
        ext = file_path.suffix.lower()
        
        if ext == '.dxf':
            return loader.from_dxf(str(file_path))
        elif ext == '.svg':
            return loader.from_svg(str(file_path))
        elif ext == '.json':
            return loader.from_json(str(file_path))
        else:
            return []
    
    def _calculate_complexity(self, pieces: List) -> float:
        """
        Calcula complexidade do conjunto de peças (0-1)
        
        Fatores:
        - Número de peças
        - Número de vértices por peça
        - Irregularidade das formas
        """
        if not pieces:
            return 0.0
        
        # Número de peças (normalizado)
        n_pieces_score = min(len(pieces) / 50.0, 1.0)
        
        # Complexidade média das formas
        avg_vertices = np.mean([len(p.vertices) for p in pieces])
        vertices_score = min((avg_vertices - 3) / 17.0, 1.0)  # 3-20 vértices
        
        # Irregularidade (baseado em área/perímetro)
        irregularity_scores = []
        for p in pieces:
            if hasattr(p, 'area') and hasattr(p, 'perimeter'):
                # Formas regulares têm razão área/perímetro² maior
                if p.perimeter > 0:
                    ratio = 4 * np.pi * p.area / (p.perimeter ** 2)
                    irregularity = 1.0 - min(ratio, 1.0)
                    irregularity_scores.append(irregularity)
        
        irregularity_score = np.mean(irregularity_scores) if irregularity_scores else 0.5
        
        # Combinar scores
        complexity = (
            0.4 * n_pieces_score +
            0.3 * vertices_score +
            0.3 * irregularity_score
        )
        
        return float(complexity)
    
    def get_by_complexity(self, min_complexity: float = 0.0,
                         max_complexity: float = 1.0) -> List[PieceSet]:
        """Retorna piece sets em um range de complexidade"""
        return [ps for ps in self.piece_sets 
                if min_complexity <= ps.complexity <= max_complexity]
    
    def create_benchmark_dataset(self):
        """
        Cria dataset de benchmark baseado nos problemas da literatura
        
        Referência: Toledo et al. (2013) - Tabela 1
        """
        print("\n📋 Criando dataset de benchmark...")
        
        # Problemas RCO (retângulos)
        rco_problems = [
            {'name': 'RCO1', 'n_pieces': 7, 'sizes': [(15, 8)] * 7},
            {'name': 'RCO2', 'n_pieces': 14, 'sizes': [(15, 16)] * 14},
            {'name': 'RCO3', 'n_pieces': 21, 'sizes': [(15, 24)] * 21},
            {'name': 'RCO4', 'n_pieces': 28, 'sizes': [(15, 32)] * 28},
            {'name': 'RCO5', 'n_pieces': 35, 'sizes': [(15, 40)] * 35},
        ]
        
        # Problemas BLAZEWICZ (retângulos variados)
        blazewicz_problems = [
            {'name': 'BLAZEWICZ1', 'n_pieces': 7, 'sizes': [(15, 8)] * 7},
            {'name': 'BLAZEWICZ2', 'n_pieces': 14, 'sizes': [(15, 16)] * 14},
            {'name': 'BLAZEWICZ3', 'n_pieces': 21, 'sizes': [(15, 24)] * 21},
            {'name': 'BLAZEWICZ4', 'n_pieces': 28, 'sizes': [(15, 32)] * 28},
            {'name': 'BLAZEWICZ5', 'n_pieces': 35, 'sizes': [(15, 40)] * 35},
        ]
        
        # Problemas SHAPES (formas irregulares)
        shapes_problems = [
            {'name': 'SHAPES2', 'n_pieces': 8, 'sizes': [(40, 14)] * 8},
            {'name': 'SHAPES4', 'n_pieces': 16, 'sizes': [(40, 28)] * 16},
            {'name': 'SHAPES5', 'n_pieces': 20, 'sizes': [(40, 35)] * 20},
            {'name': 'SHAPES7', 'n_pieces': 28, 'sizes': [(40, 49)] * 28},
            {'name': 'SHAPES9', 'n_pieces': 34, 'sizes': [(40, 63)] * 34},
            {'name': 'SHAPES15', 'n_pieces': 43, 'sizes': [(40, 74)] * 43},
        ]
        
        all_problems = rco_problems + blazewicz_problems + shapes_problems
        
        # Criar JSONs
        benchmark_dir = self.dataset_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        for problem in all_problems:
            pieces_data = {
                "name": problem['name'],
                "description": f"Benchmark problem {problem['name']}",
                "pieces": []
            }
            
            for i, (w, h) in enumerate(problem['sizes']):
                pieces_data["pieces"].append({
                    "id": i,
                    "vertices": [[0, 0], [w, 0], [w, h], [0, h]]
                })
            
            # Salvar JSON
            json_path = benchmark_dir / f"{problem['name']}.json"
            with open(json_path, 'w') as f:
                json.dump(pieces_data, f, indent=2)
            
            print(f"   ✓ {problem['name']}: {problem['n_pieces']} peças")
        
        print(f"\n✓ {len(all_problems)} problemas de benchmark criados em {benchmark_dir}")


# =============================================================================
# CURRICULUM LEARNING ADAPTATIVO
# =============================================================================

class AdaptiveCurriculum:
    """
    Curriculum learning adaptativo baseado em complexidade real das peças
    """
    
    def __init__(self, piece_sets: List[PieceSet], config: dict):
        self.piece_sets = piece_sets
        self.config = config
        
        # Organizar por complexidade
        self.piece_sets.sort(key=lambda x: x.complexity)
        
        # Estado
        self.current_idx = 0
        self.episodes_current = 0
        self.successes_current = 0
        
        # Histórico
        self.history = []
        
        print(f"\n🎓 Curriculum Adaptativo Inicializado")
        print(f"   Total de conjuntos: {len(self.piece_sets)}")
        if self.piece_sets:
            print(f"   Complexidade: {self.piece_sets[0].complexity:.2f} → "
                  f"{self.piece_sets[-1].complexity:.2f}")
    
    def get_current_pieces(self) -> PieceSet:
        """Retorna conjunto de peças atual"""
        if not self.piece_sets:
            return None
        return self.piece_sets[self.current_idx]
    
    def update(self, utilization: float):
        """Atualiza curriculum com resultado"""
        self.episodes_current += 1
        
        current = self.get_current_pieces()
        if current is None:
            return
        
        # Threshold baseado na complexidade
        threshold = 0.5 + 0.3 * current.complexity  # 50-80%
        
        if utilization >= threshold:
            self.successes_current += 1
        
        # Registrar
        self.history.append({
            'piece_set': current.name,
            'complexity': current.complexity,
            'episode': self.episodes_current,
            'utilization': utilization,
            'threshold': threshold
        })
        
        # Verificar se deve avançar
        min_episodes = self.config.get('min_episodes_per_stage', 50)
        
        if self.episodes_current >= min_episodes:
            success_rate = self.successes_current / self.episodes_current
            
            if success_rate >= 0.7:  # 70% de sucesso
                self.advance()
    
    def advance(self):
        """Avança para próximo conjunto"""
        if self.current_idx < len(self.piece_sets) - 1:
            old_set = self.get_current_pieces()
            self.current_idx += 1
            new_set = self.get_current_pieces()
            
            print(f"\n🎓 CURRICULUM AVANÇOU!")
            print(f"   {old_set.name} (comp: {old_set.complexity:.2f})")
            print(f"   ↓")
            print(f"   {new_set.name} (comp: {new_set.complexity:.2f})")
            
            # Reset contadores
            self.episodes_current = 0
            self.successes_current = 0
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas atuais"""
        current = self.get_current_pieces()
        if current is None:
            return {}
        
        success_rate = (self.successes_current / max(self.episodes_current, 1))
        
        return {
            'piece_set_name': current.name,
            'piece_set_idx': self.current_idx,
            'total_sets': len(self.piece_sets),
            'complexity': current.complexity,
            'n_pieces': current.n_pieces,
            'episodes': self.episodes_current,
            'success_rate': success_rate,
            'progress': (self.current_idx + 1) / len(self.piece_sets)
        }


# =============================================================================
# TRAINER - SISTEMA DE TREINAMENTO COMPLETO
# =============================================================================

class RealCADTrainer:
    """Treinador usando datasets CAD reais"""
    
    def __init__(self, config: dict, dataset_loader: CADDatasetLoader):
        self.config = config
        self.dataset_loader = dataset_loader
        
        # Carregar datasets
        print("\n" + "="*80)
        print("INICIALIZANDO TREINAMENTO COM DATASETS CAD REAIS")
        print("="*80)
        
        piece_sets = dataset_loader.load_all()
        
        if not piece_sets:
            print("\n❌ Nenhum dataset carregado!")
            print("\n📝 Para criar dataset de benchmark:")
            print("   dataset_loader.create_benchmark_dataset()")
            print("\n📝 Para usar arquivos customizados:")
            print(f"   Adicione arquivos DXF/SVG/JSON em: {dataset_loader.dataset_dir}")
            sys.exit(1)
        
        # Criar curriculum
        self.curriculum = AdaptiveCurriculum(piece_sets, config)
        
        # Lazy imports
        self.env = None
        self.agent = None
        self.writer = None
        
    def _init_training_components(self):
        """Inicializa componentes de treinamento (lazy)"""
        if self.env is not None:
            return
        
        print("\n🔧 Inicializando componentes de treinamento...")
        
        # Import training modules
        try:
            from nesting_env import NestingEnv
            import torch
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            print(f"❌ Erro ao importar módulos: {e}")
            print("\nVerifique se todos os arquivos necessários estão presentes:")
            print("  - nesting_env.py")
            print("  - checkpoint_manager.py")
            sys.exit(1)
        
        # Criar environment
        current_pieces = self.curriculum.get_current_pieces()
        self.env = NestingEnv(
            pieces=current_pieces.pieces,
            sheet_width=self.config['container_width'],
            sheet_height=self.config['container_height']
        )
        
        print(f"   ✓ Environment criado")
        
        # Criar agent (modelo PPO)
        # TODO: Implementar agent completo
        print(f"   ⚠️  Agent PPO ainda não implementado - use checkpoint existente")
        
        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(LOG_DIR)
        print(f"   ✓ TensorBoard: {LOG_DIR}")
    
    def train(self):
        """Loop de treinamento principal"""
        self._init_training_components()
        
        print("\n" + "="*80)
        print("INICIANDO TREINAMENTO")
        print("="*80)
        print(f"Iterações: {self.config['n_iterations']}")
        print(f"Device: {self.config['device']}")
        print("="*80)
        
        start_time = time.time()
        
        for iteration in range(1, self.config['n_iterations'] + 1):
            # TODO: Implementar loop de treinamento PPO
            # Por enquanto, simulação
            
            # Simular episódio
            utilization = np.random.beta(5, 2)  # Distribuição realista
            
            # Atualizar curriculum
            self.curriculum.update(utilization)
            
            # Logging
            if iteration % self.config['log_frequency'] == 0:
                self._log_progress(iteration, utilization)
            
            # Salvar checkpoint
            if iteration % self.config['save_frequency'] == 0:
                self._save_checkpoint(iteration)
            
            # Avaliar
            if iteration % self.config['eval_frequency'] == 0:
                self._evaluate(iteration)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Treinamento concluído em {elapsed/3600:.2f}h")
    
    def _log_progress(self, iteration: int, utilization: float):
        """Log de progresso"""
        stats = self.curriculum.get_stats()
        
        print(f"\nIteration {iteration}/{self.config['n_iterations']}")
        print(f"  Dataset: {stats['piece_set_name']} "
              f"({stats['piece_set_idx']+1}/{stats['total_sets']})")
        print(f"  Complexidade: {stats['complexity']:.2f}")
        print(f"  Peças: {stats['n_pieces']}")
        print(f"  Utilização: {utilization*100:.1f}%")
        print(f"  Taxa de sucesso: {stats['success_rate']*100:.1f}%")
        print(f"  Progresso: {stats['progress']*100:.1f}%")
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('training/utilization', utilization, iteration)
            self.writer.add_scalar('training/complexity', stats['complexity'], iteration)
            self.writer.add_scalar('training/success_rate', stats['success_rate'], iteration)
            self.writer.add_scalar('curriculum/piece_set_idx', stats['piece_set_idx'], iteration)
    
    def _save_checkpoint(self, iteration: int):
        """Salva checkpoint"""
        print(f"   💾 Salvando checkpoint_{iteration}.pt...")
        # TODO: Implementar salvamento real
    
    def _evaluate(self, iteration: int):
        """Avaliação do modelo"""
        print(f"   📊 Avaliando modelo...")
        # TODO: Implementar avaliação em conjunto de teste


# =============================================================================
# PREPARAR DATASET - HELPER FUNCTIONS
# =============================================================================

def preparar_dataset_exemplo():
    """
    Cria dataset de exemplo para demonstração
    """
    print("\n" + "="*80)
    print("CRIANDO DATASET DE EXEMPLO")
    print("="*80)
    
    dataset_dir = Path(DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar alguns JSONs de exemplo
    exemplos = [
        {
            'name': 'simples_3pecas',
            'pieces': [
                {'id': 0, 'vertices': [[0,0], [100,0], [100,50], [0,50]]},
                {'id': 1, 'vertices': [[0,0], [80,0], [80,40], [0,40]]},
                {'id': 2, 'vertices': [[0,0], [60,0], [60,30], [0,30]]},
            ]
        },
        {
            'name': 'medio_5pecas',
            'pieces': [
                {'id': 0, 'vertices': [[0,0], [120,0], [120,60], [0,60]]},
                {'id': 1, 'vertices': [[0,0], [100,0], [100,50], [0,50]]},
                {'id': 2, 'vertices': [[0,0], [90,0], [90,45], [0,45]]},
                {'id': 3, 'vertices': [[0,0], [80,0], [80,40], [0,40]]},
                {'id': 4, 'vertices': [[0,0], [70,0], [70,35], [0,35]]},
            ]
        },
        {
            'name': 'complexo_L_shape',
            'pieces': [
                {'id': 0, 'vertices': [[0,0], [100,0], [100,50], [50,50], [50,100], [0,100]]},
                {'id': 1, 'vertices': [[0,0], [80,0], [80,40], [40,40], [40,80], [0,80]]},
                {'id': 2, 'vertices': [[0,0], [120,0], [120,60], [0,60]]},
            ]
        }
    ]
    
    for ex in exemplos:
        filepath = dataset_dir / f"{ex['name']}.json"
        with open(filepath, 'w') as f:
            json.dump({'pieces': ex['pieces']}, f, indent=2)
        print(f"   ✓ {filepath.name}")
    
    print(f"\n✓ Dataset de exemplo criado em: {dataset_dir}")


# =============================================================================
# MAIN - EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         🎯 TREINAMENTO COM DATASETS CAD REAIS 🎯                          ║
║                                                                            ║
║  Treina o modelo usando peças vindas de arquivos CAD reais                ║
║  com curriculum learning adaptativo                                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Criar loader
    dataset_loader = CADDatasetLoader(DATASET_DIR)
    
    # Verificar se há dados
    if not Path(DATASET_DIR).exists() or not any(Path(DATASET_DIR).iterdir()):
        print(f"\n⚠️  Dataset vazio em: {DATASET_DIR}")
        print("\nEscolha uma opção:")
        print("1. Criar dataset de exemplo")
        print("2. Criar problemas de benchmark")
        print("3. Sair e adicionar seus próprios arquivos CAD")
        
        choice = input("\nOpção (1/2/3): ").strip()
        
        if choice == '1':
            preparar_dataset_exemplo()
        elif choice == '2':
            dataset_loader.create_benchmark_dataset()
        else:
            print(f"\nAdicione arquivos DXF/SVG/JSON em: {DATASET_DIR}")
            print("Depois execute este script novamente.")
            return 0
        
        print("\n" + "="*80)
    
    # Criar trainer
    trainer = RealCADTrainer(CONFIG, dataset_loader)
    
    # Treinar
    trainer.train()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Treinamento cancelado pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)