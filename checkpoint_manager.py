# """
# Gerenciador de Checkpoints - Encontra e carrega automaticamente o checkpoint mais recente
# """
import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any
import torch


class CheckpointManager:
    #"""Gerencia checkpoints de treinamento, encontrando e carregando automaticamente o mais recente"""
    
    def __init__(self, base_dir: str = "scripts"):
        # """
        # Inicializa o gerenciador de checkpoints
        
        # Args:
        #     base_dir: Diretório base onde procurar checkpoints (padrão: 'scripts')
        # """
        self.base_dir = Path(base_dir)
        
    def find_latest_checkpoint(self, pattern: str = "*.pt") -> Optional[Path]:
        # """
        # Encontra o checkpoint mais recente no diretório base
        
        # Args:
        #     pattern: Padrão de arquivo para buscar (padrão: '*.pt')
            
        # Returns:
        #     Path do checkpoint mais recente ou None se não encontrado
        # """
        if not self.base_dir.exists():
            print(f"⚠️  Diretório {self.base_dir} não encontrado")
            return None
            
        # Procura todos os checkpoints
        checkpoints = list(self.base_dir.glob(pattern))
        
        if not checkpoints:
            print(f"⚠️  Nenhum checkpoint encontrado em {self.base_dir}")
            return None
            
        # Ordena por data de modificação (mais recente primeiro)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        latest = checkpoints[0]
        print(f"✓ Checkpoint mais recente encontrado: {latest.name}")
        print(f"  Modificado em: {self._format_timestamp(latest.stat().st_mtime)}")
        
        return latest
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None, 
                       device: str = 'cpu') -> Optional[Dict[str, Any]]:
        # """
        # Carrega um checkpoint
        
        # Args:
        #     checkpoint_path: Caminho específico do checkpoint (se None, usa o mais recente)
        #     device: Dispositivo para carregar ('cpu' ou 'cuda')
            
        # Returns:
        #     Dicionário com dados do checkpoint ou None se falhar
        # """
        # Se não especificado, busca o mais recente
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            
        if checkpoint_path is None:
            return None
            
        try:
            print(f"\n📥 Carregando checkpoint: {checkpoint_path.name}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Mostra informações do checkpoint
            self._print_checkpoint_info(checkpoint)
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ Erro ao carregar checkpoint: {e}")
            return None
    
    def list_all_checkpoints(self, pattern: str = "*.pt") -> list:
        # """
        # Lista todos os checkpoints disponíveis
        
        # Args:
        #     pattern: Padrão de arquivo para buscar
            
        # Returns:
        #     Lista de Paths dos checkpoints, ordenados do mais recente ao mais antigo
        # """
        if not self.base_dir.exists():
            print(f"⚠️  Diretório {self.base_dir} não encontrado")
            return []
            
        checkpoints = list(self.base_dir.glob(pattern))
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not checkpoints:
            print(f"⚠️  Nenhum checkpoint encontrado em {self.base_dir}")
            return []
        
        print(f"\n📋 Checkpoints disponíveis em {self.base_dir}:")
        print("=" * 80)
        for i, cp in enumerate(checkpoints, 1):
            size_mb = cp.stat().st_size / (1024 * 1024)
            timestamp = self._format_timestamp(cp.stat().st_mtime)
            print(f"{i}. {cp.name}")
            print(f"   Tamanho: {size_mb:.2f} MB | Modificado: {timestamp}")
            
        return checkpoints
    
    def _print_checkpoint_info(self, checkpoint: Dict[str, Any]):
        #"""Exibe informações sobre o checkpoint carregado"""
        print("\n" + "=" * 80)
        print("📊 INFORMAÇÕES DO CHECKPOINT")
        print("=" * 80)
        
        # Informações básicas
        if 'epoch' in checkpoint:
            print(f"Época: {checkpoint['epoch']}")
        if 'iteration' in checkpoint:
            print(f"Iteração: {checkpoint['iteration']}")
        if 'curriculum_stage' in checkpoint:
            print(f"Estágio do Currículo: {checkpoint['curriculum_stage']}")
            
        # Métricas de treinamento
        if 'avg_reward' in checkpoint:
            print(f"Recompensa Média: {checkpoint['avg_reward']:.4f}")
        if 'avg_utilization' in checkpoint:
            print(f"Utilização Média: {checkpoint['avg_utilization']:.2%}")
        if 'best_reward' in checkpoint:
            print(f"Melhor Recompensa: {checkpoint['best_reward']:.4f}")
            
        # Componentes do modelo
        components = []
        if 'actor_state_dict' in checkpoint:
            components.append('Actor')
        if 'critic_state_dict' in checkpoint:
            components.append('Critic')
        if 'optimizer_state_dict' in checkpoint:
            components.append('Optimizer')
        if 'scheduler_state_dict' in checkpoint:
            components.append('Scheduler')
            
        if components:
            print(f"Componentes: {', '.join(components)}")
            
        print("=" * 80)
    
    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        #"""Formata timestamp Unix para string legível"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


# Função de conveniência para uso rápido
def load_latest_checkpoint(base_dir: str = "scripts", device: str = 'cpu') -> Optional[Dict[str, Any]]:
    # """
    # Função de conveniência para carregar o checkpoint mais recente
    
    # Args:
    #     base_dir: Diretório onde procurar checkpoints
    #     device: Dispositivo para carregar ('cpu' ou 'cuda')
        
    # Returns:
    #     Dicionário com dados do checkpoint ou None
        
    # Example:
    #     >>> checkpoint = load_latest_checkpoint()
    #     >>> if checkpoint:
    #     ...     model.load_state_dict(checkpoint['actor_state_dict'])
    # """
    manager = CheckpointManager(base_dir)
    return manager.load_checkpoint(device=device)


if __name__ == "__main__":
    # Demonstração
    print("🔍 CHECKPOINT MANAGER - Demonstração\n")
    
    manager = CheckpointManager("scripts")
    
    # Lista todos os checkpoints
    all_checkpoints = manager.list_all_checkpoints()
    
    # Carrega o mais recente
    if all_checkpoints:
        checkpoint = manager.load_checkpoint()
        
        if checkpoint:
            print("\n✅ Checkpoint carregado com sucesso!")
            print(f"   Chaves disponíveis: {list(checkpoint.keys())}")
    else:
        print("\n📝 Nenhum checkpoint encontrado. Execute o treinamento primeiro!")
        print("   Exemplos de uso:")
        print("   1. python train_ppo.py")
        print("   2. python exemplo_simples.py")