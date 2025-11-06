# intelligent-nesting
# Guia de Instalação

## 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/intelligent-nesting.git
cd intelligent-nesting
```

## 2. Criar Ambiente Virtual
```bash
# Usando venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate  # Windows

# Usando conda
conda create -n nesting python=3.10
conda activate nesting
```

## 3. Instalar Dependências
```bash
# Instalação básica
pip install -r requirements.txt

# Instalação em modo desenvolvimento
pip install -e .

# Ou instalar tudo de uma vez
pip install -e ".[dev,viz]"
```

## 4. Instalar PyTorch Geometric
```bash
# Para CUDA 11.8
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Para CPU apenas
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## 5. Verificar Instalação
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

## 6. Download de Datasets (opcional)
```bash
python scripts/download_datasets.py
```

## 7. Executar Testes
```bash
pytest tests/ -v
```

## 8. Treinar Modelo
```bash
# CNN pretraining
python experiments/01_cnn_pretraining.py

# RL training
python experiments/04_rl_training.py --config config/default.yaml
```

## Troubleshooting

### Erro: CUDA out of memory
Solução: Reduzir batch_size no config

### Erro: Shapely não instala
Solução: 
```bash
# Linux
sudo apt-get install libgeos-dev

# Mac
brew install geos
```

### Erro: PyTorch Geometric
Solução: Verificar compatibilidade CUDA/PyTorch e reinstalar

print("=" * 70)
print("ESTRUTURA DO PROJETO CRIADA COM SUCESSO!")
print("=" * 70)
print("\nPróximos Passos:")
print("1. Criar os diretórios conforme estrutura acima")
print("2. Copiar requirements.txt, setup.py e configs")
print("3. Instalar dependências: pip install -r requirements.txt")
print("4. Começar implementação dos módulos")
print("\nVamos agora implementar cada módulo detalhadamente!")