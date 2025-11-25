# 🎯 Sistema de Treinamento com Arquivos CAD Reais

## Visão Geral

Sistema completo para treinar o modelo de nesting usando **peças reais** de arquivos CAD, ao invés de peças sintéticas.

---

## 📦 Arquivos Criados

| Arquivo | Descrição |
|---------|-----------|
| [train_with_real_cad.py](train_with_real_cad.py) | ⭐ Sistema de treinamento completo |
| [GUIA_TREINAMENTO_CAD_REAL.md](GUIA_TREINAMENTO_CAD_REAL.md) | 📖 Documentação detalhada |
| [validate_cad_dataset.py](validate_cad_dataset.py) | 🔍 Validador de dataset |

---

## 🚀 Início Rápido (5 minutos)

### 1. Criar Dataset de Exemplo

```bash
python train_with_real_cad.py
# Escolha opção 1: Criar dataset de exemplo
```

### 2. Validar Dataset

```bash
python validate_cad_dataset.py
```

### 3. Treinar

```bash
python train_with_real_cad.py
```

---

## 📁 Estrutura de Diretórios

```
seu_projeto/
├── datasets/
│   └── cad_pieces/          # Seus arquivos CAD aqui
│       ├── simples.json
│       ├── produto_A.dxf
│       ├── produto_B.svg
│       └── benchmark/       # Problemas padrão (auto-criado)
│           ├── RCO1.json
│           ├── RCO2.json
│           └── ...
│
├── logs/
│   └── real_cad_training/   # TensorBoard logs
│
├── scripts/                 # Checkpoints salvos
│   ├── checkpoint_100.pt
│   ├── checkpoint_200.pt
│   └── ...
│
└── train_with_real_cad.py   # Script principal
```

---

## 🎓 Curriculum Learning Adaptativo

O sistema ordena automaticamente suas peças por **complexidade real**:

```
📁 Dataset Carregado:
   • simples_3pecas.json     → Complexity: 0.25
   • produto_B.dxf           → Complexity: 0.42
   • formas_L.svg            → Complexity: 0.58
   • irregulares.dxf         → Complexity: 0.73
   • complexo_50pecas.json   → Complexity: 0.91

🎓 Ordem de Treinamento:
   1. simples_3pecas         (mais fácil)
   2. produto_B
   3. formas_L
   4. irregulares
   5. complexo_50pecas       (mais difícil)
```

**Sistema avança automaticamente** quando atinge 70% de taxa de sucesso.

---

## 📐 Formatos Suportados

### DXF (AutoCAD)

```
Requisitos:
✓ Use LWPOLYLINE
✓ Feche todos os polígonos
✓ Unidades em milímetros
```

### SVG

```
Requisitos:
✓ Use <polygon> ou <polyline>
✓ Paths fechados
✓ Unidades em milímetros
```

### JSON (Mais Simples)

```json
{
  "pieces": [
    {
      "id": 0,
      "vertices": [[0,0], [100,0], [100,50], [0,50]]
    },
    {
      "id": 1,
      "vertices": [[0,0], [80,0], [80,40], [40,40], [40,80], [0,80]]
    }
  ]
}
```

---

## 🔄 Workflow Típico

```
┌──────────────────────────────────────────────────────────────┐
│ 1. PREPARAR DATASET                                          │
│    Opção A: Criar exemplo                                    │
│    Opção B: Usar benchmarks                                  │
│    Opção C: Adicionar seus DXF/SVG                           │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. VALIDAR                                                   │
│    python validate_cad_dataset.py                            │
│    → Ver estatísticas                                        │
│    → Verificar complexidade                                  │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. CONFIGURAR                                                │
│    Editar CONFIG em train_with_real_cad.py:                  │
│    - n_iterations                                            │
│    - container_width/height                                  │
│    - learning_rate                                           │
│    - device (cuda/cpu)                                       │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. TREINAR                                                   │
│    python train_with_real_cad.py                             │
│    → Sistema avança automaticamente por complexidade        │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. MONITORAR                                                 │
│    tensorboard --logdir logs/real_cad_training               │
│    → Utilização, complexidade, progresso                     │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. USAR MODELO                                               │
│    python run_nesting_cad.py                                 │
│    → Modelo treinado com suas peças reais!                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Métricas Monitoradas

Durante o treinamento, você verá:

```
Iteration 250/5000
  Dataset: produto_B (2/5)
  Complexidade: 0.42
  Peças: 12
  Utilização: 68.5%
  Taxa de sucesso: 75.0%
  Progresso: 40.0%
```

**TensorBoard:**
- `training/utilization` - % de aproveitamento da chapa
- `training/complexity` - Dificuldade atual
- `training/success_rate` - Taxa de acerto
- `curriculum/piece_set_idx` - Posição no curriculum

---

## 💡 Vantagens vs Treinamento Sintético

| Aspecto | Sintético | CAD Real |
|---------|-----------|----------|
| Setup | Rápido | Requer preparação |
| Variedade | Infinita | Limitada ao dataset |
| **Relevância** | Média | **Alta** ✅ |
| **Performance em produção** | Boa | **Excelente** ✅ |
| **Validação** | Difícil | **Benchmarks** ✅ |

---

## 🎯 Casos de Uso

### 1. Indústria Metalúrgica

```bash
datasets/cad_pieces/
├── chapas_aco/
│   ├── braquetes.dxf
│   ├── suportes.dxf
│   └── conectores.dxf
```

### 2. Indústria Têxtil

```bash
datasets/cad_pieces/
├── moldes_roupa/
│   ├── camiseta_P.svg
│   ├── camiseta_M.svg
│   ├── calca_base.svg
│   └── manga_curta.svg
```

### 3. Móveis

```bash
datasets/cad_pieces/
├── pecas_moveis/
│   ├── tampo_mesa.dxf
│   ├── lateral_armario.dxf
│   └── porta_gaveta.dxf
```

### 4. Research / Benchmarks

```bash
# Usa problemas da literatura
python train_with_real_cad.py
# Opção 2: Criar benchmarks
# → RCO, BLAZEWICZ, SHAPES
```

---

## ⚙️ Configurações Importantes

```python
# Em train_with_real_cad.py

CONFIG = {
    # Para teste rápido (30 min):
    'n_iterations': 500,
    
    # Para treinamento médio (4-6h):
    'n_iterations': 2000,
    
    # Para treinamento completo (10-15h):
    'n_iterations': 5000,
    
    # GPU pequena (2GB):
    'batch_size': 32,
    'device': 'cuda',
    
    # GPU grande (8GB+):
    'batch_size': 128,
    'device': 'cuda',
    
    # Sem GPU:
    'device': 'cpu',
    'batch_size': 16,
}
```

---

## 📈 Resultados Esperados

### Após 500 iterations (teste):
- Conjuntos simples: melhora visível
- Avança 1-2 estágios no curriculum

### Após 2000 iterations:
- Conjuntos simples: 75-85% utilização
- Conjuntos médios: 65-75% utilização

### Após 5000 iterations:
- Conjuntos simples: 80-90% utilização
- Conjuntos médios: 75-85% utilização
- Conjuntos complexos: 70-80% utilização

---

## 🆘 Troubleshooting

### "Nenhum arquivo carregado"

**Causa:** Pasta vazia ou formato incorreto  
**Solução:**
```bash
python train_with_real_cad.py
# Escolha opção 1 ou 2 para criar dataset
```

### "DXF sem peças"

**Causa:** DXF não tem LWPOLYLINE  
**Solução:** No AutoCAD, use `LWPOLYLINE` ou converta com `PEDIT`

### "CUDA out of memory"

**Causa:** GPU pequena  
**Solução:**
```python
CONFIG = {
    'batch_size': 32,  # Reduzir
    'device': 'cpu',   # Ou usar CPU
}
```

---

## 📚 Documentação Completa

- [GUIA_TREINAMENTO_CAD_REAL.md](GUIA_TREINAMENTO_CAD_REAL.md) - Guia detalhado
- [train_with_real_cad.py](train_with_real_cad.py) - Código comentado
- [validate_cad_dataset.py](validate_cad_dataset.py) - Validação

---

## ✅ Checklist de Setup

- [ ] Instalei ezdxf (`pip install ezdxf`)
- [ ] Instalei svgpathtools (`pip install svgpathtools`)
- [ ] Criei dataset (exemplo, benchmark, ou custom)
- [ ] Validei dataset (`python validate_cad_dataset.py`)
- [ ] Configurei parâmetros em `train_with_real_cad.py`
- [ ] Rodei teste curto (500 iterations)
- [ ] Monitorei no TensorBoard
- [ ] Ajustei configuração
- [ ] Rodando treinamento completo ✅

---

**🎓 Com este sistema, seu modelo aprende com casos REAIS e terá muito melhor performance em produção!**

**Próximo passo:** `python train_with_real_cad.py` 🚀