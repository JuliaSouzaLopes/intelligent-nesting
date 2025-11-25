# 🎓 GUIA: Treinamento com Arquivos CAD Reais

## 🎯 Objetivo

Treinar o modelo de nesting usando **peças reais** vindas de arquivos CAD, ao invés de peças sintéticas geradas aleatoriamente.

**Benefícios:**
- ✅ Modelo aprende com casos reais de produção
- ✅ Melhor performance em problemas do seu domínio
- ✅ Adaptação às características específicas das suas peças
- ✅ Validação em benchmarks conhecidos

---

## 📋 Estratégia de Treinamento

### 1. Curriculum Learning Adaptativo

Ao invés de estágios fixos (Stage 1, 2, 3...), usamos **complexidade real** das peças:

```
Peças Simples (complexity: 0.2)
    ↓
Peças Médias (complexity: 0.5)
    ↓
Peças Complexas (complexity: 0.8)
    ↓
Peças Muito Complexas (complexity: 1.0)
```

**Complexidade calculada por:**
- Número de peças no conjunto
- Número de vértices por peça
- Irregularidade das formas

### 2. Datasets Suportados

#### A. Benchmarks da Literatura

Problemas padronizados usados em papers:

**RCO** (Retângulos):
- RCO1: 7 peças
- RCO2: 14 peças
- RCO3: 21 peças
- RCO4: 28 peças
- RCO5: 35 peças

**BLAZEWICZ** (Retângulos variados):
- BLAZEWICZ1-5: 7-35 peças

**SHAPES** (Formas irregulares):
- SHAPES2-15: 8-43 peças

#### B. Arquivos Customizados

Suas próprias peças de produção:
- DXF (AutoCAD)
- SVG (Inkscape, Illustrator)
- JSON (formato simples)

---

## 🚀 Como Usar

### Passo 1: Preparar Dataset

**Opção A: Usar Benchmarks**
```python
python train_with_real_cad.py
# Escolha opção 2: Criar problemas de benchmark
```

**Opção B: Usar Seus Arquivos CAD**

1. Crie a pasta:
```bash
mkdir -p datasets/cad_pieces
```

2. Adicione seus arquivos:
```
datasets/cad_pieces/
├── produto_A.dxf
├── produto_B.dxf
├── produto_C.svg
└── formas_customizadas.json
```

3. Execute:
```bash
python train_with_real_cad.py
```

**Opção C: Criar Dataset de Exemplo**
```python
python train_with_real_cad.py
# Escolha opção 1: Criar dataset de exemplo
```

---

### Passo 2: Configurar Treinamento

Edite no início de `train_with_real_cad.py`:

```python
# Dataset
DATASET_DIR = "datasets/cad_pieces"  # Sua pasta
DATASET_TYPE = "custom"              # ou "benchmark"

# Treinamento
CONFIG = {
    'n_iterations': 5000,      # Quantas iterações treinar
    'container_width': 1000,   # Tamanho da chapa (mm)
    'container_height': 600,
    'learning_rate': 3e-4,
    'device': 'cuda',          # 'cuda' ou 'cpu'
}
```

---

### Passo 3: Executar Treinamento

```bash
python train_with_real_cad.py
```

**Saída esperada:**
```
╔════════════════════════════════════════════════════════════════════════╗
║         🎯 TREINAMENTO COM DATASETS CAD REAIS 🎯                     ║
╚════════════════════════════════════════════════════════════════════════╝

📂 Carregando dataset de: datasets/cad_pieces
   Encontrados 5 arquivos
   ✓ produto_A.dxf: 12 peças (complexidade: 0.45)
   ✓ produto_B.dxf: 8 peças (complexidade: 0.32)
   ✓ produto_C.svg: 15 peças (complexidade: 0.67)
   ✓ formas_L.json: 6 peças (complexidade: 0.28)
   ✓ hexagonos.json: 20 peças (complexidade: 0.73)

✓ Total: 5 conjuntos de peças carregados

🎓 Curriculum Adaptativo Inicializado
   Total de conjuntos: 5
   Complexidade: 0.28 → 0.73

════════════════════════════════════════════════════════════════════════
INICIANDO TREINAMENTO
════════════════════════════════════════════════════════════════════════

Iteration 10/5000
  Dataset: formas_L (1/5)
  Complexidade: 0.28
  Peças: 6
  Utilização: 72.3%
  Taxa de sucesso: 80.0%
  Progresso: 20.0%

🎓 CURRICULUM AVANÇOU!
   formas_L (comp: 0.28)
   ↓
   produto_B (comp: 0.32)

...
```

---

### Passo 4: Monitorar Treinamento

```bash
tensorboard --logdir logs/real_cad_training
```

Abra: http://localhost:6006

**Métricas disponíveis:**
- `training/utilization` - Utilização da chapa
- `training/complexity` - Complexidade atual
- `training/success_rate` - Taxa de sucesso
- `curriculum/piece_set_idx` - Progresso no curriculum

---

## 📐 Formato dos Arquivos

### JSON (Recomendado para começar)

```json
{
  "pieces": [
    {
      "id": 0,
      "name": "Peça Principal",
      "vertices": [[0, 0], [150, 0], [150, 100], [0, 100]]
    },
    {
      "id": 1,
      "name": "Forma em L",
      "vertices": [
        [0, 0], [100, 0], [100, 50], 
        [50, 50], [50, 100], [0, 100]
      ]
    },
    {
      "id": 2,
      "name": "Hexágono",
      "vertices": [
        [50, 0], [100, 25], [100, 75],
        [50, 100], [0, 75], [0, 25]
      ]
    }
  ]
}
```

### DXF (AutoCAD)

Requisitos:
- Use `LWPOLYLINE` para desenhar peças
- Feche todos os polígonos
- Unidades em milímetros

### SVG

Requisitos:
- Use `<polygon>` ou `<polyline>` fechados
- Unidades em milímetros

---

## 🎯 Estratégia de Curriculum

O sistema avança automaticamente quando o modelo atinge **70% de taxa de sucesso** no conjunto atual:

```
1. Começa com conjunto mais simples
   └─ Treina até 70% sucesso
      └─ Avança para próximo conjunto
         └─ Treina até 70% sucesso
            └─ ...e assim por diante
```

**Threshold de sucesso:**
- Calculado baseado na complexidade
- Conjunto simples (0.2): threshold = 56%
- Conjunto médio (0.5): threshold = 65%
- Conjunto complexo (0.8): threshold = 74%

---

## 📊 Comparação: Sintético vs Real

### Treinamento Sintético (Original)

```python
# Peças geradas aleatoriamente
pieces = generate_random_polygons(n=10)
```

**Vantagens:**
- Fácil de começar
- Variedade infinita

**Desvantagens:**
- Pode não representar casos reais
- Modelo pode não generalizar bem

### Treinamento com CAD Real

```python
# Peças de arquivos CAD reais
pieces = load_from_dxf("produtos.dxf")
```

**Vantagens:**
- Aprende com casos reais ✅
- Melhor performance em produção ✅
- Validação em benchmarks ✅

**Desvantagens:**
- Requer preparação de dados
- Menos variabilidade

---

## 💡 Melhores Práticas

### 1. Organizar Dataset por Categoria

```
datasets/cad_pieces/
├── simples/
│   ├── retangulos.json
│   └── quadrados.json
├── medios/
│   ├── formas_L.dxf
│   └── trapezios.svg
└── complexos/
    ├── irregulares.dxf
    └── poligonos_complexos.json
```

### 2. Começar com Poucos Conjuntos

Teste com 3-5 conjuntos primeiro:
1. Simples (3-5 peças)
2. Médio (8-12 peças)
3. Complexo (15-20 peças)

Depois adicione mais conforme necessário.

### 3. Validar Carregamento

Antes de treinar, verifique se peças carregaram corretamente:

```python
from train_with_real_cad import CADDatasetLoader

loader = CADDatasetLoader("datasets/cad_pieces")
piece_sets = loader.load_all()

for ps in piece_sets:
    print(f"{ps.name}: {ps.n_pieces} peças, complexity: {ps.complexity:.2f}")
```

### 4. Usar Benchmarks para Validação

Sempre inclua alguns problemas de benchmark para comparar:

```python
loader.create_benchmark_dataset()
```

---

## 🔧 Troubleshooting

### Problema: "Nenhuma peça carregada de DXF"

**Causa:** DXF não tem LWPOLYLINE  
**Solução:** 
- No AutoCAD, use comando `LWPOLYLINE`
- Converta objetos existentes: `PEDIT` → `LWPOLYLINE`

### Problema: "Complexidade muito alta logo no início"

**Causa:** Peças muito complexas no dataset  
**Solução:**
- Adicione peças mais simples (retângulos)
- Sistema reordena automaticamente, mas ajuda ter variedade

### Problema: "Modelo não melhora"

**Causas possíveis:**
- Learning rate muito alto/baixo
- Dataset muito pequeno
- Peças muito difíceis

**Soluções:**
- Ajuste `learning_rate` em CONFIG
- Adicione mais variedade de peças
- Comece com casos mais simples

---

## 📈 Resultados Esperados

### Após 1000 iterations:
- Conjuntos simples: 65-75% utilização
- Conjuntos médios: início do aprendizado

### Após 3000 iterations:
- Conjuntos simples: 75-85% utilização
- Conjuntos médios: 65-75% utilização
- Conjuntos complexos: início do aprendizado

### Após 5000 iterations:
- Conjuntos simples: 80-90% utilização
- Conjuntos médios: 75-85% utilização
- Conjuntos complexos: 70-80% utilização

---

## 🎓 Próximos Passos

1. **Prepare seu dataset** (escolha uma opção acima)
2. **Execute treinamento inicial** (1000 iterations para teste)
3. **Valide resultados** com TensorBoard
4. **Ajuste configuração** se necessário
5. **Treinamento completo** (5000+ iterations)
6. **Use modelo treinado** com `run_nesting_cad.py`

---

## 📚 Referências

**Papers sobre benchmarks:**
- Toledo et al. (2013) - "MODELOS MATEMÁTICOS PARA O PROBLEMA DE CORTE DE PEÇAS IRREGULARES"
- Problemas RCO, BLAZEWICZ, SHAPES

**Curriculum Learning:**
- Bengio et al. (2009) - "Curriculum Learning"
- Sistema adapta automaticamente a dificuldade

---

## ✅ Checklist

- [ ] Instalei dependências (`pip install ezdxf svgpathtools`)
- [ ] Preparei dataset (benchmark, custom, ou exemplo)
- [ ] Configurei parâmetros em `train_with_real_cad.py`
- [ ] Executei treinamento de teste (100 iterations)
- [ ] Monitoro com TensorBoard
- [ ] Ajustei configuração baseado em resultados
- [ ] Rodei treinamento completo (5000 iterations)
- [ ] Validei modelo final em problemas de teste

---

**🎯 Boa sorte com o treinamento! Com dados reais, seu modelo será muito mais útil em produção!**