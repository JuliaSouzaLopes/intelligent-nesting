# 🚀 Guia de Início Rápido - 5 Minutos

Sistema Inteligente de Nesting 2D com Deep RL + CNN

---

## ⚡ Setup em 3 Comandos

```bash
pip install -r requirements.txt
pip install -e .
python scripts/quick_test.py
```

**Resultado esperado:** ✅ Todos os testes passam

---

## 🎯 Treinar Imediatamente

### Opção 1: Treinamento Rápido (teste)

```bash
python scripts/train_complete_system.py --iterations 100
```

- Tempo: ~10 minutos
- Apenas para testar o sistema
- Não espere bons resultados

### Opção 2: Treinamento Real

```bash
python scripts/train_complete_system.py --iterations 5000 --device cuda
```

- Tempo: ~10 horas (GPU)
- Produz modelo útil
- Utilização esperada: 75-85%

### Opção 3: Treinamento Completo

```bash
python scripts/train_complete_system.py --iterations 10000 --device cuda
```

- Tempo: ~20 horas (GPU)
- Melhor modelo possível
- Utilização esperada: 80-90%

---

## 📊 Monitorar Treinamento

**Terminal 1 (treinamento):**
```bash
python scripts/train_complete_system.py --iterations 5000
```

**Terminal 2 (tensorboard):**
```bash
tensorboard --logdir logs/ppo_nesting
```

**Abrir navegador:** http://localhost:6006

### O que observar:

1. **`train/total_loss`**: Deve diminuir (convergência)
2. **`collection/avg_utilization`**: Deve aumentar
3. **`eval/utilization_mean`**: Métrica principal (target: 80%+)
4. **`curriculum/current_stage`**: Deve aumentar gradualmente

---

## 🎓 O que é Curriculum Learning?

O sistema treina em **8 estágios de dificuldade crescente**:

```
Stage 1: 3-5 retângulos simples        [60% threshold]
   ↓
Stage 2: + rotação                     [65% threshold]
   ↓
Stage 3: + mais peças                  [70% threshold]
   ↓
...
   ↓
Stage 8: 30-50 peças irregulares       [80% threshold]
```

**Sistema avança automaticamente** quando performance > threshold!

---

## 💾 Checkpoints

O sistema salva automaticamente:

### Checkpoints Regulares
```
checkpoints/checkpoint_00100.pt
checkpoints/checkpoint_00200.pt
...
```

### Melhor Modelo
```
checkpoints/best_model.pt  ← Use este!
```

---

## 🔄 Retomar Treinamento

```bash
python scripts/train_complete_system.py \
    --resume checkpoints/checkpoint_01000.pt \
    --iterations 15000
```

---

## 📈 Resultados Esperados

| Iterations | Tempo (GPU) | Utilização | Stage |
|------------|-------------|------------|-------|
| 100        | 10 min      | ~40%       | 1-2   |
| 1,000      | 2 horas     | ~65%       | 3-4   |
| 5,000      | 10 horas    | ~80%       | 6-7   |
| 10,000     | 20 horas    | ~85%       | 7-8   |

---

## 🐛 Problemas Comuns

### 1. Import Error

```bash
# Solução:
pip install -e .
```

### 2. CUDA Out of Memory

Edite `scripts/train_complete_system.py`:
```python
config = {
    'batch_size': 32,  # reduzir de 64
    # ...
}
```

### 3. Treinamento Lento

```python
# Verificar:
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## 📊 Estrutura de Dados

### Observação (Estado)

```python
observation = {
    'layout_image': np.array(6, 256, 256),     # Imagem 6-channel
    'current_piece': np.array(10,),            # Features da peça atual
    'remaining_pieces': np.array(10,),         # Features agregadas
    'stats': np.array(5,)                      # Stats globais
}
```

### Ação

```python
action = {
    'position': np.array([x, y]),  # [0, 1] normalizado
    'rotation': int                # 0-35 (bins de 10 graus)
}
```

### Recompensa

```python
reward = (
    +1.0    # placement válido
    +0.5    # bônus se toca outras peças
    +0.3    # bônus se próximo ao canto
    +0.1    # progresso
    -0.01   # penalidade de tempo
    -5.0    # placement inválido (colisão)
    +100×U  # bônus final (U = utilização)
)
```

---

## 🎯 Próximos Passos

1. ✅ **Executar quick_test.py**
2. ✅ **Treinar com 100 iterations (teste)**
3. ✅ **Verificar TensorBoard**
4. ✅ **Treinar com 5000 iterations (real)**
5. ✅ **Avaliar modelo**
6. ✅ **Exportar resultados**

---

## 📚 Documentação Completa

Ver `README_COMPLETE.md` para:
- Arquitetura detalhada
- Todos os parâmetros de configuração
- Guia de desenvolvimento
- Troubleshooting completo
- Referências e papers

---

## 🆘 Precisa de Ajuda?

1. **Quick test falhou?**
   - Verifique instalação: `pip list | grep torch`
   - Rode novamente: `python scripts/quick_test.py`

2. **Treinamento não inicia?**
   - Verifique GPU: `nvidia-smi`
   - Use CPU: `--device cpu`

3. **Resultados ruins?**
   - Treine mais: `--iterations 10000`
   - Ajuste learning rate
   - Verifique curriculum advancement

---

## ✨ Comandos Úteis

```bash
# Teste completo
python scripts/quick_test.py

# Treino rápido
python scripts/train_complete_system.py --iterations 100

# Treino real
python scripts/train_complete_system.py --iterations 5000 --device cuda

# Retomar
python scripts/train_complete_system.py --resume checkpoints/best_model.pt --iterations 10000

# TensorBoard
tensorboard --logdir logs/ppo_nesting

# Listar checkpoints
ls -lh checkpoints/
```

---

## 🎉 Pronto!

Você agora tem um sistema completo de nesting com:

- ✅ Deep RL (PPO)
- ✅ CNN para processar layouts
- ✅ Curriculum learning
- ✅ Auto-save de checkpoints
- ✅ TensorBoard monitoring
- ✅ GPU acceleration

**Bom treinamento! 🚀**

---

**Tempo estimado até modelo funcional:** 10 horas (GPU RTX 3090)

**Utilização esperada:** 80-85% em problemas com 20-30 peças