# Capítulo 08: Projeções Lineares

## 🎯 Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender transformações lineares e seu papel em redes neurais
2. Implementar projeções lineares manualmente com matmul
3. Entender `nn.Linear` e quando usá-lo
4. Debugar shapes em projeções lineares
5. Entender como pesos e bias funcionam

---

## 💡 Intuição

Uma **projeção linear** é simplesmente uma multiplicação matricial + bias.

$$\mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}$$

Onde:
- $\mathbf{x}$ = entrada [batch, in_features]
- $\mathbf{W}$ = pesos [in_features, out_features]
- $\mathbf{b}$ = bias [out_features]
- $\mathbf{y}$ = saída [batch, out_features]

**Por que "projeção"?** Estamos projetando dados de um espaço de dimensão `in_features` para `out_features`. É uma transformação linear.

**Em redes neurais**: A maioria das operações interessantes (embeddings, attention, MLPs) é composta de projeções lineares + ativações não-lineares.

---

## 📐 Matemática

### Simples: Vetor à Vetor

Entrada: $\mathbf{x} \in \mathbb{R}^d$
Pesos: $\mathbf{W} \in \mathbb{R}^{d \times d'}$
Bias: $\mathbf{b} \in \mathbb{R}^{d'}$

Saída: $\mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b} \in \mathbb{R}^{d'}$

Cada elemento de $\mathbf{y}$:

$$y_j = \sum_{i=1}^d x_i W_{ij} + b_j$$

### Batch: Múltiplos Vetores

Entrada: $\mathbf{X} \in \mathbb{R}^{B \times d}$ (batch de B vetores)
Pesos: $\mathbf{W} \in \mathbb{R}^{d \times d'}$
Bias: $\mathbf{b} \in \mathbb{R}^{d'}$ (broadcasting)

Saída: $\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b} \in \mathbb{R}^{B \times d'}$

Cada linha de Y é computada igual ao caso simples.

### Gradientes

Se $L$ é a loss, então:

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{y}}$$

$$\frac{\partial L}{\partial \mathbf{b}} = \sum_{i} \frac{\partial L}{\partial y_i}$$

(PyTorch computa tudo automaticamente via backprop)

---

## 🔨 Implementação Manual

```python
import torch

class LinearManual:
    """Transformação linear manual: y = x @ W + b"""
    
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicializar W e b (pequenos, randômicos)
        self.W = torch.randn(in_features, out_features) * 0.01
        self.b = torch.randn(out_features) * 0.01
        
        # Habilitar gradientes
        self.W.requires_grad = True
        self.b.requires_grad = True
    
    def forward(self, x):
        # x: [batch, in_features]
        # W: [in_features, out_features]
        # y: [batch, out_features]
        y = x @ self.W + self.b
        return y
    
    def parameters(self):
        return [self.W, self.b]

# Uso
layer = LinearManual(in_features=4, out_features=2)
x = torch.randn(3, 4)  # Batch de 3, cada um dim 4
y = layer.forward(x)   # [3, 2]

# Loss
loss = y.sum()
loss.backward()

print(f"W gradientes: {layer.W.grad}")
print(f"b gradientes: {layer.b.grad}")
```

---

## 🎯 nn.Linear

PyTorch fornece `nn.Linear` que faz tudo:

```python
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)

x = torch.randn(3, 4)  # Batch de 3
y = layer(x)           # [3, 2]

print(layer.weight.shape)  # [out, in] - Nota: transposta!
print(layer.bias.shape)    # [out]
```

**Nota importante**: Em `nn.Linear`, pesos são armazenados como `[out_features, in_features]`, não `[in_features, out_features]`. PyTorch transpõe internamente na computação: `y = x @ W.T + b`.

---

## 🧪 Experimento 1: Linear Manual vs nn.Linear

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Linear Manual vs nn.Linear")
print("=" * 70)

# ========== CONFIGURAÇÃO ==========
in_features = 4
out_features = 2
batch_size = 3

print(f"\nConfiguraçao:")
print(f"  in_features: {in_features}")
print(f"  out_features: {out_features}")
print(f"  batch_size: {batch_size}")

# ========== MANUAL ==========
print("\n1. LINEAR MANUAL")
print("-" * 70)

W = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
])  # [4, 2]

b = torch.tensor([0.1, 0.2])  # [2]

x = torch.ones(1, 4)  # [1, 4]
y_manual = x @ W + b
print(f"x = {x}")
print(f"W = \n{W}")
print(f"b = {b}")
print(f"y = x @ W + b = {y_manual}")

# Verificação manual
# y[0,0] = 1*1 + 1*3 + 1*5 + 1*7 + 0.1 = 16 + 0.1 = 16.1
# y[0,1] = 1*2 + 1*4 + 1*6 + 1*8 + 0.2 = 20 + 0.2 = 20.2
print(f"Esperado: [[16.1, 20.2]]")

# ========== nn.Linear ==========
print("\n2. nn.Linear")
print("-" * 70)

layer = nn.Linear(in_features=4, out_features=2)
# Copiar pesos para comparar
layer.weight.data = W.T  # nn.Linear armazena [out, in]
layer.bias.data = b

y_pytorch = layer(x)
print(f"y = {y_pytorch}")

# ========== BATCH ==========
print("\n3. BATCH FORWARD")
print("-" * 70)

x_batch = torch.randn(batch_size, in_features)
y_batch = layer(x_batch)

print(f"x_batch shape: {x_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")
print(f"y_batch:\n{y_batch}")

# ========== GRADIENTES ==========
print("\n4. GRADIENTES")
print("-" * 70)

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
layer = nn.Linear(4, 2)

y = layer(x)
loss = y.sum()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Weight gradientes:\n{layer.weight.grad}")
print(f"Bias gradientes: {layer.bias.grad}")

# ========== SHAPES ==========
print("\n5. SHAPES EM SEQUÊNCIAS")
print("-" * 70)

# Sequência de tokens: [batch, seq_len, embedding_dim]
x_seq = torch.randn(2, 10, 64)  # 2 exemplos, 10 tokens, embedding_dim 64

# Projeção
layer = nn.Linear(64, 128)
y_seq = layer(x_seq)  # [2, 10, 128]

print(f"x_seq shape: {x_seq.shape}")
print(f"layer: Linear(64, 128)")
print(f"y_seq shape: {y_seq.shape}")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

---

## 🧪 Experimento 2: Shapes em Diferentes Contextos

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Shapes em Diferentes Contextos")
print("=" * 70)

# ========== CONTEXTO 1: Embedding Projection ==========
print("\n1. EMBEDDING PROJECTION")
print("-" * 70)

# Embeddings: [batch, seq_len, embedding_dim]
embeddings = torch.randn(4, 10, 64)

# Projetar para nova dimensão
proj = nn.Linear(64, 32)
projected = proj(embeddings)

print(f"Embeddings: {embeddings.shape}")
print(f"Projeção Linear(64, 32)")
print(f"Resultado: {projected.shape}")

# ========== CONTEXTO 2: Query-Key-Value ==========
print("\n2. QUERY-KEY-VALUE PROJECTIONS")
print("-" * 70)

x = torch.randn(2, 10, 64)  # Embeddings

# 3 projeções diferentes
W_q = nn.Linear(64, 64)
W_k = nn.Linear(64, 64)
W_v = nn.Linear(64, 64)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

print(f"Input: {x.shape}")
print(f"Q = Linear(64,64)(x): {Q.shape}")
print(f"K = Linear(64,64)(x): {K.shape}")
print(f"V = Linear(64,64)(x): {V.shape}")

# ========== CONTEXTO 3: Feedforward ==========
print("\n3. FEEDFORWARD NETWORK")
print("-" * 70)

x = torch.randn(4, 10, 64)

# Expandir depois contrair
expand = nn.Linear(64, 256)
contract = nn.Linear(256, 64)

x_expanded = expand(x)  # [4, 10, 256]
x_final = contract(x_expanded)  # [4, 10, 64]

print(f"Input: {x.shape}")
print(f"Depois Linear(64, 256): {x_expanded.shape}")
print(f"Depois Linear(256, 64): {x_final.shape}")

# ========== CONTEXTO 4: Classification Head ==========
print("\n4. CLASSIFICATION HEAD")
print("-" * 70)

# Pooling: média de sequência
x = torch.randn(4, 10, 64)
x_pooled = x.mean(dim=1)  # [4, 64]

# Projeção para logits (num_classes)
num_classes = 10
head = nn.Linear(64, num_classes)
logits = head(x_pooled)  # [4, 10]

print(f"Input: {x.shape}")
print(f"Depois pooling: {x_pooled.shape}")
print(f"Depois Linear(64, {num_classes}): {logits.shape}")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

---

## ❌ Erros Comuns

### Erro 1: Shapes Incompatíveis

```python
# ❌ Errado
x = torch.randn(3, 5)  # [3, 5]
W = torch.randn(10, 5)  # [10, 5]
y = x @ W  # ERRO: (3,5) @ (10,5) não funciona

# ✓ Certo
W = torch.randn(5, 10)  # [5, 10]
y = x @ W  # [3, 10]
```

### Erro 2: Confundir peso de nn.Linear

```python
# nn.Linear internamente usa W.T
layer = nn.Linear(3, 2)
print(layer.weight.shape)  # [2, 3], não [3, 2]!
```

### Erro 3: Broadcasting errado com bias

```python
# ❌ Errado
x = torch.randn(4, 5)  # [4, 5]
W = torch.randn(5, 3)  # [5, 3]
b = torch.randn(4, 3)  # [4, 3] - ERRADO!
y = x @ W + b  # Broadcasting estranho

# ✓ Certo
b = torch.randn(3)  # [3]
y = x @ W + b  # [4, 3]
```

---

## ✍️ Exercícios

### Exercício 8.1: Linear Manual
Implemente LinearManual com in_features=3, out_features=2. Faça forward com x [1, 3].

### Exercício 8.2: nn.Linear
Crie nn.Linear(5, 3). Qual é o shape de weight? E de bias?

### Exercício 8.3: Batch Forward
x tem shape [32, 100]. Passe por Linear(100, 50). Qual é output?

### Exercício 8.4: Gradientes
Crie Linear, faça forward, backward. Verifique gradientes de W e b.

### Exercício 8.5: Sequências
x tem shape [2, 10, 64] (sequência). Passe por Linear(64, 128). Output?

---

## 📚 Gabarito

### Exercício 8.1: Linear Manual
```python
class LinearManual:
    def __init__(self, in_features, out_features):
        self.W = torch.randn(in_features, out_features) * 0.01
        self.b = torch.randn(out_features) * 0.01
        self.W.requires_grad = True
        self.b.requires_grad = True
    
    def forward(self, x):
        return x @ self.W + self.b

layer = LinearManual(3, 2)
x = torch.randn(1, 3)
y = layer.forward(x)  # [1, 2]
```

### Exercício 8.2: nn.Linear
```python
layer = nn.Linear(5, 3)
print(layer.weight.shape)  # [3, 5]
print(layer.bias.shape)    # [3]
```

### Exercício 8.3: Batch Forward
```python
x = torch.randn(32, 100)
layer = nn.Linear(100, 50)
y = layer(x)
print(y.shape)  # [32, 50]
```

### Exercício 8.4: Gradientes
```python
layer = nn.Linear(4, 2)
x = torch.randn(3, 4)
y = layer(x)
loss = y.sum()
loss.backward()
print(layer.weight.grad.shape)  # [2, 4]
print(layer.bias.grad.shape)    # [2]
```

### Exercício 8.5: Sequências
```python
x = torch.randn(2, 10, 64)
layer = nn.Linear(64, 128)
y = layer(x)
print(y.shape)  # [2, 10, 128]
```

---

## 🎯 Exercícios de Fixação (Opcionais)

### Fixação 8.1: Composição de Lineares
Crie duas camadas Linear(10, 5) e Linear(5, 2). Passe x [1, 10] através de ambas.

### Fixação 8.2: Identidade
Crie Linear(5, 5). Inicialize W como matriz identidade. Verifique que y ≈ x.

### Fixação 8.3: Zero-Initialization
Crie Linear. Inicialize com zeros. Qual é o output? O que acontece com gradientes?

### Fixação 8.4: Ortogonal Initialization
Use `torch.nn.init.orthogonal_` para inicializar. O que muda em relação ao normal?

### Fixação 8.5: Regularização L2
Adicione regularização L2 dos pesos à loss:

$$L_{total} = L_{task} + \lambda \|W\|_2$$

---

## 🎓 Resumo

- **Projeção linear**: $y = x @ W + b$
- **Manual**: Entender shapes e gradientes
- **nn.Linear**: Abstração conveniente (armazena W transposto)
- **Batch processing**: Broadcasting automático
- **Sequências**: Linear aplica a cada token independentemente

Próximo: **Origem de W e b** — por que esses valores são importantes e como inicializar.

---

**Próximo**: [Capítulo 09: Origem dos Parâmetros W e b](09_origem_parametros.md)
