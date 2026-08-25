# Capítulo 09: Origem dos Parâmetros W e b

## 🎯 Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender por que W e b têm as formas que têm
2. Conhecer estratégias de inicialização comuns
3. Entender problemas de inicialização (vanishing/exploding gradients)
4. Saber escolher inicialização para seu caso
5. Debugar problemas de treinamento relacionados a W/b

---

## 💡 Intuição

Os parâmetros W e b não são "dados". São valores que o modelo **aprende durante o treinamento**.

Mas comece em algum lugar. A **inicialização** desses valores influencia se o treinamento funcionará bem ou não.

Inicialização ruim = treinamento lento ou travado.
Inicialização boa = convergência rápida.

---

## 📐 Por Que W Tem Shape [in_features, out_features]?

Uma transformação linear projeta de um espaço para outro:

$$\mathbf{y} = \mathbf{x} \mathbf{W}$$

- $\mathbf{x} \in \mathbb{R}^{1 \times d_{in}}$ (um exemplo)
- $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$
- $\mathbf{y} \in \mathbb{R}^{1 \times d_{out}}$

Para o batch:
- $\mathbf{X} \in \mathbb{R}^{B \times d_{in}}$ (B exemplos)
- $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$ (mesma para todo batch)
- $\mathbf{Y} \in \mathbb{R}^{B \times d_{out}}$

**W precisa ter shape [d_in, d_out] para que a multiplicação faça sentido.**

### Visualizar

```
Entrada:        Pesos:          Saída:
[batch]         [in, out]       [batch]
   1  ×  d_in    d_in × d_out  =   1  × d_out
   1  ×  d_in    d_in × d_out  =   1  × d_out
   ...
   B  × d_in     d_in × d_out  =   B  × d_out
```

---

## 🎯 Por Que b Tem Shape [out_features]?

O bias adiciona um offset (deslocamento) a cada dimensão de saída:

$$y_j = \sum_{i} x_i W_{ij} + b_j$$

Cada dimensão de saída tem seu próprio bias. Logo, b tem shape [out_features].

**Broadcasting**: b é automaticamente broadcast para todas as linhas do batch.

```python
X = torch.randn(32, 100)  # [32, 100]
W = torch.randn(100, 50)  # [100, 50]
b = torch.randn(50)       # [50]

Y = X @ W + b  # [32, 50]
# b é broadcast: [50] -> [1, 50] -> [32, 50]
```

---

## 🎲 Estratégias de Inicialização

### 1. Zeros (❌ Ruim)

```python
torch.nn.init.zeros_(W)
torch.nn.init.zeros_(b)
```

**Problema**: Simetria. Todos neurônios começam idênticos. Aprendem a fazer coisa idêntica.

### 2. Aleatório Uniforme [0, 1)

```python
torch.nn.init.uniform_(W, 0, 1)
```

**Problema**: Valores grandes causam saídas grandes, o que causa problemas com softmax e gradientes.

### 3. Gaussiana Pequena (✅ Comum)

```python
torch.nn.init.normal_(W, mean=0, std=0.01)
torch.nn.init.zeros_(b)  # Bias em zero
```

**Vantagem**: Simples, quebra simetria, valores pequenos inicialmente.

### 4. Xavier/Glorot (✅ Recomendado)

Mantém variância de sinais através de camadas:

$$\text{std} = \sqrt{\frac{2}{d_{in} + d_{out}}}$$

```python
torch.nn.init.xavier_uniform_(W)  # Uniforme [-limit, limit]
torch.nn.init.xavier_normal_(W)   # Normal
```

**Vantagem**: Teórico fundamento. Funciona bem para sigmoid/tanh.

### 5. Kaiming/He (✅ Para ReLU)

Ajustado para redes com ReLU:

$$\text{std} = \sqrt{\frac{2}{d_{in}}}$$

```python
torch.nn.init.kaiming_uniform_(W)
torch.nn.init.kaiming_normal_(W)
```

**Vantagem**: Otimizado para ReLU, evita problemas de sinal desaparecendo.

### 6. Ortogonal (✅ Avançado)

```python
torch.nn.init.orthogonal_(W)
```

W é uma matriz ortogonal. Preserva normas de sinais.

**Vantagem**: Propriedades matemáticas fortes, bom para RNNs.

---

## 📊 Comparação: O Que Acontece

Simule o forward pass com diferentes inicializações:

```python
import torch
import torch.nn as nn
from torch.nn import init

# Configuração
d_in = 100
d_out = 100
batch_size = 32
num_layers = 10

x = torch.randn(batch_size, d_in)

strategies = {
    "Zeros": lambda W: init.zeros_(W),
    "Uniform [0,1)": lambda W: init.uniform_(W, 0, 1),
    "Normal (0, 0.01)": lambda W: init.normal_(W, 0, 0.01),
    "Xavier": lambda W: init.xavier_normal_(W),
    "Kaiming": lambda W: init.kaiming_normal_(W),
    "Orthogonal": lambda W: init.orthogonal_(W),
}

for name, init_fn in strategies.items():
    x_curr = x.clone()
    norms = []
    
    for layer in range(num_layers):
        W = torch.randn(d_in, d_out)
        init_fn(W)
        b = torch.zeros(d_out)
        
        x_curr = x_curr @ W + b
        norms.append(x_curr.norm().item())
    
    print(f"{name:20s}: norms = {norms[:3]} ... {norms[-1]:.4f}")
```

**Observação**: Com inicialização ruim, norms crescem ou diminuem drasticamente (exploding/vanishing gradients).

---

## ⚠️ Problemas Comuns de Inicialização

### Problema 1: Exploding Gradients

W inicializado muito grande → sinais crescem exponencialmente → gradientes explode → training falha.

### Problema 2: Vanishing Gradients

W inicializado muito pequeno → sinais encolhem exponencialmente → gradientes → 0 → no learning.

### Problema 3: Simetria

W inicializado com valor fixo (ex: 0.5) → simetria → todos neurônios aprendem a coisa idêntica.

### Solução

Use Xavier/Kaiming com inicialização aleatória. PyTorch faz automaticamente:

```python
layer = nn.Linear(100, 100)
# PyTorch já inicializa W com Xavier por padrão!
```

---

## 🧪 Experimento: Inicialização e Treinamento

```python
import torch
import torch.nn as nn
from torch.nn import init

print("=" * 70)
print("EXPERIMENTO: Inicialização e Seu Impacto")
print("=" * 70)

# ========== DADOS ==========
torch.manual_seed(42)
X_train = torch.randn(100, 10)
y_train = (X_train[:, 0] > 0).float()  # Binary classification

# ========== TREINAMENTO COM DIFERENTES INICIALIZAÇÕES ==========
strategies = {
    "Zeros": lambda W, b: (init.zeros_(W), init.zeros_(b)),
    "Normal(0,0.01)": lambda W, b: (init.normal_(W, 0, 0.01), init.zeros_(b)),
    "Xavier": lambda W, b: (init.xavier_normal_(W), init.zeros_(b)),
    "Kaiming": lambda W, b: (init.kaiming_normal_(W), init.zeros_(b)),
}

for strategy_name, init_fn in strategies.items():
    print(f"\n{strategy_name}")
    print("-" * 70)
    
    # Modelo
    model = nn.Linear(10, 1)
    init_fn(model.weight, model.bias)
    
    # Otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Treinamento
    losses = []
    for epoch in range(100):
        y_pred = torch.sigmoid(model(X_train))
        loss = nn.BCELoss()(y_pred, y_train.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Report
    initial_loss = losses[0]
    final_loss = losses[-1]
    improved = initial_loss - final_loss
    
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")
    print(f"  Improvement:  {improved:.6f}")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

---

## 🔍 Debugar Problemas de Inicialização

```python
# 1. Verificar normas de W
print(f"W.norm() = {W.norm().item():.4f}")
# Esperado: pequeno (< 1)

# 2. Verificar distribuição de W
print(f"W.mean() = {W.mean().item():.4f}")  # Esperado: ~0
print(f"W.std() = {W.std().item():.4f}")    # Esperado: pequeno

# 3. Após forward pass, verificar normas de ativações
x = torch.randn(32, d_in)
for layer in model:
    x = layer(x)
    print(f"After layer: x.norm() = {x.norm().item():.4f}")
# Esperado: estável, não crescendo/encolhendo

# 4. Verificar gradientes durante backward
loss = model(x).sum()
loss.backward()
for name, param in model.named_parameters():
    print(f"{name}: grad.norm() = {param.grad.norm().item():.6f}")
# Esperado: gradientes não muito grandes nem muito pequenos
```

---

## ✍️ Exercícios

### Exercício 9.1: Shape de W
Linear(5, 3): qual é o shape de W? (Lembre: nn.Linear armazena [out, in])

### Exercício 9.2: Inicializar Zeros
W, b = zeros. Faça forward. O que sai?

### Exercício 9.3: Inicializar Normal
W ~ N(0, 0.01), b = 0. Qual é W.std()?

### Exercício 9.4: Xavier vs Kaiming
Quando usar Xavier? Quando usar Kaiming?

### Exercício 9.5: Verificar Inicialização
Crie nn.Linear. Imprima W.norm(). É razoável?

---

## 📚 Gabarito

### Exercício 9.1: Shape de W
```python
layer = nn.Linear(5, 3)
print(layer.weight.shape)  # [3, 5]
# nn.Linear armazena [out_features, in_features]
```

### Exercício 9.2: Inicializar Zeros
```python
W = torch.zeros(5, 3)
b = torch.zeros(3)
x = torch.ones(1, 5)
y = x @ W + b  # [1, 3] de zeros
print(y)  # tensor([[0., 0., 0.]])
```

### Exercício 9.3: Inicializar Normal
```python
W = torch.randn(5, 3) * 0.01
print(W.std().item())  # ~0.01
```

### Exercício 9.4: Xavier vs Kaiming
- **Xavier**: Bom para sigmoid/tanh. Mantém variância através de camadas.
- **Kaiming**: Para ReLU. Ajusta para perda de sinal que ReLU causa (mata negativos).

### Exercício 9.5: Verificar Inicialização
```python
layer = nn.Linear(100, 100)
print(layer.weight.norm().item())  # Típico: 1.0 - 2.0 (Xavier)
```

---

## 🎯 Exercícios de Fixação (Opcionais)

### Fixação 9.1: Calcular Xavier
$$\text{std} = \sqrt{\frac{2}{d_{in} + d_{out}}}$$

Para Linear(100, 100), calcule std esperado. Crie W com esse std, verifique.

### Fixação 9.2: Calcular Kaiming
$$\text{std} = \sqrt{\frac{2}{d_{in}}}$$

Para Linear(100, 100), calcule std. Crie W, verifique.

### Fixação 9.3: Treinar com Diferentes Inicializações
Treine modelo simples com 3 inicializações diferentes. Compare velocidade de convergência.

### Fixação 9.4: Simetria
Crie rede onde todos W têm mesmo valor (não aleatório). Treine. Observe que não aprende.

### Fixação 9.5: Verificar Gradientes
Após backward, verifique que W.grad.norm() é razoável (não NaN, não infinito).

---

## 🎓 Resumo

- **W shape**: [in_features, out_features] (ou transposto em nn.Linear)
- **b shape**: [out_features]
- **Inicialização**: Aleatória, quebra simetria
- **Xavier**: General (sigmoid/tanh)
- **Kaiming**: Para ReLU
- **Problemas**: Vanishing/exploding gradients se mal inicializado

Próximo: **Atenção** — o mecanismo mais importante em Transformers.

---

**Próximo**: [Capítulo 10: Mecanismo de Atenção](10_mecanismo_atencao.md)
