# Capítulo 05: Operações Básicas

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Realizar operações elemento-wise eficientemente
2. Usar indexação e slicing em tensores
3. Manipular shapes com reshape, flatten, squeeze
4. Usar reduções (sum, mean, max) com diferentes eixos
5. Aplicar operações comuns em deep learning (exponencial, logaritmo, ativações)

---

## Por Que Isso Importa

Neste capítulo, cobrimos as "ferramentas do dia a dia" que você usará constantemente. Não são conceitos profundos — são habilidades práticas que precisa dominar.

Pense como aprender uma linguagem de programação: primeiro você aprende `for` loops, `if` statements, e funções. Aqui, estamos aprendendo a "sintaxe" do PyTorch.

---

## Indexação e Slicing (Pegar Partes de Tensores)

### Básico

```python
x = torch.arange(10)  # [0, 1, 2, ..., 9]

# Acesso por índice
print(x[0])      # 0 (primeiro elemento)
print(x[5])      # 5
print(x[-1])     # 9 (último)
print(x[-2])     # 8 (penúltimo)
```

### Slicing

```python
x = torch.arange(10)  # [0, 1, 2, ..., 9]

# Fatias (slices)
print(x[2:5])        # [2, 3, 4] (índices 2, 3, 4; não inclui 5)
print(x[:3])         # [0, 1, 2] (início até 3, não inclui)
print(x[5:])         # [5, 6, 7, 8, 9] (5 até fim)
print(x[::2])        # [0, 2, 4, 6, 8] (cada 2 elementos)
print(x[::-1])       # [9, 8, 7, ..., 0] (reverso)
```

### Matrizes (2D)

```python
X = torch.arange(12).reshape(3, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# Primeira linha
print(X[0])          # [0, 1, 2, 3]

# Elemento específico
print(X[1, 2])       # 6 (linha 1, coluna 2)

# Coluna específica
print(X[:, 2])       # [2, 6, 10]

# Submatriz
print(X[1:3, 1:3])   # [[5, 6], [9, 10]]
```

### Indexação Avançada

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Índices booleanos
mask = x > 2.5
print(x[mask])  # [3., 4., 5.]

# Gather (pega elementos em índices específicos)
indices = torch.tensor([0, 2, 4])
print(torch.index_select(x, 0, indices))  # [1., 3., 5.]
```

---

## Operações Elemento-wise

### Aritméticas

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)      # [5, 7, 9]
print(a - b)      # [-3, -3, -3]
print(a * b)      # [4, 10, 18] (NÃO é matriz mult!)
print(a / b)      # [0.25, 0.4, 0.5]
print(a ** b)     # [1, 32, 729]
```

### Funções Matemáticas

```python
x = torch.tensor([0.0, 0.5, 1.0, 2.0])

# Exponencial e logaritmo
print(torch.exp(x))     # [1, 1.65, 2.72, 7.39]
print(torch.log(x))     # [nan, -0.69, 0, 0.69]

# Trigonométricas
print(torch.sin(x))     # [0, 0.48, 0.84, 0.91]
print(torch.cos(x))     # [1, 0.88, 0.54, -0.42]

# Valor absoluto
print(torch.abs(torch.tensor([-1.0, -2.0, 3.0])))  # [1, 2, 3]

# Raiz quadrada
print(torch.sqrt(torch.tensor([1.0, 4.0, 9.0])))  # [1, 2, 3]

# Arredondamento
print(torch.round(torch.tensor([1.3, 1.7, 2.5])))  # [1, 2, 2]
```

---

## Reduções (Aggregating Data)

### Sum, Mean, Max

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Soma tudo
print(x.sum())           # 21.0

# Soma ao longo dimensão 0 (linhas -> coluna)
print(x.sum(dim=0))      # [5, 7, 9]

# Soma ao longo dimensão 1 (colunas -> linha)
print(x.sum(dim=1))      # [6, 15]

# Média
print(x.mean())          # 3.5
print(x.mean(dim=1))     # [2, 5]

# Max e argmax
print(x.max())           # 6
print(x.argmax())        # 5 (índice linear)
print(x.argmax(dim=1))   # [2, 2] (coluna do max em cada linha)

# Min
print(x.min())           # 1

# Keepdim (mantém dimensão)
print(x.sum(dim=1, keepdim=True))  # [[6], [15]]
```

### Cumulativo

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(torch.cumsum(x, dim=0))  # [1, 3, 6, 10]
print(torch.cumprod(x, dim=0)) # [1, 2, 6, 24]
```

---

## Manipulação de Shapes

### Reshape e View

```python
x = torch.arange(6)  # [0, 1, 2, 3, 4, 5]

# Reshape (cria cópia se necessário)
y = x.reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]

# View (rápido se contíguo)
z = x.view(3, 2)     # [[0, 1], [2, 3], [4, 5]]

# -1 infere dimensão
w = x.reshape(-1, 1)  # [[0], [1], [2], [3], [4], [5]] (shape [6, 1])
```

### Flatten e Unflatten

```python
x = torch.arange(12).reshape(3, 4)

# Flatten (para 1D)
flat = x.flatten()   # [0, 1, 2, ..., 11]

# Unflatten (volta para 3D)
unflat = flat.unflatten(0, (3, 4))  # Volta ao shape [3, 4]
```

### Squeeze e Unsqueeze

```python
x = torch.tensor([[1.0, 2.0, 3.0]])  # Shape [1, 3]

# Squeeze remove dimensões tamanho 1
squeezed = x.squeeze()  # Shape [3]
squeezed_dim = x.squeeze(dim=0)  # Força remover dim 0

# Unsqueeze adiciona dimensão tamanho 1
unsqueezed = x.unsqueeze(0)  # Shape [1, 1, 3]
unsqueezed = x.unsqueeze(2)  # Shape [1, 3, 1]
```

### Transpose e Permute

```python
# Transposta simples (inverte 2 últimas dims)
x = torch.randn(2, 3, 4)
y = x.T              # Shape [4, 3, 2]

# Permute (reordena todas dims)
z = x.permute(2, 0, 1)  # Shape [4, 2, 3]

# Transpose específico
w = x.transpose(0, 2)  # Troca dims 0 e 2
```

### Concatenação e Stacking

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# Cat (ao longo dim existente)
c = torch.cat([a, b], dim=0)  # Shape [4, 3]
d = torch.cat([a, b], dim=1)  # Shape [2, 6]

# Stack (cria nova dimensão)
s = torch.stack([a, b], dim=0)  # Shape [2, 2, 3]

# Unstack (oposto de stack)
s_list = torch.stack([a, b], dim=0).unbind(dim=0)  # [a, b]
```

---

## Ativações (Funções Não-Lineares)

Essas não são "operações básicas" puro, mas as usaremos constantemente. Familiarize-se:

### ReLU

```python
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])

relu = torch.nn.ReLU()
print(relu(x))  # [0, 0, 1, 2]

# Ou função direta
print(torch.relu(x))  # [0, 0, 1, 2]
```

### Sigmoid

```python
x = torch.tensor([-1.0, 0.0, 1.0])

sigmoid = torch.nn.Sigmoid()
print(sigmoid(x))  # [0.27, 0.5, 0.73]

# Ou função direta
print(torch.sigmoid(x))  # [0.27, 0.5, 0.73]
```

### Softmax

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 1.0, 1.0]])

softmax = torch.nn.Softmax(dim=1)
print(softmax(x))
# [[0.09, 0.24, 0.67],
#  [0.33, 0.33, 0.33]]
```

### Tanh

```python
x = torch.tensor([-1.0, 0.0, 1.0])
print(torch.tanh(x))  # [-0.76, 0, 0.76]
```

---

## Experimento: Operações em Sequências

Crie `experimento_operacoes.py`:

```python
import torch

print("=" * 70)
print("EXPERIMENTO: Operações Básicas em Tensores")
print("=" * 70)

# ========== 1. INDEXAÇÃO ==========
print("\n1. INDEXAÇÃO E SLICING")
print("-" * 70)

x = torch.arange(10)
print(f"x = {x}")
print(f"x[3] = {x[3]}")
print(f"x[2:7] = {x[2:7]}")
print(f"x[::2] = {x[::2]}")

# ========== 2. REDUÇÕES ==========
print("\n2. REDUÇÕES")
print("-" * 70)

X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(f"X = \n{X}")
print(f"sum() = {X.sum()}")
print(f"sum(dim=0) = {X.sum(dim=0)}")
print(f"sum(dim=1) = {X.sum(dim=1)}")
print(f"mean(dim=1) = {X.mean(dim=1)}")

# ========== 3. RESHAPE ==========
print("\n3. RESHAPE")
print("-" * 70)

x = torch.arange(12)
print(f"Original: {x.shape} = {x}")

reshaped = x.reshape(3, 4)
print(f"Reshaped [3, 4]:\n{reshaped}")

flat = reshaped.flatten()
print(f"Flatten: {flat.shape} = {flat}")

# ========== 4. SQUEEZE/UNSQUEEZE ==========
print("\n4. SQUEEZE/UNSQUEEZE")
print("-" * 70)

x = torch.randn(1, 3, 1)
print(f"Original shape: {x.shape}")

squeezed = x.squeeze()
print(f"After squeeze(): {squeezed.shape}")

unsqueezed = squeezed.unsqueeze(0)
print(f"After unsqueeze(0): {unsqueezed.shape}")

# ========== 5. CAT/STACK ==========
print("\n5. CAT E STACK")
print("-" * 70)

a = torch.randn(2, 3)
b = torch.randn(2, 3)

cat_result = torch.cat([a, b], dim=0)
print(f"a.shape: {a.shape}, b.shape: {b.shape}")
print(f"cat([a, b], dim=0).shape: {cat_result.shape}")

stack_result = torch.stack([a, b], dim=0)
print(f"stack([a, b], dim=0).shape: {stack_result.shape}")

# ========== 6. ELEMENTO-WISE ==========
print("\n6. OPERAÇÕES ELEMENTO-WISE")
print("-" * 70)

x = torch.tensor([-1.0, 0.0, 1.0, 2.0])

print(f"x = {x}")
print(f"exp(x) = {torch.exp(x)}")
print(f"sqrt(abs(x)) = {torch.sqrt(torch.abs(x))}")
print(f"sin(x) = {torch.sin(x)}")
print(f"log(abs(x)+1) = {torch.log(torch.abs(x) + 1)}")

# ========== 7. ATIVAÇÕES ==========
print("\n7. ATIVAÇÕES")
print("-" * 70)

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(f"x = {x}")
print(f"relu(x) = {torch.relu(x)}")
print(f"sigmoid(x) = {torch.sigmoid(x)}")
print(f"tanh(x) = {torch.tanh(x)}")

# ========== 8. SOFTMAX (PROBS) ==========
print("\n8. SOFTMAX")
print("-" * 70)

logits = torch.tensor([[1.0, 2.0, 3.0]])
probs = torch.softmax(logits, dim=1)

print(f"logits = {logits}")
print(f"softmax(logits, dim=1) = {probs}")
print(f"sum = {probs.sum():.4f}  # Deve ser ~1.0")

# ========== 9. ARGMAX (PRED) ==========
print("\n9. ARGMAX")
print("-" * 70)

scores = torch.tensor([[1.0, 3.0, 2.0],
                       [4.0, 1.0, 2.0]])

pred = scores.argmax(dim=1)
print(f"scores = \n{scores}")
print(f"argmax(dim=1) = {pred}  # Índices da coluna máxima")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_operacoes.py
```

---

## Erros Comuns

### Erro 1: Confundir * com @

```python
# ❌ Errado
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a * b  # Elemento-wise, não matriz mult

# ✓ Certo
c = a @ b.T  # Se quiser matriz mult
```

### Erro 2: Esquecer keepdim

```python
# ❌ Errado - perde dimensão
x = torch.randn(32, 10, 64)
mean = x.mean(dim=1)  # [32, 64] - perdeu dimensão 1

# ✓ Certo - mantém estrutura
mean = x.mean(dim=1, keepdim=True)  # [32, 1, 64]
```

### Erro 3: Broadcasting inesperado

```python
# ❌ Errado - result tem shape inesperado
a = torch.randn(1, 10)
b = torch.randn(5, 1)
c = a + b  # [5, 10] - broadcasting pode surprender

# ✓ Certo - ser explícito
c = a.expand(5, 10) + b.expand(5, 10)
```

---

## Para Você Praticar

### Exercício 5.1: Indexação
Crie um tensor [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Acesse apenas elementos pares.

### Exercício 5.2: Reshape
Crie um tensor com 24 elementos. Reshape para [2, 3, 4]. Qual é o shape?

### Exercício 5.3: Redução
Crie uma matriz [3, 4]. Compute soma por linha (dim=1) e por coluna (dim=0).

### Exercício 5.4: Ativação
Crie um tensor [-2, -1, 0, 1, 2]. Aplique ReLU, sigmoid e tanh.

### Exercício 5.5: Softmax
Crie logits [1, 2, 3]. Compute softmax. Verifique que soma é 1.

---

## Respostas

### Exercício 5.1: Indexação
```python
x = torch.arange(1, 11)
pares = x[1::2]  # [2, 4, 6, 8, 10]
```

### Exercício 5.2: Reshape
```python
x = torch.arange(24)
reshaped = x.reshape(2, 3, 4)
print(reshaped.shape)  # torch.Size([2, 3, 4])
```

### Exercício 5.3: Redução
```python
X = torch.randn(3, 4)
soma_linhas = X.sum(dim=1)  # [3]
soma_colunas = X.sum(dim=0)  # [4]
```

### Exercício 5.4: Ativação
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.relu(x))     # [0, 0, 0, 1, 2]
print(torch.sigmoid(x))  # [0.12, 0.27, 0.5, 0.73, 0.88]
print(torch.tanh(x))     # [-0.96, -0.76, 0, 0.76, 0.96]
```

### Exercício 5.5: Softmax
```python
logits = torch.tensor([1.0, 2.0, 3.0])
probs = torch.softmax(logits, dim=0)
print(probs)        # [0.09, 0.24, 0.67]
print(probs.sum())  # tensor(1.)
```

---

## Desafios Avançados (Opcionais)

### Fixação 5.1: Batch Processing
Crie um tensor [32, 10, 64] representando batch de 32 exemplos, cada um com 10 tokens e embedding dim 64.
- Compute a média de embedding por exemplo (dim 1): shape esperado [32, 64]
- Compute a média global de todos embeddings: shape esperado []

### Fixação 5.2: Masking Prático
Crie um tensor [3, 4]. Crie uma máscara booleana para elementos > 0.5. Incremente todos elementos mascarados por 10.

```python
x = torch.randn(3, 4)
mask = x > 0.5
x[mask] += 10
```

### Fixação 5.3: Batch Matrix Mult
PyTorch suporta batch matmul:
```python
A = torch.randn(32, 4, 6)  # 32 matrizes de [4, 6]
B = torch.randn(32, 6, 5)  # 32 matrizes de [6, 5]
C = A @ B  # [32, 4, 5] - multiplica batch-wise
```

Crie dois tensores [10, 8, 3] e [10, 3, 5]. Faça batch matmul. Qual é o resultado?

### Fixação 5.4: Implementar Log-Sum-Exp
Log-sum-exp numericamente estável:
$$\log(\sum_i e^{x_i}) = \max(x) + \log(\sum_i e^{x_i - \max(x)})$$

Implemente isso em PyTorch e compare com `torch.logsumexp`.

### Fixação 5.5: Batch Normalization
Normalize um batch [32, 64] para média 0 e std 1, usando:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

---

## Resumo

- **Indexação**: Acesso e slicing são práticos
- **Reduções**: sum, mean, max com dim controls
- **Shapes**: reshape, squeeze, unsqueeze são essenciais
- **Elemento-wise**: *, +, -, exp, log, etc.
- **Ativações**: ReLU, sigmoid, tanh, softmax

Agora você tem "ferramentas de trabalho" suficientes. Nos próximos capítulos, começamos a construir a LLM de verdade.

---

**Próximo**: [Capítulo 06: O Conceito de Embedding](06_embedding_conceito.md)
