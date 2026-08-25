# Capítulo 10: Mecanismo de Atenção

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender a intuição de atenção e por que é importante
2. Conhecer Query, Key, Value e seu papel
3. Computar attention scores manualmente
4. Entender por que atenção é uma forma de ponderação
5. Visualizar o que atenção está fazendo

---

## Por Que Isso Importa

Imagine que você está lendo uma frase:

> "O gato subiu no sofá porque estava cansado"

Ao processar "cansado", qual palavra é mais importante para entender isso? Provavelmente "gato", não "sofá".

Em redes neurais recorrentes antigas, o modelo processa sequencial: gato → subiu → no → sofá → porque → ... → cansado. A informação sobre "gato" passa por 5 transformações antes de chegar em "cansado", diluindo.

**Atenção** deixa qualquer token "olhar para trás" diretamente para qualquer outro token, ponderando por relevância.

---

## 📐 Query, Key, Value

O mecanismo de atenção funciona com três quantidades:

- **Query (Q)**: "O que estou procurando?" (posição atual)
- **Key (K)**: "Sou relevante para isso?" (todas as posições)
- **Value (V)**: "Qual informação devo passar?" (todas as posições)

Analogia: 

> Você (Query) entra em uma biblioteca (Keys/Values).
> Os livros têm títulos (Keys) que indicam se são relevantes.
> Os livros têm conteúdo (Values) que você lê.
> Você pega apenas livros relevantes (baseado em Query vs Keys).

---

## 🔢 O Cálculo Passo a Passo

### Entrada

Embeddings: $X \in \mathbb{R}^{[n, d]}$
- n = comprimento de sequência
- d = dimensão de embedding

### Passo 1: Projetar para Q, K, V

$$Q = X W_Q$$
$$K = X W_K$$
$$V = X W_V$$

Onde $W_Q, W_K, W_V \in \mathbb{R}^{[d, d]}$ (matrizes de projeção aprendidas).

Resultado: Q, K, V todos em $\mathbb{R}^{[n, d]}$ (mesma dimensão)

### Passo 2: Computar Scores

Para cada posição i e j, calcule similaridade entre Q[i] e K[j]:

$$\text{scores}[i,j] = Q[i] \cdot K[j] = \sum_k Q[i,k] K[j,k]$$

Em forma matricial:

$$\text{scores} = Q K^T \in \mathbb{R}^{[n, n]}$$

### Passo 3: Normalizar com Softmax

Scores brutos podem ser grandes. Converta para probabilidades:

$$\text{attention\_weights}[i,:] = \text{softmax}\left(\frac{\text{scores}[i,:]}{sqrt(d)}\right)$$

Nota: Dividimos por $\sqrt{d}$ para evitar que softmax sature.

### Passo 4: Ponderar Values

Use pesos de atenção para ponderar valores:

$$\text{output}[i] = \sum_j \text{attention\_weights}[i,j] \cdot V[j]$$

Em forma matricial:

$$\text{output} = \text{attention\_weights} \cdot V$$

---

## 📊 Exemplo Concreto Numérico

Vamos computar manualmente com números pequenos.

### Setup

- Sequência: ["O", "gato", "dormia"]
- Embedding dim: 4
- X shape: [3, 4]

```
X = [
    [0.1, -0.2, 0.3, 0.4],  # "O"
    [0.2, 0.1, -0.1, 0.5],  # "gato"
    [-0.1, 0.3, 0.2, -0.2]  # "dormia"
]
```

### Matrizes de Projeção

```
W_Q = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]  # Identidade, por simplicidade

W_K = W_V = W_Q  # Identidade também
```

(Normalmente seriam aleatórios, aqui usamos identidade para clareza)

### Passo 1: Q, K, V

```
Q = K = V = X (porque W_Q = W_K = W_V = I)
```

### Passo 2: Scores

```
scores = Q @ K^T = X @ X^T

scores[0,0] = X[0] · X[0] = 0.1² + 0.2² + 0.3² + 0.4² = 0.3
scores[0,1] = X[0] · X[1] = 0.1*0.2 + (-0.2)*0.1 + 0.3*(-0.1) + 0.4*0.5
             = 0.02 - 0.02 - 0.03 + 0.2 = 0.17
scores[0,2] = X[0] · X[2] = 0.1*(-0.1) + (-0.2)*0.3 + 0.3*0.2 + 0.4*(-0.2)
             = -0.01 - 0.06 + 0.06 - 0.08 = -0.09
```

(Computaríamos todas 9 entradas assim)

### Passo 3: Attention Weights (Softmax)

Para posição 0:
```
scores[0,:] = [0.3, 0.17, -0.09]
scores[0,:] / sqrt(4) = [0.15, 0.085, -0.045]

# softmax([0.15, 0.085, -0.045])
exp(0.15) = 1.162
exp(0.085) = 1.089
exp(-0.045) = 0.956
soma = 3.207

probs = [1.162/3.207, 1.089/3.207, 0.956/3.207]
      = [0.362, 0.339, 0.298]
```

### Passo 4: Output

```
output[0] = 0.362 * V[0] + 0.339 * V[1] + 0.298 * V[2]
          = 0.362 * X[0] + 0.339 * X[1] + 0.298 * X[2]
          # Ponderação de todos os tokens
```

**Interpretação**: Ao processar "O", a atenção dá 36.2% peso ao próprio "O", 33.9% a "gato", 29.8% a "dormia".

---

## Experimento: Atenção Manual

Crie `experimento_atencao_manual.py`:

```python
import torch
import torch.nn.functional as F

print("=" * 70)
print("EXPERIMENTO: Atenção Manual")
print("=" * 70)

# ========== CONFIGURAÇÃO ==========
n = 3  # Comprimento de sequência
d = 4  # Dimensão de embedding

print(f"\nConfiguraçao:")
print(f"  n (seq_len): {n}")
print(f"  d (embedding_dim): {d}")

# ========== ENTRADA ==========
print("\n1. ENTRADA (X)")
print("-" * 70)

X = torch.tensor([
    [0.1, -0.2, 0.3, 0.4],   # "O"
    [0.2, 0.1, -0.1, 0.5],   # "gato"
    [-0.1, 0.3, 0.2, -0.2]   # "dormia"
], dtype=torch.float32)

print(f"X shape: {X.shape}")
print(f"X =\n{X}")

# ========== PROJEÇÕES ==========
print("\n2. PROJEÇÕES Q, K, V")
print("-" * 70)

# Matrizes de projeção
W_Q = torch.eye(d)  # Identidade por simplicidade
W_K = torch.eye(d)
W_V = torch.eye(d)

Q = X @ W_Q  # [3, 4]
K = X @ W_K  # [3, 4]
V = X @ W_V  # [3, 4]

print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
print(f"Q =\n{Q}")

# ========== SCORES ==========
print("\n3. ATTENTION SCORES")
print("-" * 70)

scores = Q @ K.T  # [3, 3]
print(f"scores = Q @ K^T:")
print(f"scores shape: {scores.shape}")
print(f"scores =\n{scores}")

# ========== NORMALIZAÇÃO ==========
print("\n4. NORMALIZAÇÃO PELO SQRT(d)")
print("-" * 70)

scores_scaled = scores / (d ** 0.5)
print(f"scores / sqrt(d) =\n{scores_scaled}")

# ========== SOFTMAX ==========
print("\n5. SOFTMAX")
print("-" * 70)

attention_weights = F.softmax(scores_scaled, dim=1)
print(f"attention_weights shape: {attention_weights.shape}")
print(f"attention_weights (por linha deve somar 1) =\n{attention_weights}")
print(f"Somas por linha: {attention_weights.sum(dim=1)}")

# ========== PONDERAÇÃO ==========
print("\n6. OUTPUT = attention_weights @ V")
print("-" * 70)

output = attention_weights @ V  # [3, 4]
print(f"output shape: {output.shape}")
print(f"output =\n{output}")

# ========== INTERPRETAÇÃO ==========
print("\n7. INTERPRETAÇÃO")
print("-" * 70)

print("Atenção por posição:")
print(f"Posição 0 ('O'):       [O={attention_weights[0,0]:.3f}, gato={attention_weights[0,1]:.3f}, dormia={attention_weights[0,2]:.3f}]")
print(f"Posição 1 ('gato'):    [O={attention_weights[1,0]:.3f}, gato={attention_weights[1,1]:.3f}, dormia={attention_weights[1,2]:.3f}]")
print(f"Posição 2 ('dormia'):  [O={attention_weights[2,0]:.3f}, gato={attention_weights[2,1]:.3f}, dormia={attention_weights[2,2]:.3f}]")

print("\nInterpretação:")
print("Cada token 'olha' para todos tokens com diferentes pesos")
print("Pesos maiores significam 'mais importante para entender este token'")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_atencao_manual.py
```

---

## Experimento 2: Matriz de Atenção Visual

```python
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("EXPERIMENTO: Visualizar Matriz de Atenção")
print("=" * 70)

# Setup
n = 5
d = 8

torch.manual_seed(42)

# Embedding randômico
X = torch.randn(n, d)

# Projeções aprendidas (aleatórias)
W_Q = torch.randn(d, d) * 0.1
W_K = torch.randn(d, d) * 0.1
W_V = torch.randn(d, d) * 0.1

# Forward
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T / (d ** 0.5)
attention_weights = F.softmax(scores, dim=1)
output = attention_weights @ V

print("Matriz de atenção (attention_weights):")
print(attention_weights)

print("\nInterpretação:")
print("- Linhas: posições de origem (query)")
print("- Colunas: posições de destino (key/value)")
print("- Valores: peso de atenção (0-1)")

# Padrão
print("\nPadrões observáveis:")
for i in range(n):
    max_j = attention_weights[i].argmax().item()
    max_weight = attention_weights[i, max_j].item()
    print(f"Pos {i}: maior atenção para pos {max_j} ({max_weight:.3f})")
```

---

## Erros Comuns

### Erro 1: Não dividir por sqrt(d)

```python
# ❌ Errado
scores = Q @ K.T
attn = F.softmax(scores, dim=1)  # Scores podem ser muito grandes!

# ✓ Certo
scores = Q @ K.T / (d ** 0.5)
attn = F.softmax(scores, dim=1)
```

### Erro 2: Softmax na dimensão errada

```python
# ❌ Errado
attn = F.softmax(scores, dim=0)  # Errado! Soma colunas a 1

# ✓ Certo
attn = F.softmax(scores, dim=1)  # Soma linhas a 1
```

### Erro 3: Esquecer de transpor K

```python
# ❌ Errado
scores = Q @ K  # Shapes: [n,d] @ [n,d] = ERRO

# ✓ Certo
scores = Q @ K.T  # [n,d] @ [d,n] = [n,n]
```

---

## Exercícios

### Exercício 10.1: Computar Scores Manualmente
Q = [[1, 0], [0, 1]], K = [[1, 1], [0, 0]]. Compute scores = Q @ K.T.

### Exercício 10.2: Softmax
scores = [1, 2, 3]. Compute softmax manualmente.

### Exercício 10.3: Atenção Completa
X [2, 3], W_Q = W_K = W_V = I. Compute saída de atenção completa.

### Exercício 10.4: Diferentes Escalas
Compute attenção com e sem divisão por sqrt(d). Compare.

### Exercício 10.5: Interpretação
Dada matriz de atenção, para cada token, identifique qual token ele "mais olha".

---

## Gabarito

### Exercício 10.1: Scores
```python
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
K = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
scores = Q @ K.T
# scores[0,0] = 1*1 + 0*0 = 1
# scores[0,1] = 1*1 + 0*0 = 1
# scores[1,0] = 0*1 + 1*0 = 0
# scores[1,1] = 0*1 + 1*0 = 0
# resultado: [[1, 1], [0, 0]]
```

### Exercício 10.2: Softmax
```python
import torch
scores = torch.tensor([1.0, 2.0, 3.0])
probs = torch.softmax(scores, dim=0)
# Manual: exp([1,2,3]) = [2.718, 7.389, 20.086]
# soma = 30.192
# probs = [0.090, 0.245, 0.665]
print(probs)  # tensor([0.0900, 0.2447, 0.6652])
```

### Exercício 10.3: Atenção Completa
```python
import torch.nn.functional as F

X = torch.randn(2, 3)
W_Q = W_K = W_V = torch.eye(3)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T / (3 ** 0.5)
attn = F.softmax(scores, dim=1)
output = attn @ V

print(output.shape)  # [2, 3]
```

### Exercício 10.4: Com e Sem Scaling
```python
# Sem escala
attn1 = F.softmax(Q @ K.T, dim=1)

# Com escala
attn2 = F.softmax(Q @ K.T / (d ** 0.5), dim=1)

# Com escala tem distribuição mais uniforme
print(attn1[0])  # Mais "picos"
print(attn2[0])  # Mais "plana"
```

### Exercício 10.5: Interpretação
```python
attn = torch.tensor([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.1, 0.1, 0.8]
])

for i in range(3):
    max_j = attn[i].argmax()
    print(f"Token {i} mais olha para {max_j}")
# Token 0 mais olha para 0
# Token 1 mais olha para 1
# Token 2 mais olha para 2
```

---

## Desafios Avançados (Opcionais)

### Fixação 10.1: Diferentes d
Compute atenção com d=4 e d=64. Como a divisão por sqrt(d) muda a distribuição?

### Fixação 10.2: Verificar Simetria
Se Q = K = V, a matriz de atenção é simétrica? Verifique numericamente.

### Fixação 10.3: Gradientes de Atenção
Faça backward em atenção. Verifique que W_Q, W_K, W_V recebem gradientes.

### Fixação 10.4: Entropia de Atenção
Compute entropia da distribuição de atenção. Quando é alta (uniforme) vs baixa (concentrada)?

### Fixação 10.5: Masking Preview
(Adiantando do próximo cap) Se você quiser que posição i NÃO olhe para posição j > i, como você fazeria?

Dica: use um vetor de máscara antes do softmax.

---

## Resumo

- **Atenção**: Mecanismo de ponderação baseado em relevância
- **Q, K, V**: Query, Key, Value — papéis diferentes
- **Scores**: Similaridade entre Q e K (dot product)
- **Softmax**: Converte scores em probabilidades
- **Output**: Média ponderada de Values
- **Escalamento**: Dividir por sqrt(d) é crucial

Próximo capítulo: **Masking causal** — para modelos autoregressivos.

---

**Próximo**: [Capítulo 11: Query, Key, Value Explicados](11_query_key_value.md)
