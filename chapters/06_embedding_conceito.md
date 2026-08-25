# Capítulo 06: O Conceito de Embedding

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o que é um embedding e por que precisamos deles
2. Visualizar embeddings como posições em espaço multidimensional
3. Compreender "similaridade" no espaço de embeddings
4. Entender por que embeddings densos são melhor que one-hot
5. Ver exemplos concretos de embeddings em linguagem

---

## Por Que Isso Importa

Imagine que você quer representar palavras para uma rede neural.

### Abordagem Ingênua: One-Hot

Para um vocabulário de 4 palavras: ["gato", "cachorro", "peixe", "passaro"]

```
"gato"     = [1, 0, 0, 0]
"cachorro" = [0, 1, 0, 0]
"peixe"    = [0, 0, 1, 0]
"passaro"  = [0, 0, 0, 1]
```

**Problemas**:
1. **Espaço desperdiçado**: vocabulário de 10000 palavras = vetores de 10000 dimensões, quase 100% zeros
2. **Sem semântica**: "gato" e "cachorro" são completamente ortogonais, sem noção de que são parecidos
3. **Difícil para NN**: redes neurais preferem trabalhar com números densos

### Abordagem Melhor: Embeddings Densos

Em vez de one-hot, aprendemos uma representação densa:

```
"gato"     = [0.2, -0.5, 0.8, 0.1]
"cachorro" = [0.3, -0.4, 0.7, 0.15]
"peixe"    = [0.1, 0.6, -0.2, -0.3]
"passaro"  = [0.4, 0.5, -0.1, 0.2]
```

**Vantagens**:
1. **Compacto**: [10000, 64] matriz ao invés de [10000, 10000]
2. **Semântico**: "gato" e "cachorro" estão próximos no espaço
3. **Aprendível**: a rede neural aprende essas representações
4. **Transferível**: embeddings podem ser reutilizados entre modelos

---

## Embedding como Lookup Table

Formalmente, um **embedding** é uma matriz $\mathbf{E} \in \mathbb{R}^{V \times d}$ onde:
- $V$ = tamanho do vocabulário (número de tokens únicos)
- $d$ = dimensão do embedding

Para um token $i$:

$$\text{embedding}(i) = \mathbf{E}[i, :]$$

Ou seja: a $i$-ésima linha da matriz.

### Exemplo Concreto

Suponha:
- Vocabulário: ["gato", "cachorro", "peixe"] (3 tokens)
- Embedding dim: 4
- Matriz $\mathbf{E}$ tem shape [3, 4]

```
       d1    d2    d3    d4
gato   0.2  -0.5   0.8   0.1
cachorro 0.3 -0.4  0.7   0.15
peixe   0.1  0.6  -0.2  -0.3
```

Para obter embedding de "gato" (índice 0):

$$\text{embedding}(\text{"gato"}) = [0.2, -0.5, 0.8, 0.1]$$

Em PyTorch:

```python
E = torch.tensor([
    [0.2, -0.5, 0.8, 0.1],
    [0.3, -0.4, 0.7, 0.15],
    [0.1, 0.6, -0.2, -0.3]
])

token_id = 0  # "gato"
emb = E[token_id, :]  # [0.2, -0.5, 0.8, 0.1]
```

---

## 🌍 Espaço Semântico

Uma propriedade importante: **similaridade** entre embeddings reflete similaridade de significado.

### Similaridade por Dot Product

Quanto maior o dot product entre dois embeddings, mais "parecidos" eles são:

```python
e_gato = torch.tensor([0.2, -0.5, 0.8, 0.1])
e_cachorro = torch.tensor([0.3, -0.4, 0.7, 0.15])
e_peixe = torch.tensor([0.1, 0.6, -0.2, -0.3])

# Similaridade gato-cachorro
sim_gato_cachorro = torch.dot(e_gato, e_cachorro)
# 0.2*0.3 + (-0.5)*(-0.4) + 0.8*0.7 + 0.1*0.15
# = 0.06 + 0.2 + 0.56 + 0.015 = 0.835

# Similaridade gato-peixe
sim_gato_peixe = torch.dot(e_gato, e_peixe)
# 0.2*0.1 + (-0.5)*0.6 + 0.8*(-0.2) + 0.1*(-0.3)
# = 0.02 - 0.3 - 0.16 - 0.03 = -0.47

print(f"Gato-Cachorro: {sim_gato_cachorro:.3f}")  # 0.835
print(f"Gato-Peixe: {sim_gato_peixe:.3f}")        # -0.47
```

Gato é mais parecido com cachorro (0.835) do que com peixe (-0.47). Faz sentido!

### Similaridade por Cosseno

Podemos normalizar por magnitude (cosseno da ângulo):

$$\text{cosseno}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

```python
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

sim = cosine_similarity(e_gato, e_cachorro)
print(f"Cosseno(gato, cachorro): {sim:.3f}")
```

Cosseno sempre está entre -1 e 1, facilitando interpretação.

---

## 🎯 Treinamento de Embeddings

Os valores na matriz $\mathbf{E}$ **não são fixados**. São aprendidos durante o treinamento.

Inicialmente, podem ser aleatórios:

```python
V = 10000  # tamanho vocabulário
d = 64     # dimensão embedding

# Inicialização aleatória
E = torch.randn(V, d) * 0.01  # Escala pequena
E.requires_grad = True  # Vamos aprender esses valores
```

Durante o treinamento, gradientes atualizam $\mathbf{E}$:

```python
# Forward pass (veremos depois)
loss = ... # Loss computado com base em embeddings

# Backward (calcula gradientes)
loss.backward()

# Update (com otimizador, tipo Adam)
optimizer.step()  # Atualiza E baseado em gradientes
```

Após treinamento em bilhões de textos, **emergem representações semânticas**. Exemplos reais:

- Embeddings de GloVe: "king" - "man" + "woman" ≈ "queen"
- Embeddings de Word2Vec: "Paris" - "France" + "Italy" ≈ "Rome"

Isso é "mágico" porque nenhuma regra foi codificada — a rede aprendeu!

---

## 📊 Visualizando Embeddings

Embeddings em alta dimensão (d=64, d=256, etc.) são impossíveis de visualizar direto. Usamos **redução de dimensionalidade**:

### t-SNE e UMAP

Técnicas que projetam embeddings de alta dimensão para 2D preservando proximidades:

```python
# Pseudo-código (usando sklearn)
from sklearn.manifold import TSNE

E_2d = TSNE(n_components=2).fit_transform(E.detach().numpy())
# Agora pode plotar com matplotlib

import matplotlib.pyplot as plt
plt.scatter(E_2d[:, 0], E_2d[:, 1])
plt.show()
```

Em uma visualização típica de embeddings de palavras treinados:
- Palavras relacionadas (animais, alimentos, etc.) formam clusters
- Relações semânticas aparecem como padrões (ex: "king - man + woman" é um vetor)

---

## 🔄 Batch de Embeddings

Na prática, queremos embeddings de múltiplos tokens de uma vez.

### Caso Simples: Sequência de Tokens

Entrada: sequência de IDs [7, 3, 12, 5] (4 tokens)
Matriz E: [V, d] onde V = vocab_size, d = embedding_dim

```python
token_ids = torch.tensor([7, 3, 12, 5])  # Shape [4]
embeddings = E[token_ids]  # Shape [4, d]

# Se E tem shape [vocab_size, d], resultado tem shape [4, d]
```

### Batch de Sequências

Entrada: batch de 32 sequências, cada uma com 10 tokens.

Tensor de entrada: [32, 10] (batch_size, seq_len)
Matriz E: [vocab_size, d]

```python
batch_token_ids = torch.randint(0, vocab_size, (32, 10))
# Shape [32, 10]

batch_embeddings = E[batch_token_ids]
# Shape [32, 10, d]
```

PyTorch "faz a mágica" de indexação em batch. Veremos isso no próximo capítulo.

---

## Experimento: Embeddings Simples

Crie `experimento_embeddings.py`:

```python
import torch

print("=" * 70)
print("EXPERIMENTO: Embeddings Conceitual")
print("=" * 70)

# ========== 1. MATRIZ DE EMBEDDING ==========
print("\n1. MATRIZ DE EMBEDDING")
print("-" * 70)

# Vocabulário: ["gato", "cachorro", "peixe", "passaro"]
# Embedding dim: 4

E = torch.tensor([
    [0.2, -0.5, 0.8, 0.1],      # gato (idx 0)
    [0.3, -0.4, 0.7, 0.15],     # cachorro (idx 1)
    [0.1, 0.6, -0.2, -0.3],     # peixe (idx 2)
    [0.4, 0.5, -0.1, 0.2]       # passaro (idx 3)
])

vocab = ["gato", "cachorro", "peixe", "passaro"]

print("Matriz E (shape [4, 4]):")
print(E)

# ========== 2. LOOKUP ==========
print("\n2. LOOKUP DE EMBEDDINGS")
print("-" * 70)

for idx, word in enumerate(vocab):
    embedding = E[idx]
    print(f"{word:10s} -> {embedding.tolist()}")

# ========== 3. SIMILARIDADE ==========
print("\n3. SIMILARIDADE POR DOT PRODUCT")
print("-" * 70)

pairs = [
    (0, 1, "gato", "cachorro"),
    (0, 2, "gato", "peixe"),
    (0, 3, "gato", "passaro"),
    (1, 2, "cachorro", "peixe"),
]

for i, j, word_i, word_j in pairs:
    e_i = E[i]
    e_j = E[j]
    similarity = torch.dot(e_i, e_j)
    print(f"Similaridade({word_i}, {word_j}): {similarity:.4f}")

# ========== 4. SIMILARIDADE COSSENO ==========
print("\n4. SIMILARIDADE COSSENO")
print("-" * 70)

def cosine_sim(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

for i, j, word_i, word_j in pairs:
    e_i = E[i]
    e_j = E[j]
    cos_sim = cosine_sim(e_i, e_j)
    print(f"Cosseno({word_i}, {word_j}): {cos_sim:.4f}")

# ========== 5. BATCH LOOKUP ==========
print("\n5. BATCH LOOKUP")
print("-" * 70)

# Sequência de tokens: "gato cachorro peixe"
token_ids = torch.tensor([0, 1, 2])
sequence_embeddings = E[token_ids]

print(f"Token IDs: {token_ids.tolist()}")
print(f"Sequence embeddings shape: {sequence_embeddings.shape}")
print(f"Embeddings:\n{sequence_embeddings}")

# ========== 6. BATCH DE SEQUÊNCIAS ==========
print("\n6. BATCH DE SEQUÊNCIAS")
print("-" * 70)

# Duas sequências: [gato, cachorro] e [peixe, passaro]
batch_ids = torch.tensor([
    [0, 1],
    [2, 3]
])  # Shape [2, 2]

batch_embeddings = E[batch_ids]
print(f"Batch shape: {batch_ids.shape}")
print(f"Batch embeddings shape: {batch_embeddings.shape}")
print(f"Batch embeddings (first sequence):\n{batch_embeddings[0]}")

# ========== 7. APRENDIZADO SIMPLES ==========
print("\n7. APRENDIZADO SIMPLES (SIMULADO)")
print("-" * 70)

# Simular treinamento: aumentar similaridade gato-cachorro
E_train = E.clone().detach().requires_grad_(True)

# Loss: queremos que gato e cachorro sejam similares
target_similarity = 0.99

loss_values = []
for step in range(10):
    sim = torch.dot(E_train[0], E_train[1])
    loss = (sim - target_similarity) ** 2
    loss_values.append(loss.item())
    
    loss.backward()
    
    # Update (gradient descent)
    with torch.no_grad():
        E_train -= 0.1 * E_train.grad
        E_train.grad.zero_()

print(f"Loss inicial: {loss_values[0]:.6f}")
print(f"Loss final: {loss_values[-1]:.6f}")
print(f"Similaridade gato-cachorro (antes): {torch.dot(E[0], E[1]):.4f}")
print(f"Similaridade gato-cachorro (depois): {torch.dot(E_train[0], E_train[1]).detach():.4f}")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_embeddings.py
```

---

## Erros Comuns

### Erro 1: Confundir token ID com embedding

```python
# ❌ Errado
token_id = 5
x = token_id  # É um número, não um embedding

# ✓ Certo
embedding = E[token_id]  # Lookup na matriz
```

### Erro 2: Shapes errados em batch

```python
# ❌ Errado
token_ids = torch.tensor([[1, 2], [3, 4]])  # [2, 2]
embeddings = E[token_ids]  # ❌ ERRO se E é [V, d]

# ✓ Certo
embeddings = E[token_ids]  # [2, 2, d]
```

### Erro 3: Não fazer requires_grad

```python
# ❌ Errado - embeddings não são aprendidos
E = torch.randn(V, d)  # requires_grad=False por padrão

# ✓ Certo
E = torch.randn(V, d, requires_grad=True)
```

---

## Exercícios

### Exercício 6.1: Matriz de Embedding
Crie uma matriz de embedding [5, 3] representando 5 tokens e embedding dim 3.

### Exercício 6.2: Lookup
Acesse o embedding do token com ID 2.

### Exercício 6.3: Similaridade
Dados dois embeddings e1 = [1, 2, 3] e e2 = [2, 1, 0], compute o dot product.

### Exercício 6.4: Batch
Crie token_ids = [0, 2, 4]. Faça batch lookup em E.

### Exercício 6.5: Norma
Normalize um embedding para ter norma L2 = 1.

---

## Gabarito

### Exercício 6.1: Matriz de Embedding
```python
E = torch.randn(5, 3)  # [5, 3]
```

### Exercício 6.2: Lookup
```python
token_id = 2
emb = E[token_id]  # [3]
```

### Exercício 6.3: Similaridade
```python
e1 = torch.tensor([1.0, 2.0, 3.0])
e2 = torch.tensor([2.0, 1.0, 0.0])
dot_product = torch.dot(e1, e2)  # 1*2 + 2*1 + 3*0 = 4
```

### Exercício 6.4: Batch
```python
token_ids = torch.tensor([0, 2, 4])
batch_emb = E[token_ids]  # [3, 3]
```

### Exercício 6.5: Norma
```python
e = torch.tensor([3.0, 4.0])
e_normalized = e / torch.norm(e)  # [0.6, 0.8]
```

---

## Desafios Avançados (Opcionais)

### Fixação 6.1: Distância Euclidiana
Dois embeddings e1, e2. Compute distância Euclidiana.

$$d(e1, e2) = \|e1 - e2\|_2$$

### Fixação 6.2: Matriz de Similaridade
Crie uma matriz [N, N] de similaridades entre N embeddings.

Dica: `torch.mm(E, E.T)` computa todos dot products.

### Fixação 6.3: Embedding Mais Parecido
Dado um embedding de "gato" e uma matriz E de todos embeddings, encontre qual token é mais similar (argmax do dot product).

### Fixação 6.4: Normalizar Embeddings
L2-normalize todos embeddings em E:

```python
E_norm = E / E.norm(dim=1, keepdim=True)
```

### Fixação 6.5: Projeção no Espaço de Embedding
Dado um embedding arbitrário q (query), projete-o no espaço de embedding mais próximo (encontre embedding mais similar).

---

## Resumo

- **Embedding**: Representação densa de tokens, aprendível
- **Lookup**: Acessar matriz E por token ID
- **Similaridade**: Dot product reflete significado
- **Treinamento**: Gradientes atualizam valores em E
- **Batch**: Indexação em batch é eficiente

No próximo capítulo: **implementamos** embeddings em PyTorch.

---

**Próximo**: [Capítulo 07: Implementação de Embeddings](07_embedding_implementacao.md)
