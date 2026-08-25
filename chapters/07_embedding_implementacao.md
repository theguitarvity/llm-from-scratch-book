# Capítulo 07: Implementação de Embeddings

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Implementar embeddings manualmente com PyTorch
2. Entender `nn.Embedding` e quando usá-lo
3. Inicializar embeddings corretamente
4. Debugar problemas comuns em embeddings
5. Usar embeddings em um fluxo de forward pass

---

## Por Que Isso Importa

No capítulo anterior, vimos embeddings conceitualmente. Agora vamos **implementar de verdade**.

Você já sabe que embedding é apenas um lookup em uma matriz. Vamos:
1. Criar a matriz
2. Fazer indexação
3. Entender gradientes fluindo de volta
4. Usar nn.Embedding para abstrair

---

## 🔨 Implementação Manual

### Versão Mais Simples

```python
import torch

class EmbeddingManual:
    """Embedding manual sem autograd"""
    
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Inicializar matriz (randômica, pequena)
        self.weight = torch.randn(vocab_size, embedding_dim) * 0.01
    
    def forward(self, token_ids):
        """token_ids: [batch, seq_len] ou qualquer shape"""
        return self.weight[token_ids]  # Lookup simples

# Uso
emb = EmbeddingManual(vocab_size=100, embedding_dim=64)
token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
output = emb.forward(token_ids)
print(output.shape)  # [2, 3, 64]
```

### Com Gradientes

Para que a rede neural aprenda, precisamos de `requires_grad=True`:

```python
class Embedding:
    """Embedding com suporte a gradientes"""
    
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Inicializar com requires_grad=True
        self.weight = torch.randn(vocab_size, embedding_dim) * 0.01
        self.weight.requires_grad = True
    
    def forward(self, token_ids):
        return self.weight[token_ids]
    
    def parameters(self):
        """Retorna parâmetros treináveis"""
        return [self.weight]

# Uso com treinamento
emb = Embedding(vocab_size=10, embedding_dim=4)

token_ids = torch.tensor([0, 1, 2])  # 3 tokens
embeddings = emb.forward(token_ids)  # [3, 4]

# Simular loss
loss = embeddings.sum()  # Dummy loss
loss.backward()

print(f"Gradientes: {emb.weight.grad.shape}")  # [10, 4]
print(f"Gradientes não-zero: {(emb.weight.grad != 0).sum()}")  # Apenas índices acessados
```

---

## 🎯 PyTorch's nn.Embedding

PyTorch fornece `nn.Embedding` que faz tudo isso:

```python
import torch.nn as nn

# Criar embedding
embedding = nn.Embedding(vocab_size=100, embedding_dim=64)

# Forward
token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
output = embedding(token_ids)
print(output.shape)  # [2, 3, 64]

# Acessar pesos
print(embedding.weight.shape)  # [100, 64]

# Todos os parâmetros
for param in embedding.parameters():
    print(param.shape)  # [100, 64]
```

### Inicialização

`nn.Embedding` inicializa os pesos com `N(0, 1)` por padrão. Você pode personalizar:

```python
embedding = nn.Embedding(100, 64)

# Inicialização customizada
torch.nn.init.uniform_(embedding.weight, -0.05, 0.05)

# Ou normal
torch.nn.init.normal_(embedding.weight, 0, 0.01)

# Ou fixar um embedding (padding token)
embedding.weight.data[0] = 0  # Token 0 = padding, embeddings zeros
```

---

## Experimento 1: Embedding Manual vs nn.Embedding

Crie `experimento_embedding_manual.py`:

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Embedding Manual vs nn.Embedding")
print("=" * 70)

# ========== CONFIGURAÇÃO ==========
vocab_size = 5
embedding_dim = 3
batch_size = 2
seq_len = 4

print(f"\nConfiguraçao:")
print(f"  vocab_size: {vocab_size}")
print(f"  embedding_dim: {embedding_dim}")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")

# ========== MANUAL ==========
print("\n1. EMBEDDING MANUAL")
print("-" * 70)

class SimpleEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        self.weight = torch.randn(vocab_size, embedding_dim) * 0.01
        self.weight.requires_grad = True
    
    def forward(self, token_ids):
        return self.weight[token_ids]

emb_manual = SimpleEmbedding(vocab_size, embedding_dim)
token_ids = torch.tensor([[0, 1, 2, 3],
                          [1, 2, 3, 4]])  # [2, 4]

embeddings_manual = emb_manual.forward(token_ids)
print(f"Token IDs shape: {token_ids.shape}")
print(f"Embeddings shape: {embeddings_manual.shape}")
print(f"Embeddings:\n{embeddings_manual}")

# ========== nn.Embedding ==========
print("\n2. nn.Embedding")
print("-" * 70)

emb_pytorch = nn.Embedding(vocab_size, embedding_dim)
embeddings_pytorch = emb_pytorch(token_ids)

print(f"Embeddings shape: {embeddings_pytorch.shape}")
print(f"Weight shape: {emb_pytorch.weight.shape}")

# ========== GRADIENTES ==========
print("\n3. GRADIENTES")
print("-" * 70)

# Computar loss simples
loss = embeddings_pytorch.sum()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Weight gradientes shape: {emb_pytorch.weight.grad.shape}")
print(f"Weight gradientes (apenas tokens acessados não-zero):\n{emb_pytorch.weight.grad}")

# ========== PADDING ==========
print("\n4. PADDING TOKEN")
print("-" * 70)

# Padding token (idx 0) não deve ter gradientes significativos
emb_with_padding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

# Token 0 começa com zeros
print(f"Embedding de padding token (idx 0):\n{emb_with_padding.weight[0]}")

# Dados com padding
token_ids_with_padding = torch.tensor([[0, 1, 2, 0]])
embeddings_padded = emb_with_padding(token_ids_with_padding)
print(f"Embeddings (com padding token):\n{embeddings_padded}")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_embedding_manual.py
```

---

## Experimento 2: Matriz de Similaridade

Crie `experimento_similaridade.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("EXPERIMENTO: Matriz de Similaridade com Embeddings")
print("=" * 70)

# ========== DADOS ==========
vocab_size = 6
embedding_dim = 4

vocab = ["gato", "cachorro", "peixe", "passaro", "leão", "tubarão"]

emb = nn.Embedding(vocab_size, embedding_dim)

# ========== 1. EMBEDDINGS INDIVIDUAIS ==========
print("\n1. EMBEDDINGS")
print("-" * 70)

for idx, word in enumerate(vocab):
    e = emb.weight[idx]
    print(f"{word:10s}: {e.detach().tolist()}")

# ========== 2. MATRIZ DE SIMILARIDADE (Dot Product) ==========
print("\n2. MATRIZ DE SIMILARIDADE (Dot Product)")
print("-" * 70)

# E @ E^T dá todas os dot products
similarity_matrix = emb.weight @ emb.weight.T
print("Matriz de similaridade:\n", similarity_matrix.detach())

# ========== 3. MATRIZ DE SIMILARIDADE (Cosseno) ==========
print("\n3. MATRIZ DE SIMILARIDADE (Cosseno)")
print("-" * 70)

# Normalizar embeddings
E_norm = F.normalize(emb.weight, p=2, dim=1)  # L2 normalize
cosine_similarity = E_norm @ E_norm.T
print("Matriz de similaridade (cosseno):\n", cosine_similarity.detach())

# ========== 4. ENCONTRAR VIZINHO MAIS PRÓXIMO ==========
print("\n4. TOP-3 VIZINHOS MAIS PRÓXIMOS")
print("-" * 70)

for idx, word in enumerate(vocab):
    similarities = cosine_similarity[idx]
    top_k = torch.topk(similarities, k=4)  # 4 melhores (incluindo ele mesmo)
    
    print(f"\n{word}:")
    for score, neighbor_idx in zip(top_k.values, top_k.indices):
        print(f"  {vocab[neighbor_idx.item()]:10s} {score.item():.4f}")

# ========== 5. DISTÂNCIA EUCLIDIANA ==========
print("\n5. MATRIZ DE DISTÂNCIA EUCLIDIANA")
print("-" * 70)

# ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
norms = (emb.weight ** 2).sum(dim=1, keepdim=True)  # [V, 1]
distances = norms + norms.T - 2 * (emb.weight @ emb.weight.T)
distances = torch.sqrt(torch.clamp(distances, min=1e-8))  # Evitar sqrt negativo

print("Matriz de distâncias:\n", distances.detach())

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_similaridade.py
```

---

## 📋 Inicializações Comuns

```python
import torch
import torch.nn as nn
from torch.nn import init

embedding = nn.Embedding(vocab_size=1000, embedding_dim=64)

# 1. Normal (padrão de PyTorch)
init.normal_(embedding.weight, mean=0, std=1)

# 2. Uniforme pequeno (recomendado para embeddings)
init.uniform_(embedding.weight, -0.05, 0.05)

# 3. Xavier (mantém variância através de camadas)
init.xavier_uniform_(embedding.weight)
init.xavier_normal_(embedding.weight)

# 4. Kaiming (para ReLU)
init.kaiming_uniform_(embedding.weight)

# 5. Fixar padding token
embedding.weight.data[padding_idx] = 0
```

**Recomendação prática**:
```python
# Para embeddings de linguagem
init.uniform_(embedding.weight, -0.05, 0.05)
embedding.weight.data[padding_idx] = 0
```

---

## Entendendo Gradientes em Embeddings

```python
# Dados de entrada
token_ids = torch.tensor([0, 1, 2, 1])  # Token 1 aparece 2x

# Embedding
emb = nn.Embedding(3, 4)
embeddings = emb(token_ids)  # [4, 4]

# Loss simples
loss = embeddings.sum()
loss.backward()

# Observe:
# emb.weight.grad[0] ≠ 0 (uma linha)
# emb.weight.grad[1] ≠ 0 (duas vezes, apareceu 2x)
# emb.weight.grad[2] ≠ 0 (uma linha)
# Outros ≈ 0

# Gradientes são acumulados por quantas vezes apareceram!
print(emb.weight.grad)
```

Isso é importante: **tokens frequentes recebem mais atualizações**, tokens raros menos.

---

## Erros Comuns

### Erro 1: Token ID fora de range

```python
# ❌ Errado
emb = nn.Embedding(100, 64)  # vocab_size=100 (0-99)
token_ids = torch.tensor([99, 100, 101])  # ERRO: 100 está fora

# ✓ Certo
token_ids = torch.tensor([99, 50, 10])  # Todos < 100
```

### Erro 2: Esquecer padding_idx

```python
# ❌ Errado - padding token tem gradientes
emb = nn.Embedding(vocab_size, embedding_dim)

# ✓ Certo - padding não é treinado
emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

### Erro 3: Inicialização errada

```python
# ❌ Errado - embeddings muito grandes causam exploding gradients
emb = nn.Embedding(10000, 768)

# ✓ Certo
init.normal_(emb.weight, mean=0, std=0.02)
```

---

## Exercícios

### Exercício 7.1: Embedding Manual
Implemente `EmbeddingManual` sem gradientes. Faça lookup de token 3 em vocab_size=10.

### Exercício 7.2: Com Gradientes
Implemente embedding com requires_grad=True. Compute loss, backward, verifique gradientes.

### Exercício 7.3: nn.Embedding
Crie `nn.Embedding(100, 64)`. Faça forward com token_ids [[1, 2], [3, 4]]. Qual é o shape?

### Exercício 7.4: Similaridade
Dois tokens em um embedding. Compute similaridade por dot product.

### Exercício 7.5: Inicialização
Crie embedding e inicialize com `uniform(-0.05, 0.05)`. Verifique que todos valores estão naquele range.

---

## Gabarito

### Exercício 7.1: Embedding Manual
```python
class EmbeddingManual:
    def __init__(self, vocab_size, embedding_dim):
        self.weight = torch.randn(vocab_size, embedding_dim)
    
    def forward(self, token_ids):
        return self.weight[token_ids]

emb = EmbeddingManual(10, 4)
e = emb.forward(torch.tensor(3))  # [4]
```

### Exercício 7.2: Com Gradientes
```python
emb = nn.Embedding(10, 4)
token_ids = torch.tensor([0, 1, 2])
embeddings = emb(token_ids)
loss = embeddings.sum()
loss.backward()
print(emb.weight.grad.shape)  # [10, 4]
```

### Exercício 7.3: nn.Embedding
```python
emb = nn.Embedding(100, 64)
token_ids = torch.tensor([[1, 2], [3, 4]])
output = emb(token_ids)
print(output.shape)  # [2, 2, 64]
```

### Exercício 7.4: Similaridade
```python
emb = nn.Embedding(100, 64)
e0 = emb.weight[0]
e1 = emb.weight[1]
sim = torch.dot(e0, e1)
```

### Exercício 7.5: Inicialização
```python
emb = nn.Embedding(100, 64)
torch.nn.init.uniform_(emb.weight, -0.05, 0.05)
print(emb.weight.max().item() <= 0.05)  # True
print(emb.weight.min().item() >= -0.05)  # True
```

---

## Desafios Avançados (Opcionais)

### Fixação 7.1: Embedding com Padding
Crie embedding com padding_idx=0. Verifique que peso[0] é sempre 0 mesmo após atualização.

### Fixação 7.2: Freeze Embedding
Congi embedding com requires_grad=False (frozen). Verifique que backward não atualiza.

### Fixação 7.3: Compartilhar Embeddings
Crie embedding matrix E. Use a mesma para input e output de um modelo. Qual é a vantagem?

### Fixação 7.4: Embedding Dinâmico
Crie embeddings com vocab_size=100. Depois expanda para 200. Como lidar sem recriar?

Dica: Você precisa expandir a matriz de pesos.

### Fixação 7.5: Embedding Regularização
Compute a norma L2 de todos embeddings. Use como regularização na loss.

---

## Resumo

- **Implementação manual**: Lookup + requires_grad
- **nn.Embedding**: Abstração de lookup treinável
- **Gradientes**: Fluem para linha acessada em weight
- **Inicialização**: Uniforme pequeno ou normal
- **Padding**: Use padding_idx para não treinar token especial

Próximo: **Projeções lineares** — como transformar embeddings.

---

**Próximo**: [Capítulo 08: Projeções Lineares](08_projecoes_lineares.md)
