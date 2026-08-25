# Capítulo 16: Self-Attention Completo

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Implementar Self-Attention completo manualmente
2. Entender fluxo de dados através de atenção
3. Computar gradientes end-to-end
4. Usar nn.MultiheadAttention quando apropriado
5. Debugar self-attention em modelos reais

---

## Por Que Isso Importa

**Self-Attention** é o Capítulo 10 (Atenção) mas aplicado a si mesma: entrada é simultaneamente Q, K, V.

Resumo rápido:
1. Entrada X: [batch, seq_len, d_model]
2. Projetar para Q, K, V usando camadas lineares
3. Computar atenção: scores = softmax(Q @ K.T / sqrt(d)) @ V
4. Saída: [batch, seq_len, d_model]

---

## Implementação Manual

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionManual:
    """Self-Attention manual"""
    
    def __init__(self, d_model):
        self.d_model = d_model
        
        # Projeções
        self.W_Q = torch.randn(d_model, d_model) * 0.01
        self.W_K = torch.randn(d_model, d_model) * 0.01
        self.W_V = torch.randn(d_model, d_model) * 0.01
        
        # Habilitar gradientes
        self.W_Q.requires_grad = True
        self.W_K.requires_grad = True
        self.W_V.requires_grad = True
    
    def forward(self, X):
        # X: [batch, seq_len, d_model]
        batch, seq_len, d_model = X.shape
        
        # Projetar
        Q = X @ self.W_Q  # [batch, seq_len, d_model]
        K = X @ self.W_K  # [batch, seq_len, d_model]
        V = X @ self.W_V  # [batch, seq_len, d_model]
        
        # Scores
        scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)
        # [batch, seq_len, d_model] @ [batch, d_model, seq_len]
        # = [batch, seq_len, seq_len]
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        # [batch, seq_len, seq_len]
        
        # Output
        output = attn_weights @ V
        # [batch, seq_len, seq_len] @ [batch, seq_len, d_model]
        # = [batch, seq_len, d_model]
        
        return output
    
    def parameters(self):
        return [self.W_Q, self.W_K, self.W_V]

# Uso
attn = SelfAttentionManual(d_model=64)
X = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
output = attn.forward(X)
print(output.shape)  # [2, 10, 64]
```

---

## Com nn.Linear

Versão mais pythônica:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
    
    def forward(self, X):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        
        scores = Q @ K.transpose(-2, -1) / (self.d_model ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        output = attn_weights @ V
        return output
```

---

## 🧪 Experimento: Self-Attention Completo

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("EXPERIMENTO: Self-Attention Completo")
print("=" * 70)

# ========== CONFIGURAÇÃO ==========
batch_size = 2
seq_len = 4
d_model = 8

print(f"\nConfiguraçao:")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")
print(f"  d_model: {d_model}")

# ========== DADOS ==========
print("\n1. ENTRADA (X)")
print("-" * 70)

X = torch.randn(batch_size, seq_len, d_model)
print(f"X shape: {X.shape}")

# ========== PROJEÇÕES ==========
print("\n2. PROJEÇÕES (Q, K, V)")
print("-" * 70)

W_Q = nn.Linear(d_model, d_model)
W_K = nn.Linear(d_model, d_model)
W_V = nn.Linear(d_model, d_model)

Q = W_Q(X)  # [2, 4, 8]
K = W_K(X)  # [2, 4, 8]
V = W_V(X)  # [2, 4, 8]

print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

# ========== SCORES ==========
print("\n3. ATTENTION SCORES")
print("-" * 70)

# Batch matmul: [2, 4, 8] @ [2, 8, 4] = [2, 4, 4]
scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)
print(f"scores shape: {scores.shape}")
print(f"Exemplo (primeiro exemplo, primeira posição):")
print(f"scores[0, 0, :] = {scores[0, 0, :]}")

# ========== ATTENTION WEIGHTS ==========
print("\n4. SOFTMAX")
print("-" * 70)

attn_weights = F.softmax(scores, dim=-1)
print(f"attn_weights shape: {attn_weights.shape}")
print(f"Exemplo:")
print(f"attn_weights[0, 0, :] = {attn_weights[0, 0, :]}")
print(f"Sum (deve ser 1): {attn_weights[0, 0, :].sum()}")

# ========== OUTPUT ==========
print("\n5. OUTPUT")
print("-" * 70)

output = attn_weights @ V
print(f"output shape: {output.shape}")
print(f"Esperado: [2, 4, 8]")

# ========== VERIFICAÇÃO ==========
print("\n6. VERIFICAÇÃO DE SHAPES")
print("-" * 70)

print(f"Input:  {X.shape}")
print(f"Output: {output.shape}")
print(f"Shapes iguais? {X.shape == output.shape}")
print("(Esperado: True, self-attention preserva shape)")

# ========== GRADIENTES ==========
print("\n7. GRADIENTES")
print("-" * 70)

loss = output.sum()
loss.backward()

print(f"W_Q.weight.grad.shape: {W_Q.weight.grad.shape}")
print(f"W_Q.weight.grad não-zero? {(W_Q.weight.grad != 0).any()}")
print("(Esperado: True, backprop funcionou)")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

---

## Fluxo de Dados

```
Input X [batch, seq_len, d_model]
    |
    ├─→ W_Q → Q [batch, seq_len, d_model]
    ├─→ W_K → K [batch, seq_len, d_model]
    └─→ W_V → V [batch, seq_len, d_model]
    
Q @ K.T / sqrt(d) → scores [batch, seq_len, seq_len]
    |
    ├─→ softmax → attn_weights [batch, seq_len, seq_len]
    
attn_weights @ V → output [batch, seq_len, d_model]
```

---

## Exercícios

### Exercício 16.1: Forward Manual
Implemente SelfAttention e faça forward com X [1, 3, 4].

### Exercício 16.2: Verificar Output Shape
Input [32, 10, 64]. Self-Attention. Output shape?

### Exercício 16.3: Gradientes
Faça backward. Verifique que W_Q, W_K, W_V têm gradientes.

### Exercício 16.4: Matriz de Atenção
Imprima attn_weights para uma posição. Quais outras posições ela "vê"?

### Exercício 16.5: Adicionar Bias
Modifique SelfAttention para ter W_Q, W_K, W_V com bias.

---

## Gabarito

### Exercício 16.1-16.5
Veja estrutura de código acima.

---

## Exercícios de Fixação (Opcionais)

### Fixação 16.1: Causal Masking Preview
Adicione máscara triangular após scores para que posição i não veja j > i.

### Fixação 16.2: Dropout
Adicione dropout em attn_weights para regularização.

### Fixação 16.3: Diferente d_k para Q,K,V
Use d_k=32 para Q/K, d_v=8 para V. Como muda?

---

## Resumo

- **Self-Attention**: Atenção onde entrada é Q, K, V
- **Shapes**: Preserva [batch, seq_len, d_model]
- **Gradientes**: Backprop automático através de attn
- **Prototipagem**: Começar manual, depois nn.MultiheadAttention

Próximo: **Multi-Head Attention** — múltiplas "cabeças" em paralelo.

---

**Próximo**: [Capítulo 17: Multi-Head Attention](17_multihead_attention.md)
