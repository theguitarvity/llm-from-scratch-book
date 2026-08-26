# Capítulo 17: Multi-Head Attention

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender por que uma única cabeça de atenção limita a expressividade do modelo
2. Dividir d_model em múltiplas cabeças paralelas, cada uma com sua própria d_k
3. Implementar projeções Q, K, V por cabeça e computar atenção em paralelo (batched)
4. Rastrear shapes de tensores em cada etapa da transformação (reshape, transpose, concat)
5. Implementar Multi-Head Attention do zero e comparar com `nn.MultiheadAttention`

---

## Por Que Isso Importa

Nos capítulos 11 a 16 você construiu self-attention completo: Q, K, V, scores, scaling, softmax, weighted sum e causal masking. Funciona. Mas há um problema sutil: com uma única cabeça de atenção, o modelo só consegue aprender **um único padrão de relacionamento entre tokens por vez**.

Pense na frase: "O relatório que o gerente que contratamos ontem revisou estava incompleto." Para entender essa frase corretamente, o modelo precisa simultaneamente resolver múltiplas relações: qual é o sujeito de "revisou" (o gerente, não o relatório), a que "que contratamos ontem" se refere (o gerente), e a que "estava incompleto" se conecta (o relatório, lá no início). São três relações sintático-semânticas diferentes, cada uma exigindo um padrão de atenção diferente. Uma única matriz de atenção — um único conjunto de pesos softmax por posição — precisa fazer *média* entre essas necessidades conflitantes. O resultado é uma solução de compromisso: nenhuma relação é capturada tão bem quanto poderia.

É exatamente o problema que aparece quando você debuga um modelo com atenção de cabeça única e nota que a matriz de atenção parece "borrada" — pesos espalhados sem foco claro em nenhum padrão específico. O modelo está tentando usar um único mecanismo para fazer o trabalho de vários especialistas.

A solução do paper "Attention Is All You Need" é elegante: em vez de computar atenção uma vez com toda a dimensão `d_model`, divida `d_model` em `num_heads` fatias menores e compute atenção **em paralelo** dentro de cada fatia, com seus próprios W_Q, W_K, W_V independentes. Cada cabeça pode então se especializar — uma cabeça pode aprender a rastrear concordância sujeito-verbo, outra pode aprender a resolver referências de pronomes, outra pode focar em posições adjacentes. No final, você concatena as saídas de todas as cabeças e projeta de volta para `d_model` com uma matriz aprendida $W_O$. É como trocar um único generalista por um comitê de especialistas que depois combinam suas opiniões.

---

## Dividindo d_model em Cabeças

### A Ideia Central

Em vez de:

$$\text{Attention}(Q, K, V) \text{ com } Q, K, V \in \mathbb{R}^{[\text{seq\_len}, d_{model}]}$$

fazemos:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

onde cada cabeça é:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

A restrição de dimensão é:

$$d_k = d_v = \frac{d_{model}}{\text{num\_heads}}$$

Ou seja, se `d_model = 512` e `num_heads = 8`, cada cabeça trabalha com `d_k = 64`. Note que o **custo computacional total** de multi-head attention é aproximadamente o mesmo de uma única cabeça com `d_model` completo — você não está adicionando parâmetros de forma descontrolada, está *reparticionando* a mesma capacidade em subespaços paralelos.

### Shapes, Passo a Passo

Vamos rastrear os tensores com valores concretos: `batch=2`, `seq_len=5`, `d_model=8`, `num_heads=2` (logo `d_k=4`).

**Passo 1 — Entrada e projeções lineares completas**

```
X: [batch=2, seq_len=5, d_model=8]

Q = X @ W_Q   # W_Q: [8, 8]  ->  Q: [2, 5, 8]
K = X @ W_K   # W_K: [8, 8]  ->  K: [2, 5, 8]
V = X @ W_V   # W_V: [8, 8]  ->  V: [2, 5, 8]
```

Aqui já vale uma observação importante: na prática, **não criamos matrizes de projeção separadas por cabeça**. Criamos uma única `W_Q` de tamanho `[d_model, d_model]` (via `nn.Linear(d_model, d_model)`), projetamos tudo de uma vez, e *depois* dividimos o resultado em cabeças. Isso é matematicamente equivalente a ter `W_Q^1, ..., W_Q^h` separados (porque a divisão é apenas um reshape das colunas de saída), mas é muito mais eficiente computacionalmente — uma única multiplicação de matriz grande em vez de `h` multiplicações pequenas.

**Passo 2 — Reshape para separar as cabeças**

```
Q: [2, 5, 8] -> view(2, 5, num_heads=2, d_k=4) -> [2, 5, 2, 4]
             -> transpose(1, 2)                -> [2, 2, 5, 4]
             # [batch, num_heads, seq_len, d_k]
```

O `.transpose(1, 2)` é crucial: ele move `num_heads` para logo depois de `batch`, formando o shape `[batch, num_heads, seq_len, d_k]`. Isso permite tratar `(batch, num_heads)` como dimensões "empilhadas" e computar atenção para todas as cabeças e todos os exemplos do batch em uma única operação matricial batched — sem loop Python.

O mesmo se aplica a K e V:

```
K: [2, 5, 8] -> [2, 5, 2, 4] -> [2, 2, 5, 4]
V: [2, 5, 8] -> [2, 5, 2, 4] -> [2, 2, 5, 4]
```

**Passo 3 — Atenção em paralelo (batched)**

```
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
       # [2, 2, 5, 4] @ [2, 2, 4, 5] -> [2, 2, 5, 5]
       # [batch, num_heads, seq_len, seq_len]

attn_weights = softmax(scores, dim=-1)   # [2, 2, 5, 5]

head_outputs = attn_weights @ V
             # [2, 2, 5, 5] @ [2, 2, 5, 4] -> [2, 2, 5, 4]
             # [batch, num_heads, seq_len, d_k]
```

Note que `torch.matmul` faz *batched matmul* automaticamente sobre as dimensões que não são as duas últimas — é por isso que empacotar `num_heads` junto com `batch` funciona: o PyTorch trata `[batch, num_heads]` como um "batch composto" e aplica a multiplicação `[seq_len, d_k] @ [d_k, seq_len]` de forma independente para cada combinação de (exemplo, cabeça).

**Passo 4 — Concatenar cabeças e projetar**

```
head_outputs: [2, 2, 5, 4] -> transpose(1, 2) -> [2, 5, 2, 4]
                             -> reshape(2, 5, 8) -> [2, 5, 8]
                             # concatenação implícita das cabeças na última dimensão

output = concatenated @ W_O   # W_O: [8, 8]  ->  output: [2, 5, 8]
```

O `.transpose(1, 2)` seguido de `.reshape` (ou `.contiguous().view`) desfaz a separação de cabeças: volta ao layout `[batch, seq_len, num_heads, d_k]` e depois funde as duas últimas dimensões em `d_model`. Isso é literalmente a operação de "concatenar as saídas das cabeças ao longo da dimensão de features" — só que feita via reshape em vez de `torch.cat` explícito, porque os dados já estão contíguos na ordem certa.

### Exemplo Numérico Manual

Vamos fazer um exemplo minúsculo à mão: `seq_len=2`, `d_model=4`, `num_heads=2` (`d_k=2`).

```
X = [
    [1.0, 0.0, 1.0, 0.0],   # token A
    [0.0, 1.0, 0.0, 1.0]    # token B
]
```

Suponha que, após projeção linear (usando W_Q = W_K = W_V = I por simplicidade), temos Q = K = V = X.

Dividindo em 2 cabeças de d_k=2 cada:

```
Cabeça 1 usa colunas [0,1]:
Q1 = K1 = V1 = [[1.0, 0.0], [0.0, 1.0]]

Cabeça 2 usa colunas [2,3]:
Q2 = K2 = V2 = [[1.0, 0.0], [0.0, 1.0]]
```

Para a cabeça 1, scores = Q1 @ K1^T:

```
scores[0,0] = 1*1 + 0*0 = 1.0
scores[0,1] = 1*0 + 0*1 = 0.0
scores[1,0] = 0*1 + 1*0 = 0.0
scores[1,1] = 0*0 + 1*1 = 1.0

scores = [[1.0, 0.0], [0.0, 1.0]]
```

Escalando por $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

```
scores_scaled = [[0.707, 0.0], [0.0, 0.707]]
```

Softmax da linha 0: `[0.707, 0.0]` -> `exp = [2.028, 1.0]`, soma = 3.028
-> `attn[0] = [0.670, 0.330]`

Por simetria, `attn[1] = [0.330, 0.670]`.

Output da cabeça 1:

```
out1[0] = 0.670 * [1.0, 0.0] + 0.330 * [0.0, 1.0] = [0.670, 0.330]
out1[1] = 0.330 * [1.0, 0.0] + 0.670 * [0.0, 1.0] = [0.330, 0.670]
```

Como Q2=K2=V2 têm a mesma estrutura, a cabeça 2 produz o mesmo padrão numérico:

```
out2 = [[0.670, 0.330], [0.330, 0.670]]
```

Concatenando as duas cabeças (cabeça 1 nas colunas 0-1, cabeça 2 nas colunas 2-3):

```
concat = [
    [0.670, 0.330, 0.670, 0.330],
    [0.330, 0.670, 0.330, 0.670]
]
```

E finalmente `output = concat @ W_O` (com `W_O = I` neste exemplo, `output = concat`). Em um caso real, W_O misturaria as informações das duas cabeças de volta em um único espaço de representação compartilhado — é essa mistura que permite ao modelo combinar o que cada cabeça "aprendeu" separadamente.

---

## Experimento: Multi-Head Attention do Zero

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("EXPERIMENTO: Multi-Head Attention Implementado do Zero")
print("=" * 70)

torch.manual_seed(42)

# ========== CONFIGURAÇÃO ==========
batch_size = 2
seq_len = 5
d_model = 8
num_heads = 2
d_k = d_model // num_heads

print(f"\nConfiguração:")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_k (d_model / num_heads): {d_k}")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projeções lineares completas (não uma por cabeça!)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        # x: [batch, seq_len, d_model]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # x: [batch, seq_len, num_heads, d_k]
        return x.transpose(1, 2)
        # x: [batch, num_heads, seq_len, d_k]

    def combine_heads(self, x, batch_size):
        # x: [batch, num_heads, seq_len, d_k]
        x = x.transpose(1, 2).contiguous()
        # x: [batch, seq_len, num_heads, d_k]
        return x.view(batch_size, -1, self.d_model)
        # x: [batch, seq_len, d_model]

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # 1. Projeções completas
        Q = self.W_Q(x)  # [batch, seq_len, d_model]
        K = self.W_K(x)
        V = self.W_V(x)

        # 2. Dividir em cabeças
        Q = self.split_heads(Q, batch_size)  # [batch, num_heads, seq_len, d_k]
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 3. Scores escalados (batched sobre batch e num_heads)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        # [batch, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        # [batch, num_heads, seq_len, seq_len]

        head_outputs = attn_weights @ V
        # [batch, num_heads, seq_len, d_k]

        # 4. Combinar cabeças e projetar
        combined = self.combine_heads(head_outputs, batch_size)
        # [batch, seq_len, d_model]

        output = self.W_O(combined)
        # [batch, seq_len, d_model]

        return output, attn_weights


# ========== ENTRADA ==========
print("\n1. ENTRADA")
print("-" * 70)

X = torch.randn(batch_size, seq_len, d_model)
print(f"X shape: {X.shape}")

# ========== FORWARD ==========
print("\n2. FORWARD PASS")
print("-" * 70)

mha = MultiHeadAttention(d_model, num_heads)
output, attn_weights = mha(X)

print(f"output shape: {output.shape}")
print(f"attn_weights shape: {attn_weights.shape}")
print(f"  -> [batch, num_heads, seq_len, seq_len]")

# ========== VERIFICAÇÃO DE SOMA DE PESOS ==========
print("\n3. VERIFICAÇÃO: pesos de atenção somam 1 por linha")
print("-" * 70)

soma_pesos = attn_weights.sum(dim=-1)
print(f"soma por (batch, cabeça, posição), shape: {soma_pesos.shape}")
print(f"valores (devem ser ~1.0): {soma_pesos[0, 0]}")

# ========== INSPECIONANDO CABEÇAS INDIVIDUAIS ==========
print("\n4. PADRÕES DE ATENÇÃO POR CABEÇA (primeiro exemplo do batch)")
print("-" * 70)

for h in range(num_heads):
    print(f"\nCabeça {h}:")
    print(attn_weights[0, h])
    max_focos = attn_weights[0, h].argmax(dim=-1)
    print(f"  Posição de maior foco por token: {max_focos.tolist()}")

# ========== COMPARANDO COM nn.MultiheadAttention ==========
print("\n5. COMPARAÇÃO COM nn.MultiheadAttention (PyTorch nativo)")
print("-" * 70)

mha_pytorch = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)

out_pt, attn_pt = mha_pytorch(X, X, X)
print(f"Saída PyTorch nativo shape: {out_pt.shape}")
print(f"Atenção PyTorch nativo shape: {attn_pt.shape}")
print("  -> nn.MultiheadAttention retorna atenção MÉDIA entre cabeças por padrão")
print("  -> nossa implementação retorna atenção por cabeça, mais informativo para debugging")

# ========== CONTAGEM DE PARÂMETROS ==========
print("\n6. CONTAGEM DE PARÂMETROS")
print("-" * 70)

n_params_custom = sum(p.numel() for p in mha.parameters())
n_params_pytorch = sum(p.numel() for p in mha_pytorch.parameters())
print(f"Nossa implementação: {n_params_custom} parâmetros")
print(f"nn.MultiheadAttention: {n_params_pytorch} parâmetros")
print("  (diferença normal: nn.MultiheadAttention usa in_proj combinado e pode ter bias)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Experimento 2: Especialização de Cabeças Durante Treinamento

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("EXPERIMENTO: Cabeças Aprendem Padrões Diferentes")
print("=" * 70)

torch.manual_seed(42)

d_model = 16
num_heads = 4
seq_len = 6
batch_size = 1

class SimpleMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out), attn


model = SimpleMHA(d_model, num_heads)
X = torch.randn(batch_size, seq_len, d_model, requires_grad=False)

# Tarefa fictícia: aprender a "somar" tokens vizinhos (posição i deve
# atender fortemente à posição i-1 depois de treinar)
target = torch.randn(batch_size, seq_len, d_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n1. TREINANDO POR ALGUMAS ITERAÇÕES")
print("-" * 70)

for step in range(200):
    optimizer.zero_grad()
    out, attn = model(X)
    loss = F.mse_loss(out, target)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss = {loss.item():.4f}")

print("\n2. PADRÕES DE ATENÇÃO APÓS TREINAMENTO (por cabeça)")
print("-" * 70)

with torch.no_grad():
    _, attn = model(X)

for h in range(num_heads):
    print(f"\nCabeça {h} (arredondado a 2 casas):")
    print(attn[0, h].round(decimals=2))

print("\nObservação: cada cabeça converge para uma distribuição de pesos")
print("diferente das outras, mesmo partindo da mesma entrada X.")
print("Isso é a 'especialização' de cabeças em ação.")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Esquecer de dividir d_k, usando d_model inteiro em cada cabeça

```python
# ❌ Errado — cada cabeça usa d_model completo, explodindo parâmetros e
# custo computacional (e não é mais "multi-head", é "single-head repetido")
self.W_Q = nn.Linear(d_model, d_model * num_heads)

# ✓ Certo — a soma das dimensões das cabeças é igual a d_model
self.d_k = d_model // num_heads
self.W_Q = nn.Linear(d_model, d_model)  # projeta para d_model, dividido depois em heads
```

### Erro 2: Esquecer o `.contiguous()` antes do `.view()` após `transpose`

```python
# ❌ Errado — transpose() retorna uma view não-contígua na memória;
# .view() falha ou (pior) produz dados corrompidos silenciosamente em
# algumas versões/dispositivos
x = head_outputs.transpose(1, 2).view(batch_size, seq_len, d_model)
# RuntimeError: view size is not compatible with input tensor's size and stride

# ✓ Certo
x = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
# ou, de forma equivalente e mais robusta:
x = head_outputs.transpose(1, 2).reshape(batch_size, seq_len, d_model)
```

### Erro 3: Escalar por sqrt(d_model) em vez de sqrt(d_k)

```python
# ❌ Errado — usar a dimensão total em vez da dimensão por cabeça
scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)

# ✓ Certo — cada cabeça opera em d_k dimensões, e é essa a variância
# relevante para o argumento de scaling (ver Capítulo 13)
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
```

### Erro 4: Aplicar a máscara causal na dimensão errada do tensor 4D

```python
# ❌ Errado — mask com shape [seq_len, seq_len] quebra broadcasting
# contra scores [batch, num_heads, seq_len, seq_len] se não expandido corretamente
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float("-inf"))  # funciona por broadcast, mas é frágil

# ✓ Certo — seja explícito sobre as dimensões esperadas
mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
scores = scores.masked_fill(mask == 0, float("-inf"))
# broadcast explícito: [1,1,T,T] contra [batch,num_heads,T,T]
```

---

## Exercícios

### Exercício 17.1: Calcular d_k
Dado `d_model = 512` e `num_heads = 8`, qual é `d_k`? E se `num_heads = 16`? O que acontece com a capacidade de cada cabeça individual quando `num_heads` aumenta mantendo `d_model` fixo?

### Exercício 17.2: Shapes ao Longo do Pipeline
Para `batch=4, seq_len=10, d_model=64, num_heads=4`, escreva o shape do tensor após cada uma destas operações: projeção linear, `view` para separar cabeças, `transpose(1,2)`, cálculo de scores, softmax, `attn @ V`, `transpose` de volta, `view` para combinar, projeção final W_O.

### Exercício 17.3: Implementar split_heads e combine_heads
Escreva as duas funções `split_heads(x, batch_size, num_heads, d_k)` e `combine_heads(x, batch_size, d_model)` do zero, sem olhar o experimento acima.

### Exercício 17.4: Contar Parâmetros
Para `d_model=256, num_heads=8`, sem bias, quantos parâmetros tem uma camada de Multi-Head Attention completa (W_Q, W_K, W_V, W_O)? Compare com uma única "super-cabeça" de atenção com `d_model=256` completo — quantos parâmetros ela teria a mais ou a menos?

### Exercício 17.5: Cabeças Idênticas
Se você inicializar `W_Q, W_K, W_V` de forma que todas as cabeças recebam exatamente os mesmos pesos, o que acontece com a expressividade do modelo em comparação com usar `num_heads=1`? Justifique.

---

## Gabarito

### Exercício 17.1: Calcular d_k
```python
d_model = 512
for num_heads in [8, 16]:
    d_k = d_model // num_heads
    print(f"num_heads={num_heads} -> d_k={d_k}")
# num_heads=8  -> d_k=64
# num_heads=16 -> d_k=32
```
Quando `num_heads` aumenta com `d_model` fixo, cada cabeça individual recebe *menos* dimensões para representar suas relações — mais cabeças significa mais "pontos de vista" simultâneos, mas cada um mais restrito. É um trade-off entre diversidade de padrões e capacidade por padrão.

### Exercício 17.2: Shapes ao Longo do Pipeline
```python
import torch

batch, seq_len, d_model, num_heads = 4, 10, 64, 4
d_k = d_model // num_heads

x = torch.randn(batch, seq_len, d_model)
print("entrada:", x.shape)                      # [4, 10, 64]

q = torch.randn(batch, seq_len, d_model)
print("após projeção linear:", q.shape)          # [4, 10, 64]

q = q.view(batch, seq_len, num_heads, d_k)
print("após view (split heads):", q.shape)       # [4, 10, 4, 16]

q = q.transpose(1, 2)
print("após transpose(1,2):", q.shape)           # [4, 4, 10, 16]

k = q.clone()
scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
print("scores:", scores.shape)                   # [4, 4, 10, 10]

attn = torch.softmax(scores, dim=-1)
print("attn (após softmax):", attn.shape)        # [4, 4, 10, 10]

v = q.clone()
out = attn @ v
print("attn @ V:", out.shape)                    # [4, 4, 10, 16]

out = out.transpose(1, 2)
print("após transpose de volta:", out.shape)     # [4, 10, 4, 16]

out = out.contiguous().view(batch, seq_len, d_model)
print("após view (combine heads):", out.shape)   # [4, 10, 64]

out_final = out @ torch.randn(d_model, d_model)
print("após W_O:", out_final.shape)              # [4, 10, 64]
```

### Exercício 17.3: Implementar split_heads e combine_heads
```python
import torch

def split_heads(x, batch_size, num_heads, d_k):
    # x: [batch, seq_len, d_model]
    x = x.view(batch_size, -1, num_heads, d_k)
    return x.transpose(1, 2)
    # retorna: [batch, num_heads, seq_len, d_k]

def combine_heads(x, batch_size, d_model):
    # x: [batch, num_heads, seq_len, d_k]
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, -1, d_model)
    # retorna: [batch, seq_len, d_model]

# Teste rápido
batch_size, seq_len, d_model, num_heads = 2, 5, 8, 2
d_k = d_model // num_heads
x = torch.randn(batch_size, seq_len, d_model)
split = split_heads(x, batch_size, num_heads, d_k)
print(split.shape)  # [2, 2, 5, 4]
combined = combine_heads(split, batch_size, d_model)
print(combined.shape)  # [2, 5, 8]
print(torch.allclose(x, combined))  # True — split/combine são inversos exatos
```

### Exercício 17.4: Contar Parâmetros
```python
d_model = 256
num_heads = 8

# W_Q, W_K, W_V, W_O são todos [d_model, d_model]
params_mha = 4 * (d_model * d_model)
print(f"Multi-Head (num_heads={num_heads}): {params_mha} parâmetros")
# 4 * 256 * 256 = 262144

# Uma "super-cabeça" com d_model completo também usa W_Q,W_K,W_V,W_O [d_model,d_model]
params_single = 4 * (d_model * d_model)
print(f"Single-head (d_model completo): {params_single} parâmetros")
# 262144 -- EXATAMENTE O MESMO!
```
A contagem de parâmetros é idêntica entre multi-head e single-head com `d_model` completo — a diferença não está em "quantos parâmetros", mas em *como* eles são organizados e usados: multi-head reparticiona o mesmo espaço em subespaços paralelos independentes, permitindo múltiplos padrões de atenção simultâneos sem custo extra de parâmetros.

### Exercício 17.5: Cabeças Idênticas
Se todas as cabeças recebem pesos idênticos, cada uma computa exatamente a mesma matriz de atenção e a mesma saída (dentro do seu subespaço de `d_k` dimensões). Concatenar `h` cópias idênticas de uma computação de `d_k` dimensões e projetar com `W_O` não dá mais expressividade do que uma única cabeça de `d_k` dimensões — na prática, seria **pior** que `num_heads=1` com `d_model` completo, porque cada cabeça só enxerga uma fatia de `d_k` dimensões da entrada, e sem diversidade entre cabeças você perde a capacidade de capturar múltiplos padrões sem ganhar nada em troca. A força do multi-head vem justamente da **diversidade** entre as cabeças, que surge da inicialização aleatória e do gradiente descendente empurrando cada cabeça para nichos diferentes.

---

## Desafios Avançados (Opcionais)

### Fixação 17.1: Cabeças com Dimensões Diferentes
Implemente uma variante de Multi-Head Attention onde cada cabeça tem uma `d_k` diferente (por exemplo, cabeça 0 com `d_k=8`, cabeça 1 com `d_k=24`), desde que a soma seja igual a `d_model`. O que precisa mudar na implementação de `split_heads`?

### Fixação 17.2: Atenção Cruzada (Cross-Attention) Multi-Head
Adapte a implementação para que Q venha de uma sequência e K, V venham de outra sequência de comprimento diferente (útil em arquiteturas encoder-decoder). Verifique que os shapes de `scores` ficam `[batch, num_heads, seq_len_q, seq_len_kv]`.

### Fixação 17.3: Custo Computacional em FLOPs
Derive a fórmula de FLOPs para uma camada de Multi-Head Attention em função de `batch, seq_len, d_model, num_heads`. Mostre que o termo dominante (`seq_len^2 * d_model`) não depende de `num_heads` — por quê?

### Fixação 17.4: Poda de Cabeças (Head Pruning)
Depois de treinar o modelo do Experimento 2, "zere" a saída de uma cabeça específica (force `attn_weights` daquela cabeça a ser uniforme, ou zere sua contribuição em `W_O`) e meça o impacto na loss. Cabeças diferentes têm impactos diferentes?

### Fixação 17.5: Visualizar Entropia por Cabeça
Compute a entropia da distribuição de atenção de cada cabeça (média sobre posições). Cabeças com entropia baixa estão "focadas" em poucos tokens; cabeças com entropia alta estão "espalhando" atenção. Isso muda ao longo do treinamento?

---

## Resumo

- **Limitação de cabeça única**: uma única matriz de atenção precisa fazer média entre múltiplos padrões de relação conflitantes, perdendo especificidade.
- **Divisão em cabeças**: `d_model` é particionado em `num_heads` subespaços de `d_k = d_model / num_heads` dimensões, cada um com sua própria atenção.
- **Projeção única, divisão depois**: projete Q, K, V com matrizes completas `[d_model, d_model]` e só então faça `view` + `transpose` para separar em cabeças — isso é equivalente a ter matrizes por cabeça, mas muito mais eficiente.
- **Batched matmul**: empacotar `[batch, num_heads]` como dimensões líderes permite computar todas as cabeças de todos os exemplos em paralelo, sem loops.
- **Concatenação via reshape**: combinar as cabeças de volta a `d_model` é um `transpose` + `contiguous` + `view`, seguido de projeção com `W_O` que mistura a informação entre cabeças.
- **Mesmo custo de parâmetros**: multi-head não adiciona parâmetros em relação a uma única cabeça com `d_model` completo — reorganiza a mesma capacidade para permitir múltiplos padrões simultâneos.

Próximo capítulo: **Residual Connections** — como conexões de atalho permitem empilhar dezenas de blocos Transformer sem que o gradiente desapareça.

---

**Próximo**: [Capítulo 18: Residual Connections](18_residual_connections.md)
