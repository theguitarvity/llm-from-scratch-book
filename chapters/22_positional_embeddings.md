# Capítulo 22: Positional Embeddings

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Provar, conceitualmente e numericamente, que a atenção sozinha é invariante à permutação da ordem dos tokens
2. Entender por que essa invariância é um problema para modelar linguagem
3. Implementar positional encoding sinusoidal e explicar por que essa forma específica foi escolhida
4. Implementar positional embeddings aprendidos e compará-los com a abordagem sinusoidal
5. Descrever, em linhas gerais, a ideia por trás de RoPE (Rotary Position Embeddings) e por que ela se tornou popular em LLMs modernos

---

## Por Que Isso Importa

Imagine as duas frases:

> "O cachorro mordeu o carteiro"
> "O carteiro mordeu o cachorro"

Elas contêm exatamente as mesmas palavras. Se você embaralhar apenas a ordem, o significado muda completamente — quem morde e quem é mordido se invertem. Para qualquer humano, é óbvio que a ordem das palavras carrega informação essencial.

Aqui está o problema: o mecanismo de atenção, do jeito que construímos até agora (Capítulos 10-17), **não sabe distinguir essas duas frases**. Se você passar os embeddings de "cachorro", "mordeu" e "carteiro" para a atenção em qualquer ordem, o *conjunto* de attention scores calculado entre cada par de tokens é exatamente o mesmo — a atenção calcula similaridade par a par via produto escalar entre Query e Key, uma operação que não tem noção nenhuma de "quem veio antes de quem" na sequência. A atenção enxerga a entrada como um *conjunto* de tokens, não como uma *sequência ordenada*.

Isso é surpreendente à primeira vista, porque intuitivamente parece que o modelo "deveria" saber a ordem. Mas se você olhar para a matemática pura de $Q K^T$, não há nada ali que dependa da posição — apenas do conteúdo (os valores dos embeddings). É como dar a alguém um saco de palavras escritas em post-its separados: sem numerá-los, a pessoa não tem como saber a ordem original da frase, mesmo que consiga ler cada post-it perfeitamente.

A solução é simples em princípio, mas cheia de nuances interessantes na prática: precisamos **injetar informação de posição** diretamente nos embeddings, antes que eles entrem no primeiro bloco Transformer. Neste capítulo você vai entender por que a atenção é "cega" à ordem, ver a prova disso rodando código, e implementar as duas abordagens clássicas para resolver o problema — positional encoding sinusoidal (fixo, sem parâmetros aprendidos) e positional embeddings aprendidos (uma tabela de lookup, como os embeddings de token) — além de conhecer brevemente RoPE, a abordagem usada em LLMs mais recentes como a família Llama.

---

## A Atenção é Permutation Invariant

### Prova conceitual

Considere uma sequência de embeddings $X \in \mathbb{R}^{[n, d]}$, com linhas $x_1, x_2, \ldots, x_n$. Suponha que trocamos a ordem de duas linhas quaisquer, digamos $x_i$ e $x_j$, produzindo uma nova matriz $X'$. Formalmente, $X' = P X$, onde $P$ é uma matriz de permutação (uma matriz de zeros e uns que reorganiza as linhas).

Os scores de atenção são calculados como $\text{scores} = Q K^T = (X W_Q)(X W_K)^T$. Se aplicarmos a mesma permutação $P$ à entrada:

$$\text{scores}' = (P X W_Q)(P X W_K)^T = P (X W_Q)(X W_K)^T P^T = P \cdot \text{scores} \cdot P^T$$

O que essa equação diz é que a matriz de scores da entrada permutada é **exatamente a mesma matriz de scores original, apenas com linhas e colunas reorganizadas na mesma ordem em que permutamos a entrada**. Nenhum valor novo aparece, nenhum valor desaparece — apenas a posição de cada valor dentro da matriz muda, acompanhando a permutação.

O mesmo vale para o softmax (aplicado linha a linha, então permuta junto) e para a multiplicação final por $V$ (que também é permutado da mesma forma). O resultado final é que o output da atenção para o token que estava na posição $i$ "segue" esse token para onde quer que ele seja movido — o *conteúdo* do output de cada token não muda, apenas sua posição na sequência de outputs.

Em outras palavras: **atenção trata a entrada como um conjunto (set), não como uma sequência ordenada**. Se você embaralhar a ordem dos tokens de entrada, e depois desembaralhar a saída na mesma ordem, o resultado é idêntico ao de não ter embaralhado nada.

### Verificação numérica

Vamos comprovar isso diretamente:

```python
import torch

torch.manual_seed(42)

d = 8
X = torch.randn(4, d)  # 4 tokens, embedding dim 8

W_Q = torch.randn(d, d) * 0.1
W_K = torch.randn(d, d) * 0.1
W_V = torch.randn(d, d) * 0.1

def self_attention(X, W_Q, W_K, W_V):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.T / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V

# Ordem original
out_original = self_attention(X, W_Q, W_K, W_V)

# Permutar a ordem dos tokens (por exemplo, inverter)
perm = torch.tensor([3, 2, 1, 0])
X_perm = X[perm]

out_perm = self_attention(X_perm, W_Q, W_K, W_V)

# Desfazer a permutação no output
out_perm_undone = out_perm[perm.argsort()]

diff = (out_original - out_perm_undone).abs().max().item()
print(f"Diferença máxima: {diff:.2e}")  # ~0, confirmando invariância de permutação
```

Isso confirma a prova: sem informação de posição, a atenção "não sabe" a ordem. Ela produz o resultado correto por token, mas o token em si não carrega nenhuma pista de onde estava na sequência original.

### Por que isso é um problema

Para tarefas de linguagem, a ordem é informação semântica essencial (lembre do exemplo "cachorro mordeu carteiro" vs. "carteiro mordeu cachorro"). Além disso, um modelo de linguagem autoregressivo precisa saber, ao prever o próximo token, quantos tokens já foram gerados e em que posições relativas eles estão — informação de "distância" entre tokens é usada constantemente para captar dependências de curto e longo alcance (concordância entre sujeito e verbo próximos, correferência entre um pronome e uma entidade mencionada muitos tokens atrás, etc). Sem posição, todo esse tipo de raciocínio fica impossível.

A solução: somar (ou de alguma forma combinar) aos embeddings de token uma representação vetorial que codifica a posição de cada token na sequência, **antes** de entrar no primeiro bloco Transformer.

---

## Positional Encoding Sinusoidal

### A fórmula

A abordagem original do paper "Attention is All You Need" define o positional encoding usando funções seno e cosseno de frequências diferentes:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Onde:
- $pos$ é a posição do token na sequência (0, 1, 2, ...)
- $i$ é o índice da dimensão dentro do embedding (percorre pares consecutivos, $0 \le i < d_{model}/2$)
- $d_{model}$ é a dimensão do embedding

Ou seja, para cada posição, geramos um vetor de tamanho $d_{model}$ em que as dimensões pares recebem valores de seno e as dimensões ímpares recebem valores de cosseno, cada par de dimensões usando uma frequência diferente (dimensões mais "baixas" no índice oscilam mais rápido; dimensões mais "altas" oscilam mais devagar).

Esse vetor $PE_{pos} \in \mathbb{R}^{d_{model}}$ é então **somado** diretamente ao embedding de token na mesma posição:

$$x_{pos} = \text{TokenEmbedding}(pos) + PE_{pos}$$

O shape não muda: $X \in \mathbb{R}^{[n, d_{model}]}$ continua exatamente com a mesma forma, apenas com um sinal adicional embutido nos valores.

### Por que essa fórmula específica?

Três propriedades tornam essa escolha elegante:

1. **Cada posição recebe um padrão único**: como as frequências variam continuamente ao longo das dimensões, duas posições diferentes praticamente nunca produzem o mesmo vetor de codificação (exceto em casos degenerados de sequências extremamente longas).

2. **Permite generalizar para sequências mais longas do que as vistas em treino**: como seno e cosseno são funções definidas para qualquer valor real, você pode calcular $PE_{pos}$ para qualquer $pos$, mesmo posições nunca vistas durante o treinamento. Isso não garante que o modelo generalize *semanticamente* bem para sequências muito mais longas (isso depende de outros fatores), mas pelo menos a codificação de posição em si nunca "quebra" ou fica indefinida.

3. **Permite ao modelo aprender posições relativas via combinações lineares**: para qualquer deslocamento fixo $k$, existe uma transformação linear que mapeia $PE_{pos}$ para $PE_{pos+k}$, independente de $pos$. Isso decorre das identidades trigonométricas de soma de ângulos:
$$\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$
$$\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$
Ou seja, $PE_{pos+k}$ pode ser expresso como uma combinação linear (com coeficientes que dependem apenas de $k$, não de $pos$) de $PE_{pos}$. Isso significa que, em princípio, camadas lineares do modelo (como as projeções $W_Q, W_K$) podem aprender a extrair informação sobre distância *relativa* entre tokens, não apenas posição absoluta — o que é frequentemente mais útil para entender relações linguísticas.

### Implementação

```python
import torch
import math

def positional_encoding_sinusoidal(seq_len, d_model):
    """Retorna tensor [seq_len, d_model] com positional encoding sinusoidal."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]

    # termo de frequência: 10000^(2i/d_model), calculado de forma numericamente estável
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )  # [d_model/2]

    pe[:, 0::2] = torch.sin(position * div_term)  # dimensões pares
    pe[:, 1::2] = torch.cos(position * div_term)  # dimensões ímpares

    return pe  # [seq_len, d_model]
```

---

## Positional Embeddings Aprendidos

### A ideia

Em vez de usar uma fórmula fixa, podemos tratar a posição exatamente como tratamos tokens: como um índice em uma tabela de lookup, cujos valores são **parâmetros aprendidos** durante o treinamento.

```python
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len):
        positions = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
        return self.embedding(positions)   # [seq_len, d_model]
```

Assim como a tabela de token embeddings mapeia `token_id -> vetor denso`, essa tabela mapeia `posição -> vetor denso`. Durante o treinamento, o gradiente flui de volta para essa tabela, e o modelo aprende a representação de posição que melhor minimiza a loss — sem que precisemos escolher manualmente uma fórmula.

### Trade-offs entre as duas abordagens

| Aspecto | Sinusoidal | Aprendido |
|---|---|---|
| Parâmetros extras | Zero | `max_seq_len × d_model` |
| Generaliza para sequências mais longas que o treino | Sim, naturalmente (a fórmula é definida para qualquer posição) | Não diretamente — posições nunca vistas não têm embedding treinado |
| Flexibilidade | Fixa, não se adapta aos dados | Aprende o padrão mais útil para a tarefa, a partir dos dados |
| Uso em LLMs modernos | Menos comum atualmente | Foi usado em GPT-2/GPT-3; hoje muitos modelos preferem RoPE (veja abaixo) |

Na prática, positional embeddings aprendidos costumam ter desempenho ligeiramente melhor em tarefas dentro do comprimento de sequência visto em treino (porque se adaptam aos dados), mas perdem a capacidade de extrapolar para sequências mais longas — se seu modelo foi treinado com `max_seq_len = 1024` e você tentar rodá-lo com sequência de 2000 tokens, simplesmente não existem embeddings de posição treinados para os índices acima de 1024.

---

## Uma Palavra Sobre RoPE

Modelos de linguagem mais recentes — como a família Llama, Mistral, e muitos outros LLMs open-source relevantes — abandonaram tanto a codificação sinusoidal somada quanto os embeddings de posição aprendidos, em favor de uma técnica chamada **RoPE (Rotary Position Embeddings)**.

A ideia central de RoPE é diferente das duas abordagens anteriores: em vez de *somar* um vetor de posição ao embedding do token, RoPE **rotaciona** os vetores de Query e Key (dentro do cálculo de atenção) por um ângulo que depende da posição do token. Cada par de dimensões consecutivas do vetor Q ou K é tratado como coordenadas de um ponto 2D, e esse ponto é rotacionado por um ângulo proporcional à posição.

A propriedade elegante de RoPE é que, quando você calcula o produto escalar $Q_i \cdot K_j$ entre um token na posição $i$ e outro na posição $j$, o resultado matemático depende **apenas da distância relativa $(i - j)$**, não das posições absolutas. Isso incorpora diretamente na própria operação de atenção a noção de posição relativa — sem precisar somar nada aos embeddings de entrada, e com boas propriedades de extrapolação para sequências mais longas que as vistas em treino.

RoPE é uma técnica mais sofisticada, com detalhes matemáticos que envolvem rotações em espaços de múltiplas dimensões pareadas; ela está fora do escopo detalhado deste capítulo, mas é importante saber que ela existe e é a escolha dominante em LLMs modernos de ponta. Para os propósitos deste livro, vamos seguir com as duas abordagens mais simples (sinusoidal e aprendida), que são suficientes para construir e entender um Transformer funcional, e mencionaremos RoPE novamente quando discutirmos otimizações avançadas no Capítulo 34.

---

## Experimento: Positional Embeddings

```python
import torch
import torch.nn as nn
import math

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Positional Embeddings")
print("=" * 70)

# ========== 1. PROVANDO A INVARIÂNCIA DE PERMUTAÇÃO DA ATENÇÃO ==========
print("\n1. PROVANDO QUE ATENÇÃO É PERMUTATION INVARIANT")
print("-" * 70)

d = 8
seq_len = 4
X = torch.randn(seq_len, d)

W_Q = torch.randn(d, d) * 0.1
W_K = torch.randn(d, d) * 0.1
W_V = torch.randn(d, d) * 0.1

def self_attention(X):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores = Q @ K.T / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V

out_original = self_attention(X)
perm = torch.tensor([3, 1, 0, 2])
out_perm = self_attention(X[perm])
out_perm_undone = out_perm[perm.argsort()]

diff = (out_original - out_perm_undone).abs().max().item()
print(f"X original shape: {X.shape}")
print(f"Diferença máxima após permutar e desfazer: {diff:.2e}")
print("Confirma: atenção sem info de posição é 'cega' à ordem dos tokens")

# ========== 2. POSITIONAL ENCODING SINUSOIDAL ==========
print("\n2. POSITIONAL ENCODING SINUSOIDAL")
print("-" * 70)

def positional_encoding_sinusoidal(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

d_model = 16
seq_len = 6
pe = positional_encoding_sinusoidal(seq_len, d_model)
print(f"PE shape: {pe.shape}")
print(f"PE (posição 0): {pe[0][:8]}")
print(f"PE (posição 1): {pe[1][:8]}")
print(f"PE (posição 5): {pe[5][:8]}")

# Cada posição é única
distancias = torch.cdist(pe, pe)
print(f"\nDistância euclidiana entre posição 0 e todas as outras:")
print(distancias[0])

# ========== 3. SOMANDO POSITIONAL ENCODING AOS TOKEN EMBEDDINGS ==========
print("\n3. SOMANDO PE AOS TOKEN EMBEDDINGS")
print("-" * 70)

vocab_size = 100
token_embedding = nn.Embedding(vocab_size, d_model)
token_ids = torch.tensor([5, 27, 5, 8, 91, 5])  # note: token 5 aparece 3 vezes

tok_emb = token_embedding(token_ids)  # [seq_len, d_model]
print(f"Token embeddings shape: {tok_emb.shape}")

x_with_pos = tok_emb + pe
print(f"X final (token + posição) shape: {x_with_pos.shape}")

# Mesmo token em posições diferentes agora tem representações diferentes
idx_token5 = (token_ids == 5).nonzero().flatten()
print(f"\nToken 5 aparece nas posições: {idx_token5.tolist()}")
for idx in idx_token5:
    print(f"  x_with_pos[{idx}][:4] = {x_with_pos[idx][:4]}")
print("Mesmo token, posições diferentes -> vetores finais diferentes (correto!)")

# ========== 4. POSITIONAL EMBEDDING APRENDIDO ==========
print("\n4. POSITIONAL EMBEDDING APRENDIDO")
print("-" * 70)

max_seq_len = 20
learned_pos_emb = nn.Embedding(max_seq_len, d_model)

positions = torch.arange(seq_len)
pe_learned = learned_pos_emb(positions)
print(f"PE aprendido shape: {pe_learned.shape}")
print(f"Parâmetros da tabela de posição aprendida: {sum(p.numel() for p in learned_pos_emb.parameters())}")
print(f"(= max_seq_len x d_model = {max_seq_len} x {d_model} = {max_seq_len * d_model})")

x_with_learned_pos = tok_emb + pe_learned
print(f"X final (token + posição aprendida) shape: {x_with_learned_pos.shape}")

# ========== 5. GENERALIZAÇÃO PARA SEQUÊNCIAS MAIS LONGAS ==========
print("\n5. GENERALIZAÇÃO PARA SEQUÊNCIAS MAIS LONGAS QUE O TREINO")
print("-" * 70)

seq_len_maior = 50  # maior que max_seq_len=20 usado no embedding aprendido

# Sinusoidal: funciona para qualquer comprimento
pe_longo_sinusoidal = positional_encoding_sinusoidal(seq_len_maior, d_model)
print(f"Sinusoidal para seq_len={seq_len_maior}: shape {pe_longo_sinusoidal.shape} (funciona sem problema)")

# Aprendido: falha se tentarmos indexar além de max_seq_len
try:
    positions_longo = torch.arange(seq_len_maior)
    pe_longo_aprendido = learned_pos_emb(positions_longo)
    print(f"Aprendido para seq_len={seq_len_maior}: OK")
except IndexError as e:
    print(f"Aprendido para seq_len={seq_len_maior}: ERRO -> {e}")
    print("(índice de posição excede o tamanho da tabela treinada)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Concatenar em vez de somar o positional encoding

```python
# Errado: concatenar muda a dimensão, quebrando compatibilidade com o resto do modelo
x = torch.cat([tok_emb, pe], dim=-1)  # [seq_len, 2*d_model] -- shape errado!

# Certo: somar preserva o shape [seq_len, d_model]
x = tok_emb + pe
```

A convenção padrão em Transformers é **somar** o positional encoding ao token embedding, não concatenar. Concatenar dobraria a dimensão e exigiria ajustar todas as matrizes de projeção subsequentes — além de não ser a prática usada na literatura padrão.

### Erro 2: Recalcular positional encoding para cada token individualmente com posição errada

```python
# Errado: usar sempre a posição 0, ou uma posição fixa
pe_fixo = positional_encoding_sinusoidal(1, d_model)  # sempre "posição 0"
x = tok_emb + pe_fixo  # erro de broadcasting/shape, e semanticamente sem sentido

# Certo: gerar PE para o comprimento exato da sequência, com posições 0..seq_len-1
pe = positional_encoding_sinusoidal(seq_len, d_model)
x = tok_emb + pe  # [seq_len, d_model] + [seq_len, d_model]
```

Cada posição da sequência precisa do seu próprio vetor de codificação correspondente ao seu índice real (0, 1, 2, ...). Usar sempre a mesma posição anula completamente o propósito da técnica.

### Erro 3: Positional embedding aprendido sem checar o limite de max_seq_len

```python
# Errado: não verificar se a sequência de entrada cabe na tabela
learned_pos_emb = nn.Embedding(max_seq_len=512, d_model=768)
# ... depois, durante inferência, alguém passa uma sequência de 600 tokens
positions = torch.arange(600)
pe = learned_pos_emb(positions)  # IndexError!

# Certo: validar ou truncar a sequência antes de indexar
seq_len = min(seq_len_real, max_seq_len)
positions = torch.arange(seq_len)
pe = learned_pos_emb(positions)
```

Diferente da codificação sinusoidal (que funciona para qualquer comprimento), a tabela de embeddings aprendidos tem um limite físico de tamanho definido em `max_seq_len`. Tentar indexar além desse limite gera um erro em tempo de execução — é importante validar o comprimento da sequência de entrada contra esse limite antes de rodar o modelo.

---

## Exercícios

### Exercício 22.1: Provando Invariância de Permutação
Repita a verificação numérica da Seção 1, mas usando uma sequência de 6 tokens e uma permutação diferente (por exemplo, embaralhamento aleatório com `torch.randperm`). Confirme que a diferença após desfazer a permutação é próxima de zero.

### Exercício 22.2: Positional Encoding Manual
Calcule manualmente (com calculadora ou papel) os dois primeiros valores de `PE[2, :]` (posição 2, dimensões 0 e 1) para `d_model=4`. Depois confirme com código.

### Exercício 22.3: Similaridade Entre Posições
Compute o positional encoding sinusoidal para `seq_len=50, d_model=64`. Calcule a similaridade de cosseno entre a posição 10 e todas as outras posições. O que você observa sobre como a similaridade varia com a distância?

### Exercício 22.4: Comparando Sinusoidal e Aprendido
Implemente ambos os métodos para `d_model=32, max_seq_len=100`. Compare o número de parâmetros extras de cada abordagem e explique o trade-off em termos de memória.

### Exercício 22.5: Testando Extrapolação
Treine (de forma simplificada, apenas alguns passos de gradiente em uma tarefa sintética) um modelo pequeno com positional embedding aprendido usando `max_seq_len=32`. Tente rodar inferência com uma sequência de 40 tokens e observe o erro. Repita o experimento trocando para positional encoding sinusoidal e confirme que funciona sem erro.

---

## Gabarito

### Exercício 22.1: Provando Invariância de Permutação
```python
import torch

torch.manual_seed(7)
d = 8
seq_len = 6
X = torch.randn(seq_len, d)
W_Q, W_K, W_V = torch.randn(d, d) * 0.1, torch.randn(d, d) * 0.1, torch.randn(d, d) * 0.1

def self_attention(X):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores = Q @ K.T / (d ** 0.5)
    return torch.softmax(scores, dim=-1) @ V

perm = torch.randperm(seq_len)
out_original = self_attention(X)
out_perm = self_attention(X[perm])
out_undone = out_perm[perm.argsort()]

diff = (out_original - out_undone).abs().max().item()
print(f"Permutação usada: {perm.tolist()}")
print(f"Diferença máxima: {diff:.2e}")  # ~0
```

### Exercício 22.2: Positional Encoding Manual
```python
import math

d_model = 4
pos = 2

# i=0: PE[2,0] = sin(2 / 10000^(0/4)) = sin(2/1) = sin(2)
pe_0 = math.sin(pos / (10000 ** (0/d_model)))
# i=0: PE[2,1] = cos(2 / 10000^(0/4)) = cos(2)
pe_1 = math.cos(pos / (10000 ** (0/d_model)))

print(f"PE[2,0] manual = {pe_0:.6f}")  # sin(2) = 0.909297
print(f"PE[2,1] manual = {pe_1:.6f}")  # cos(2) = -0.416147

import torch
def positional_encoding_sinusoidal(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = positional_encoding_sinusoidal(3, d_model)
print(f"PE[2,:2] via código = {pe[2,:2]}")  # deve bater com o cálculo manual
```

### Exercício 22.3: Similaridade Entre Posições
```python
import torch
import torch.nn.functional as F

pe = positional_encoding_sinusoidal(50, 64)
pos10 = pe[10].unsqueeze(0)  # [1, 64]
sims = F.cosine_similarity(pos10, pe, dim=-1)  # [50]

print("Similaridade da posição 10 com posições próximas e distantes:")
for p in [8, 9, 10, 11, 12, 30, 49]:
    print(f"  pos {p}: {sims[p]:.4f}")
# Espera-se: similaridade mais alta para posições próximas de 10,
# diminuindo (não necessariamente monotonicamente) com a distância
```

### Exercício 22.4: Comparando Sinusoidal e Aprendido
```python
import torch.nn as nn

d_model, max_seq_len = 32, 100

# Sinusoidal: zero parâmetros
n_params_sinusoidal = 0

# Aprendido: uma tabela de lookup
learned = nn.Embedding(max_seq_len, d_model)
n_params_learned = sum(p.numel() for p in learned.parameters())

print(f"Sinusoidal: {n_params_sinusoidal} parâmetros extras")
print(f"Aprendido: {n_params_learned} parâmetros extras (= {max_seq_len} x {d_model})")
```

### Exercício 22.5: Testando Extrapolação
```python
import torch
import torch.nn as nn

max_seq_len = 32
d_model = 16
learned_pos_emb = nn.Embedding(max_seq_len, d_model)

# Inferência com seq_len dentro do limite: OK
positions_ok = torch.arange(30)
pe_ok = learned_pos_emb(positions_ok)
print(f"seq_len=30 (dentro do limite): OK, shape {pe_ok.shape}")

# Inferência com seq_len acima do limite: erro
try:
    positions_bad = torch.arange(40)
    pe_bad = learned_pos_emb(positions_bad)
except IndexError as e:
    print(f"seq_len=40 (acima do limite): ERRO -> {e}")

# Versão sinusoidal: funciona sem erro em qualquer comprimento
pe_sinusoidal_40 = positional_encoding_sinusoidal(40, d_model)
print(f"Sinusoidal, seq_len=40: OK, shape {pe_sinusoidal_40.shape}")
```

---

## Desafios Avançados (Opcionais)

### Fixação 22.1: Implementando RoPE (Simplificado)
Implemente uma versão simplificada de RoPE, rotacionando pares de dimensões de Q e K por um ângulo proporcional à posição. Verifique numericamente que $Q_i \cdot K_j$ (após a rotação) depende apenas de $(i - j)$, não de $i$ e $j$ separadamente.

### Fixação 22.2: Visualizando o Positional Encoding
Gere um heatmap (matplotlib) do positional encoding sinusoidal para `seq_len=100, d_model=128`. Observe visualmente os padrões de onda em diferentes frequências ao longo das dimensões.

### Fixação 22.3: Positional Embeddings Relativos
Pesquise sobre "relative positional embeddings" (usado, por exemplo, em modelos como Transformer-XL e T5). Implemente uma versão simplificada que injeta um bias diretamente nos scores de atenção baseado na distância $(i-j)$, em vez de somar aos embeddings de entrada.

### Fixação 22.4: Interpolação de Posição para Extrapolação
Pesquise a técnica de "position interpolation" usada para estender o contexto de modelos com RoPE além do comprimento de treino. Implemente uma versão conceitual simples que reescala os índices de posição para caber dentro do range treinado.

### Fixação 22.5: Combinando Múltiplas Estratégias
Implemente um modelo que soma tanto o positional encoding sinusoidal quanto um positional embedding aprendido (dois sinais de posição diferentes somados). Treine em uma tarefa sintética simples e compare a convergência com cada estratégia isolada.

---

## Resumo

- **Atenção é permutation invariant**: sem informação adicional de posição, embaralhar a ordem dos tokens de entrada apenas embaralha a ordem dos outputs — o conteúdo de cada output não muda
- **Positional encoding**: um vetor somado ao embedding de token para injetar informação de posição na sequência
- **Sinusoidal**: fórmula fixa com seno/cosseno em frequências variadas; zero parâmetros extras; generaliza naturalmente para sequências mais longas que o treino; permite ao modelo aprender posições relativas via combinações lineares
- **Aprendido**: tabela de lookup treinável (como token embeddings, mas para posição); se adapta aos dados, mas não extrapola além de `max_seq_len`
- **RoPE**: técnica moderna (usada em Llama e outros LLMs recentes) que rotaciona Q e K em função da posição, fazendo com que o produto escalar dependa apenas da distância relativa entre tokens
- **Soma, não concatenação**: positional encoding é somado ao token embedding, preservando o shape `[seq_len, d_model]`

Próximo capítulo: **Arquitetura Fim a Fim** — vamos montar o Transformer completo, do token de entrada até os logits de saída, unindo tudo que construímos até aqui.

---

**Próximo**: [Capítulo 23: Arquitetura Fim a Fim](23_arquitetura_fim_a_fim.md)
