# Capítulo 21: Bloco Transformer Completo

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Descrever o fluxo completo de dados através de um bloco Transformer, na ordem correta
2. Explicar a arquitetura Pre-LN e por que ela é preferida sobre a Post-LN original
3. Implementar um `TransformerBlock` reutilizável como `nn.Module`
4. Verificar a invariante de shape (entrada == saída) através de um bloco e de uma pilha de blocos
5. Discutir a intuição de que camadas diferentes na pilha aprendem representações progressivamente mais abstratas

---

## Por Que Isso Importa

Nos últimos capítulos você construiu, peça por peça, os componentes que formam o coração de um Transformer: a atenção multi-head (Capítulos 10-17), as conexões residuais (Capítulo 18), a layer normalization (Capítulo 19) e, no capítulo anterior, a feed-forward network. Cada peça isolada resolve um problema específico — a atenção mistura informação entre tokens, o FFN processa cada token individualmente, os residuais permitem treinar redes profundas sem perder o sinal do gradiente, e a normalização estabiliza a escala das ativações.

Mas nenhuma dessas peças, sozinha, é um Transformer. O que realmente faz a arquitetura funcionar é a maneira específica como essas peças são combinadas e a ordem exata em que os dados fluem por elas. Trocar a ordem de duas operações, ou esquecer uma conexão residual, pode significar a diferença entre um modelo que treina establemente até bilhões de parâmetros e um que diverge nas primeiras iterações.

Pense em uma linha de montagem industrial: cada estação (atenção, normalização, FFN) sabe exatamente fazer sua tarefa, mas o produto final só sai correto se as estações estiverem na ordem certa e cada uma passar adiante exatamente o que a próxima espera receber. Neste capítulo, vamos montar essa linha de montagem completa — o **bloco Transformer** — e entender por que a ordem específica que usamos (chamada de arquitetura *Pre-LN*) se tornou o padrão de fato em praticamente todo LLM relevante desde o GPT-2.

E há um detalhe crucial que torna tudo isso possível de empilhar: um bloco Transformer **preserva o shape de sua entrada**. Isso significa que podemos empilhar dezenas ou centenas desses blocos, um depois do outro, sem nunca precisar mudar a forma dos dados — cada bloco simplesmente refina a representação que recebeu. É essa propriedade de composição simples que permite escalar Transformers de alguns milhões para centenas de bilhões de parâmetros apenas empilhando mais blocos.

---

## Anatomia do Bloco Transformer

### As duas sub-camadas

Um bloco Transformer decoder-only (o tipo usado em modelos GPT-like, que é nosso foco neste livro) contém exatamente duas sub-camadas principais:

1. **Multi-Head Self-Attention** (com máscara causal) — a fase de "comunicação entre tokens"
2. **Feed-Forward Network** — a fase de "processamento individual por token"

Cada uma dessas sub-camadas é envolvida por uma **conexão residual** e precedida por uma **layer normalization**. É essa combinação — normalização, sub-camada, soma residual — que define a estrutura de um bloco.

### Pre-LN vs. Post-LN

Existem duas maneiras de organizar a normalização em relação à sub-camada e ao residual:

**Post-LN (arquitetura original do paper "Attention is All You Need", 2017)**:

$$x' = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Aqui, a normalização acontece *depois* de somar o residual.

**Pre-LN (usada em GPT-2, GPT-3, Llama e na quase totalidade dos LLMs modernos)**:

$$x' = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Aqui, a normalização acontece *antes* da sub-camada, e o residual soma o input original (não normalizado) diretamente à saída da sub-camada.

A diferença parece pequena, mas tem uma consequência enorme para treinamento em larga escala: na Pre-LN, o caminho residual (o "trilho principal" por onde o gradiente flui, discutido no Capítulo 18) nunca passa por uma normalização. Isso significa que o gradiente pode fluir de volta através de dezenas de blocos empilhados praticamente sem distorção de escala. Na Post-LN, cada bloco insere uma normalização diretamente no caminho principal do gradiente, o que historicamente tornou o treinamento de redes muito profundas com Post-LN instável sem técnicas cuidadosas de *warmup* de taxa de aprendizado. Por essa razão, a Pre-LN se tornou a escolha padrão sempre que se quer treinar modelos com muitas camadas de forma estável e com menos ajuste fino de hiperparâmetros — e é a que usaremos daqui em diante.

### Fluxo completo de dados através de um bloco (Pre-LN)

```
                         x  [batch, seq_len, d_model]
                         │
                         ├─────────────────────────────┐
                         │                              │ (caminho residual,
                         ▼                              │  sem transformação)
                  LayerNorm(x)                          │
                         │                               │
                         ▼                               │
          Multi-Head Self-Attention                      │
           (com máscara causal)                          │
                         │                               │
                         ▼                               │
                    Attn_out                              │
                         │                               │
                         └──────────► (+) ◄───────────────┘
                                       │
                                       ▼
                         x1 = x + Attn_out   [batch, seq_len, d_model]
                                       │
                         ┌─────────────┴────────────────┐
                         │                               │ (caminho residual,
                         ▼                               │  sem transformação)
                  LayerNorm(x1)                          │
                         │                               │
                         ▼                               │
                Feed-Forward Network                     │
                (Linear -> GELU -> Linear)                │
                         │                               │
                         ▼                               │
                     FFN_out                              │
                         │                               │
                         └──────────► (+) ◄───────────────┘
                                       │
                                       ▼
                    x2 = x1 + FFN_out   [batch, seq_len, d_model]
                                       │
                                       ▼
                            (saída do bloco)
```

Note que o shape se mantém `[batch, seq_len, d_model]` do início ao fim — essa é a **invariante de shape** mencionada na introdução, e é ela que permite empilhar blocos livremente.

### Formalizando em equações

Para um bloco $\ell$, dado o input $x_\ell$:

$$a_\ell = \text{MultiHeadAttention}(\text{LayerNorm}_1(x_\ell))$$
$$x_\ell' = x_\ell + a_\ell$$
$$f_\ell = \text{FFN}(\text{LayerNorm}_2(x_\ell'))$$
$$x_{\ell+1} = x_\ell' + f_\ell$$

O output $x_{\ell+1}$ tem exatamente o mesmo shape que a entrada $x_\ell$, e serve como entrada do próximo bloco $\ell+1$. Repita esse processo $N$ vezes (num_layers) para formar a pilha completa do Transformer.

---

## Implementação: TransformerBlock

```python
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """Atenção multi-head com máscara causal (revisão dos Capítulos 15-17)."""
    def __init__(self, d_model, num_heads, max_seq_len=512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # máscara causal: posição i não pode olhar para j > i
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # cada um [B, T, D]

        # separar em cabeças: [B, T, D] -> [B, num_heads, T, d_head]
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, T, T]
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # [B, H, T, d_head]

        out = out.transpose(1, 2).contiguous().view(B, T, D)  # concatena cabeças
        return self.W_o(out)


class FeedForward(nn.Module):
    """Feed-forward posição-a-posição (Capítulo 20)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Um bloco Transformer completo, arquitetura Pre-LN."""
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sub-camada 1: atenção, com Pre-LN e residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Sub-camada 2: feed-forward, com Pre-LN e residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```

---

## Experimento: Bloco Transformer e Pilha de Blocos

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Bloco Transformer Completo")
print("=" * 70)

# ========== 1. CONFIGURAÇÃO ==========
print("\n1. CONFIGURAÇÃO")
print("-" * 70)

d_model = 32
num_heads = 4
d_ff = 4 * d_model
batch_size = 2
seq_len = 6
num_layers = 4

print(f"d_model: {d_model}")
print(f"num_heads: {num_heads}")
print(f"d_ff: {d_ff}")
print(f"num_layers (blocos empilhados): {num_layers}")
print(f"batch_size: {batch_size}, seq_len: {seq_len}")

# ========== 2. CRIANDO UM BLOCO E VERIFICANDO SHAPE ==========
print("\n2. UM ÚNICO BLOCO TRANSFORMER")
print("-" * 70)

block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=seq_len)
X = torch.randn(batch_size, seq_len, d_model)
print(f"X (entrada) shape: {X.shape}")

Y = block(X)
print(f"Y (saída) shape: {Y.shape}")
print(f"Invariante de shape preservada? {X.shape == Y.shape}")

# ========== 3. CONTAGEM DE PARÂMETROS DE UM BLOCO ==========
print("\n3. CONTAGEM DE PARÂMETROS DE UM BLOCO")
print("-" * 70)

n_attn = sum(p.numel() for p in block.attn.parameters())
n_ffn = sum(p.numel() for p in block.ffn.parameters())
n_ln = sum(p.numel() for p in block.ln1.parameters()) + sum(p.numel() for p in block.ln2.parameters())
n_total = sum(p.numel() for p in block.parameters())

print(f"Atenção multi-head: {n_attn} parâmetros")
print(f"Feed-forward:        {n_ffn} parâmetros")
print(f"LayerNorms (x2):      {n_ln} parâmetros")
print(f"Total do bloco:       {n_total} parâmetros")
print(f"Razão FFN/Atenção: {n_ffn / n_attn:.2f}x (esperado ~2x)")

# ========== 4. EMPILHANDO N BLOCOS ==========
print("\n4. EMPILHANDO MÚLTIPLOS BLOCOS")
print("-" * 70)

blocks = nn.ModuleList([
    TransformerBlock(d_model, num_heads, d_ff, max_seq_len=seq_len)
    for _ in range(num_layers)
])

h = X
print(f"Shape inicial: {h.shape}")
for i, blk in enumerate(blocks):
    h = blk(h)
    print(f"Após bloco {i+1}: shape = {h.shape}, "
          f"média = {h.mean().item():.4f}, desvio padrão = {h.std().item():.4f}")

print(f"\nShape final == shape inicial? {h.shape == X.shape}")

# ========== 5. VERIFICANDO QUE O RESIDUAL CARREGA SINAL ==========
print("\n5. VERIFICANDO FLUXO DO CAMINHO RESIDUAL")
print("-" * 70)

# Se zerarmos os pesos das sub-camadas, a saída deveria ser ~igual à entrada
# (pois o caminho residual passa direto)
block_zero = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=seq_len)
with torch.no_grad():
    for p in block_zero.attn.parameters():
        p.zero_()
    for p in block_zero.ffn.parameters():
        p.zero_()

X_test = torch.randn(1, seq_len, d_model)
Y_test = block_zero(X_test)
diff = (X_test - Y_test).abs().max().item()
print(f"Com sub-camadas zeradas, diferença máxima entrada/saída: {diff:.2e}")
print("(pequena, pois o residual deixa o input passar quase direto)")

# ========== 6. GRADIENTES FLUEM POR TODA A PILHA ==========
print("\n6. VERIFICANDO FLUXO DE GRADIENTES POR TODA A PILHA")
print("-" * 70)

X_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
h = X_grad
for blk in blocks:
    h = blk(h)

loss = h.sum()
loss.backward()

print(f"Gradiente em X (entrada da pilha) existe? {X_grad.grad is not None}")
print(f"Norma do gradiente em X: {X_grad.grad.norm().item():.4f}")
print("Gradiente não-nulo confirma que o sinal flui de volta por todos os blocos")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Intuição: O Que Cada Camada Aprende?

Uma pergunta natural, ao empilhar dezenas de blocos idênticos em estrutura, é: por que empilhar tantos, e o que cada um faz de diferente se todos têm a mesma arquitetura?

A resposta empírica, observada em análises de modelos treinados (via técnicas como *probing* de representações internas), sugere uma intuição — não uma lei matemática rígida — de que camadas em posições diferentes da pilha tendem a se especializar em tipos diferentes de padrões:

- **Camadas iniciais** tendem a capturar padrões mais locais e sintáticos: relações entre palavras adjacentes, concordância gramatical, estrutura de frases curtas.
- **Camadas intermediárias** começam a capturar relações um pouco mais abstratas, combinando informação sintática com pistas semânticas.
- **Camadas profundas** tendem a capturar relações de mais longo alcance e mais abstratas: correferência entre entidades distantes no texto, relações lógicas, e informação necessária para tarefas que dependem de contexto amplo.

É importante frisar: essa é uma **intuição observada empiricamente em muitos estudos**, não uma propriedade garantida pela arquitetura. Nenhuma camada é "forçada" a aprender sintaxe ou semântica — cada bloco tem exatamente a mesma estrutura computacional, e a especialização emerge do processo de treinamento por gradiente descendente, não de um design explícito. Além disso, essa progressão "sintaxe → semântica" é uma simplificação; análises mais detalhadas mostram que muitos tipos de informação estão distribuídos e redundantes por várias camadas simultaneamente. Ainda assim, a intuição é útil como ponto de partida para pensar sobre por que empilhar mais blocos geralmente aumenta a capacidade do modelo de capturar padrões cada vez mais complexos — cada bloco adicional dá à rede mais uma "rodada" de comunicação entre tokens (atenção) seguida de processamento individual (FFN), permitindo refinar progressivamente a representação de cada posição.

---

## Erros Comuns

### Erro 1: Aplicar LayerNorm depois da sub-camada (Post-LN) sem perceber

```python
# Errado (Post-LN, mais difícil de treinar em pilhas profundas sem warmup cuidadoso)
def forward(self, x):
    x = self.ln1(x + self.attn(x))
    x = self.ln2(x + self.ffn(x))
    return x

# Certo (Pre-LN, padrão em LLMs modernos)
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x
```

Não é que Post-LN esteja "errado" matematicamente — foi a arquitetura original do paper de 2017 e funciona para modelos rasos. Mas para pilhas profundas (dezenas de blocos), Pre-LN é dramaticamente mais estável e é o que praticamente todo LLM moderno usa. Confundir as duas silenciosamente é um erro fácil de cometer ao copiar pseudocódigo de fontes diferentes.

### Erro 2: Esquecer de aplicar a segunda LayerNorm antes do FFN

```python
# Errado: reutiliza a normalização da atenção, ou pula normalização antes do FFN
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(x)  # faltou LayerNorm antes do FFN!
    return x

# Certo: cada sub-camada tem sua PRÓPRIA LayerNorm com parâmetros independentes
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))  # ln2 é um módulo diferente de ln1
    return x
```

Cada sub-camada precisa da sua própria instância de `nn.LayerNorm`, com parâmetros $\gamma$ e $\beta$ aprendidos independentemente. Reutilizar o mesmo módulo, ou pular a normalização antes de uma das sub-camadas, quebra a simetria da arquitetura Pre-LN e tipicamente degrada a estabilidade do treinamento.

### Erro 3: Não usar `nn.ModuleList` ao empilhar blocos

```python
# Errado: lista Python comum — os parâmetros não são registrados pelo PyTorch!
blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
# model.parameters() NÃO vai incluir os pesos desses blocos
# optimizer não vai atualizá-los, e model.to(device) não vai movê-los

# Certo: nn.ModuleList registra corretamente os submódulos
blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
```

Usar uma lista Python padrão em vez de `nn.ModuleList` é um erro sutil e traiçoeiro: o código roda sem erros, o forward pass funciona normalmente, mas os parâmetros dos blocos armazenados na lista **não aparecem** em `model.parameters()`. Isso significa que o otimizador nunca vai atualizá-los durante o treinamento — o modelo parece treinar (a loss pode até cair um pouco, por causa dos parâmetros que *estão* registrados), mas a maior parte da capacidade do modelo fica congelada nos valores de inicialização aleatória.

---

## Exercícios

### Exercício 21.1: Verificando a Invariante de Shape
Crie um `TransformerBlock` com `d_model=64`, `num_heads=8`, `d_ff=256`. Passe tensores de entrada com diferentes `batch_size` e `seq_len` e confirme que a saída sempre tem o mesmo shape da entrada.

### Exercício 21.2: Pre-LN vs Post-LN
Implemente as duas variantes (Pre-LN e Post-LN) do `TransformerBlock`. Empilhe 20 blocos de cada tipo (sem treinar) e compare a norma da ativação de saída da última camada. O que você observa em termos de escala das ativações?

### Exercício 21.3: Contagem de Parâmetros da Pilha Completa
Para `d_model=768`, `num_heads=12`, `d_ff=3072` e `num_layers=12` (configuração aproximada do GPT-2 small), calcule o número total de parâmetros de toda a pilha de blocos Transformer (sem contar embeddings nem cabeça de saída).

### Exercício 21.4: Ablação do Residual
Modifique o `TransformerBlock` para permitir desligar as conexões residuais (um flag `use_residual=False`). Empilhe 10 blocos sem residual e observe o que acontece com a norma das ativações — e, se possível, com os gradientes via `backward()`.

### Exercício 21.5: Um Bloco "Vazio"
Crie um bloco cujas sub-camadas de atenção e FFN retornam sempre zero (`torch.zeros_like(x)`). Verifique que, graças ao residual, a saída do bloco é idêntica à entrada.

---

## Gabarito

### Exercício 21.1: Verificando a Invariante de Shape
```python
import torch

block = TransformerBlock(d_model=64, num_heads=8, d_ff=256, max_seq_len=20)

for batch_size, seq_len in [(1, 5), (4, 10), (2, 20)]:
    X = torch.randn(batch_size, seq_len, 64)
    Y = block(X)
    assert X.shape == Y.shape, f"Shape mismatch: {X.shape} != {Y.shape}"
    print(f"batch={batch_size}, seq_len={seq_len}: OK, shape = {Y.shape}")
```

### Exercício 21.2: Pre-LN vs Post-LN
```python
import torch
import torch.nn as nn

class PostLNBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x

torch.manual_seed(0)
d_model, num_heads, d_ff, seq_len = 32, 4, 128, 10

pre_ln_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, seq_len) for _ in range(20)])
post_ln_blocks = nn.ModuleList([PostLNBlock(d_model, num_heads, d_ff, seq_len) for _ in range(20)])

X = torch.randn(1, seq_len, d_model)

h_pre = X
for blk in pre_ln_blocks:
    h_pre = blk(h_pre)

h_post = X
for blk in post_ln_blocks:
    h_post = blk(h_post)

print(f"Pre-LN,  norma final: {h_pre.norm().item():.4f}")
print(f"Post-LN, norma final: {h_post.norm().item():.4f}")
# Post-LN tende a ter ativações com escala mais variável/instável em pilhas profundas
```

### Exercício 21.3: Contagem de Parâmetros da Pilha Completa
```python
d_model, num_heads, d_ff, num_layers = 768, 12, 3072, 12

block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=1024)
n_por_bloco = sum(p.numel() for p in block.parameters())
n_total = n_por_bloco * num_layers

print(f"Parâmetros por bloco: {n_por_bloco:,}")
print(f"Parâmetros na pilha completa ({num_layers} blocos): {n_total:,}")
```

### Exercício 21.4: Ablação do Residual
```python
import torch
import torch.nn as nn

class TransformerBlockAblation(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, use_residual=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.use_residual = use_residual

    def forward(self, x):
        a = self.attn(self.ln1(x))
        x = (x + a) if self.use_residual else a
        f = self.ffn(self.ln2(x))
        x = (x + f) if self.use_residual else f
        return x

torch.manual_seed(3)
d_model, num_heads, d_ff, seq_len = 32, 4, 128, 10
X = torch.randn(1, seq_len, d_model, requires_grad=True)

blocks_no_res = nn.ModuleList([
    TransformerBlockAblation(d_model, num_heads, d_ff, seq_len, use_residual=False)
    for _ in range(10)
])

h = X
for blk in blocks_no_res:
    h = blk(h)

print(f"Sem residual, norma final: {h.norm().item():.4f}")
h.sum().backward()
print(f"Sem residual, norma do gradiente em X: {X.grad.norm().item():.6f}")
# Tipicamente a norma do gradiente é muito menor (ou instável) sem residual
```

### Exercício 21.5: Um Bloco "Vazio"
```python
import torch
import torch.nn as nn

class ZeroSublayerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        zero_attn = torch.zeros_like(x)
        x = x + zero_attn  # LayerNorm calculado mas ignorado, apenas para ilustrar o fluxo
        zero_ffn = torch.zeros_like(x)
        x = x + zero_ffn
        return x

block = ZeroSublayerBlock(d_model=16)
X = torch.randn(1, 5, 16)
Y = block(X)

diff = (X - Y).abs().max().item()
print(f"Diferença máxima: {diff:.2e}")  # exatamente 0
```

---

## Desafios Avançados (Opcionais)

### Fixação 21.1: Warmup de Learning Rate para Post-LN
Pesquise e implemente um scheduler de *warmup* linear de learning rate. Treine (com dados sintéticos) uma pilha Post-LN com e sem warmup, e compare a estabilidade da loss nas primeiras iterações.

### Fixação 21.2: Sandwich-LN
Alguns modelos usam uma variante que aplica LayerNorm tanto antes quanto depois de cada sub-camada ("sandwich"). Implemente essa variante e discuta o trade-off de custo computacional extra vs. estabilidade.

### Fixação 21.3: Escala das Ativações por Profundidade
Empilhe 50 blocos Pre-LN (sem treinar, pesos aleatórios) e plote a norma média das ativações após cada bloco. A norma cresce, decresce ou se mantém estável? Por quê, dado o caminho residual sem normalização?

### Fixação 21.4: Weight Sharing Entre Blocos
Modifique a pilha para que todos os blocos compartilhem os mesmos pesos (um único `TransformerBlock` aplicado repetidamente `num_layers` vezes). Compare a contagem de parâmetros e discuta prós/contras dessa técnica (usada, por exemplo, em modelos como ALBERT).

### Fixação 21.5: Profiling de Custo por Sub-camada
Usando `torch.profiler` ou medição manual de tempo, meça separadamente quanto tempo o forward pass gasta na sub-camada de atenção vs. na sub-camada FFN, para diferentes valores de `seq_len`. Em que ponto a atenção (custo quadrático em seq_len) ultrapassa o FFN (custo linear em seq_len)?

---

## Resumo

- **Bloco Transformer**: combina Multi-Head Attention e FFN, cada uma envolvida por LayerNorm e conexão residual
- **Pre-LN**: `x + Sublayer(LayerNorm(x))` — normalização antes da sub-camada, residual soma o input original; padrão em LLMs modernos por estabilidade em pilhas profundas
- **Post-LN**: `LayerNorm(x + Sublayer(x))` — arquitetura original de 2017, mais difícil de treinar em pilhas muito profundas sem warmup cuidadoso
- **Invariante de shape**: um bloco Transformer sempre preserva `[batch, seq_len, d_model]`, permitindo empilhar N blocos livremente
- **Empilhamento**: mais blocos (num_layers) geram mais "rodadas" de comunicação (atenção) + processamento (FFN), aumentando a capacidade do modelo
- **Intuição de profundidade**: camadas iniciais tendem a capturar padrões locais/sintáticos, camadas profundas tendem a capturar relações mais abstratas e de longo alcance — uma observação empírica, não uma garantia da arquitetura

Próximo capítulo: **Positional Embeddings** — vamos resolver um problema que a atenção sozinha não consegue: saber a ordem dos tokens na sequência.

---

**Próximo**: [Capítulo 22: Positional Embeddings](22_positional_embeddings.md)
