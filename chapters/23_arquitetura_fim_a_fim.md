# Capítulo 23: Arquitetura Fim a Fim

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Descrever o fluxo completo de um Transformer decoder-only, do token de entrada aos logits de saída
2. Implementar a classe `GPTModel` completa como `nn.Module`, unindo embeddings, pilha de blocos e cabeça de saída
3. Calcular a contagem de parâmetros de cada componente e entender como ela escala com `d_model`, `num_layers` e `vocab_size`
4. Explicar e implementar weight tying entre a matriz de embedding e a cabeça de saída
5. Testar um forward pass completo, verificando shapes de entrada e saída em cada estágio

---

## Por Que Isso Importa

Você chegou ao ponto em que todas as peças já existem: sabe como transformar um token em um vetor denso (embedding), como injetar informação de posição, como fazer tokens "conversarem" entre si via atenção, como processar cada token individualmente via FFN, como manter o treinamento estável com conexões residuais e layer normalization, e como empilhar tudo isso em um bloco Transformer reutilizável.

Este capítulo é sobre montagem: pegar essas peças e construir a arquitetura completa de um modelo de linguagem autoregressivo estilo GPT — o tipo de arquitetura por trás de praticamente todo LLM relevante hoje usado para geração de texto (a família GPT, Llama, Mistral, e muitos outros). É o momento em que o livro deixa de falar sobre "componentes de um Transformer" e passa a falar sobre "um Transformer".

A boa notícia é que, depois de todo o trabalho dos capítulos anteriores, a montagem final é conceitualmente simples: é uma sequência linear de passos bem definidos, cada um recebendo a saída do anterior. Pegamos IDs de tokens (números inteiros), transformamos em vetores, somamos posição, passamos por uma pilha de blocos idênticos em estrutura, normalizamos uma última vez, e projetamos de volta para o espaço do vocabulário para obter uma distribuição de probabilidade sobre qual deveria ser o próximo token.

Entender essa arquitetura completa também é essencial para uma questão prática que todo praticante de LLMs enfrenta: quantos parâmetros esse modelo vai ter, e onde eles estão concentrados? Isso importa para escolher o tamanho do modelo que cabe na sua GPU, para estimar tempo de treinamento, e para entender trade-offs de escala (por que dobrar `d_model` custa muito mais que dobrar `num_layers`, por exemplo). Vamos fazer essas contas explicitamente neste capítulo, e também conhecer uma técnica simples e elegante chamada *weight tying*, que reduz a contagem de parâmetros compartilhando pesos entre a entrada e a saída do modelo.

---

## O Fluxo Completo: Token de Entrada até Logits de Saída

### Visão geral em diagrama

```
Entrada: sequência de token IDs
   input_ids  [batch, seq_len]   (inteiros, cada um em [0, vocab_size))
        │
        ▼
┌───────────────────────┐
│   Token Embedding      │   nn.Embedding(vocab_size, d_model)
│   [vocab_size, d_model]│
└───────────┬────────────┘
            │
            ▼
   tok_emb  [batch, seq_len, d_model]
            │
            │         ┌────────────────────────┐
            ├────────►│  Positional Embedding    │  sinusoidal ou aprendido
            │         │  [seq_len, d_model]       │
            │         └───────────┬──────────────┘
            │                     │
            ▼                     ▼
            └─────────(+)─────────┘
                        │
                        ▼
            x = tok_emb + pos_emb   [batch, seq_len, d_model]
                        │
                        ▼
              ┌──────────────────┐
              │  Dropout (opcional)│
              └─────────┬──────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   TransformerBlock  x 1        │
        └───────────────┬─────────────────┘
                        │  [batch, seq_len, d_model]
                        ▼
        ┌───────────────────────────────┐
        │   TransformerBlock  x 2        │
        └───────────────┬─────────────────┘
                        │
                       ...  (repete num_layers vezes)
                        │
        ┌───────────────────────────────┐
        │   TransformerBlock  x N        │
        └───────────────┬─────────────────┘
                        │  [batch, seq_len, d_model]
                        ▼
              ┌──────────────────┐
              │  LayerNorm final   │
              └─────────┬──────────┘
                        │  [batch, seq_len, d_model]
                        ▼
              ┌──────────────────┐
              │  Linear Head       │   nn.Linear(d_model, vocab_size)
              │  (LM Head)          │   (frequentemente com weight tying)
              └─────────┬──────────┘
                        │
                        ▼
                logits  [batch, seq_len, vocab_size]
```

O output final, `logits`, tem uma entrada para **cada posição da sequência de entrada** e, para cada posição, um valor (não normalizado) para **cada palavra possível do vocabulário**. Interpretamos `logits[b, t, :]` como a "pontuação bruta" que o modelo atribui a cada palavra do vocabulário como candidata a ser o próximo token depois da posição $t$, para o exemplo $b$ do batch. No Capítulo 26 você verá como transformar esses logits em uma distribuição de probabilidade (via softmax) e como calcular a loss de treinamento (cross-entropy) a partir deles.

### Shapes em cada estágio

| Estágio | Shape | Observação |
|---|---|---|
| `input_ids` | `[batch, seq_len]` | inteiros (índices de vocabulário) |
| `tok_emb` | `[batch, seq_len, d_model]` | lookup na tabela de embeddings |
| `pos_emb` | `[seq_len, d_model]` | broadcast automático sobre `batch` |
| `x` (após soma) | `[batch, seq_len, d_model]` | entrada da pilha de blocos |
| saída de cada bloco | `[batch, seq_len, d_model]` | invariante de shape (Capítulo 21) |
| após LayerNorm final | `[batch, seq_len, d_model]` | mesmo shape, valores normalizados |
| `logits` | `[batch, seq_len, vocab_size]` | única mudança de dimensão em todo o pipeline |

Vale notar: a **única** transformação de shape que altera a última dimensão em todo o pipeline (fora da entrada inicial) é a projeção final para `vocab_size`. Tudo entre o embedding e a LayerNorm final preserva `d_model` — é exatamente essa invariante que permite empilhar quantos blocos quisermos sem reconfigurar nada no meio do caminho.

---

## Implementação: GPTModel Completo

```python
import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


def positional_encoding_sinusoidal(seq_len, d_model, device=None):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class GPTModel(nn.Module):
    """Transformer decoder-only estilo GPT, fim a fim."""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        d_ff=None,
        num_layers=6,
        max_seq_len=512,
        dropout=0.1,
        use_learned_positional=False,
        tie_weights=True,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_learned_positional = use_learned_positional

        # 1. Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional embedding (aprendido, opcional; sinusoidal é calculado sob demanda)
        if use_learned_positional:
            self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # 3. Pilha de blocos Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        # 4. LayerNorm final
        self.ln_final = nn.LayerNorm(d_model)

        # 5. Cabeça de saída (projeção para o vocabulário)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: compartilha os pesos entre embedding de entrada e cabeça de saída
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequência de comprimento {T} excede max_seq_len={self.max_seq_len}"

        tok_emb = self.token_embedding(input_ids)  # [B, T, d_model]

        if self.use_learned_positional:
            positions = torch.arange(T, device=input_ids.device)
            pos_emb = self.positional_embedding(positions)  # [T, d_model]
        else:
            pos_emb = positional_encoding_sinusoidal(T, self.d_model, device=input_ids.device)

        x = self.dropout(tok_emb + pos_emb)  # broadcast de pos_emb sobre a dimensão batch

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Notas sobre a implementação

- **Broadcast do positional embedding**: `pos_emb` tem shape `[T, d_model]` (sem dimensão de batch), enquanto `tok_emb` tem shape `[B, T, d_model]`. O PyTorch faz *broadcasting* automático, somando o mesmo `pos_emb` a cada exemplo do batch — não é necessário replicar manualmente.
- **`lm_head` sem bias**: é comum omitir o bias na camada de projeção final quando se usa weight tying, já que o bias não participa do compartilhamento de pesos e adicionaria uma assimetria desnecessária.
- **Inicialização**: usamos desvio padrão pequeno (0.02), seguindo a convenção do GPT-2 — inicializações maiores tendem a produzir ativações com escala excessiva logo nas primeiras camadas, especialmente combinadas com a arquitetura Pre-LN.

---

## Contagem de Parâmetros por Componente

Vamos decompor de onde vêm os parâmetros de um `GPTModel`, em função de `vocab_size` ($V$), `d_model` ($d$), `d_ff$ (tipicamente $4d$), `num_layers` ($L$) e `max_seq_len` ($S$, apenas se usar embedding aprendido):

### 1. Token embedding

$$P_{\text{tok\_emb}} = V \times d$$

### 2. Positional embedding (se aprendido; sinusoidal não adiciona parâmetros)

$$P_{\text{pos\_emb}} = S \times d$$

### 3. Cada bloco Transformer

Da atenção multi-head (matrizes $W_Q, W_K, W_V$ combinadas em `W_qkv`, mais $W_O$):

$$P_{\text{attn}} = \underbrace{d \times 3d + 3d}_{W_{qkv} \text{ com bias}} + \underbrace{d \times d + d}_{W_O \text{ com bias}} \approx 4d^2$$

Do feed-forward (Capítulo 20):

$$P_{\text{ffn}} = \underbrace{d \times 4d + 4d}_{\text{linear1}} + \underbrace{4d \times d + d}_{\text{linear2}} \approx 8d^2$$

Das duas LayerNorms ($\gamma$ e $\beta$, cada uma de tamanho $d$):

$$P_{\text{ln}} = 2 \times 2d = 4d$$

Total por bloco (aproximado, ignorando termos lineares em $d$ frente aos quadráticos):

$$P_{\text{bloco}} \approx 4d^2 + 8d^2 = 12 d^2$$

Para $L$ blocos:

$$P_{\text{blocos}} \approx 12 L d^2$$

### 4. LayerNorm final

$$P_{\text{ln\_final}} = 2d$$

### 5. Cabeça de saída (LM head)

$$P_{\text{head}} = V \times d \quad \text{(sem weight tying)}$$
$$P_{\text{head}} = 0 \quad \text{(com weight tying — reutiliza os pesos do token embedding)}$$

### Total aproximado (sem weight tying)

$$P_{\text{total}} \approx 2 V d + 12 L d^2$$

### Total aproximado (com weight tying)

$$P_{\text{total}} \approx V d + 12 L d^2$$

Essa fórmula explica um fato importante sobre escala de LLMs: para modelos grandes (onde $d$ é da ordem de milhares e $L$ é dezenas), o termo $12 L d^2$ domina completamente sobre $Vd$ (onde $V$ raramente passa de algumas dezenas de milhares). Isso significa que **dobrar `d_model` custa aproximadamente 4x mais parâmetros** (por causa do termo quadrático $d^2$), enquanto **dobrar `num_layers` custa apenas 2x mais parâmetros** (termo linear em $L$). Essa assimetria é uma das razões pelas quais escalar modelos costuma envolver aumentar `d_model` e `num_layers` em conjunto, de forma equilibrada, em vez de escalar apenas uma das duas dimensões.

---

## Weight Tying

### A ideia

Observe que tanto o `token_embedding` quanto o `lm_head` são, essencialmente, transformações entre o mesmo par de espaços: um mapeia `token_id -> vetor de dimensão d_model`, o outro mapeia `vetor de dimensão d_model -> pontuação para cada token_id possível`. Um é a "leitura" do vocabulário, o outro é a "escrita" no vocabulário — e ambos operam sobre a mesma matriz de shape `[vocab_size, d_model]` (a cabeça de saída é matematicamente $x \cdot E^T$ quando $E$ é a matriz de embedding).

**Weight tying** consiste em literalmente usar a **mesma matriz de parâmetros** para as duas operações:

```python
self.lm_head.weight = self.token_embedding.weight
```

Isso funciona porque `nn.Linear(d_model, vocab_size, bias=False)` internamente armazena um peso de shape `[vocab_size, d_model]` — exatamente o mesmo shape de `nn.Embedding(vocab_size, d_model).weight`. Ao atribuir o mesmo tensor `nn.Parameter` aos dois módulos, eles passam a compartilhar fisicamente os mesmos valores na memória: qualquer atualização de gradiente em um afeta o outro automaticamente, porque são o mesmo objeto.

### Por que isso funciona bem

Além da economia óbvia de parâmetros ($V \times d$ a menos, o que pode representar dezenas de milhões de parâmetros em modelos com vocabulário grande), existe uma justificativa conceitual: se dois tokens têm embeddings de entrada similares (porque são semanticamente parecidos), faz sentido que o modelo também atribua pontuações de saída similares a esses dois tokens quando ambos são plausíveis como próxima palavra. Compartilhar os pesos incentiva essa consistência diretamente, e empiricamente costuma melhorar a qualidade do modelo, além de reduzir o número de parâmetros — um ganho duplo. Essa técnica foi popularizada por trabalhos como "Using the Output Embedding to Improve Language Models" e é usada, entre outros, no GPT-2.

---

## Experimento: GPTModel Completo

```python
import torch

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: GPTModel Fim a Fim")
print("=" * 70)

# ========== 1. CONFIGURAÇÃO DO MODELO ==========
print("\n1. CONFIGURAÇÃO")
print("-" * 70)

config = dict(
    vocab_size=1000,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=4,
    max_seq_len=128,
    dropout=0.1,
    tie_weights=True,
)
for k, v in config.items():
    print(f"  {k}: {v}")

model = GPTModel(**config)
print("\nModelo criado com sucesso")

# ========== 2. CONTAGEM DE PARÂMETROS ==========
print("\n2. CONTAGEM DE PARÂMETROS")
print("-" * 70)

n_tok_emb = sum(p.numel() for p in model.token_embedding.parameters())
n_blocks = sum(p.numel() for p in model.blocks.parameters())
n_ln_final = sum(p.numel() for p in model.ln_final.parameters())
n_head = sum(p.numel() for p in model.lm_head.parameters())
n_total = model.count_parameters()

print(f"Token embedding:       {n_tok_emb:>10,}")
print(f"Pilha de blocos ({config['num_layers']}x):  {n_blocks:>10,}")
print(f"LayerNorm final:        {n_ln_final:>10,}")
print(f"LM head:                 {n_head:>10,}  (0 esperado, weight tying ativo)")
print(f"{'-'*40}")
print(f"TOTAL:                   {n_total:>10,}")

# Confirmar weight tying
mesmo_tensor = model.lm_head.weight is model.token_embedding.weight
print(f"\nlm_head.weight é o MESMO objeto que token_embedding.weight? {mesmo_tensor}")

# ========== 3. FORWARD PASS ==========
print("\n3. FORWARD PASS")
print("-" * 70)

batch_size = 3
seq_len = 10
input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

print(f"input_ids shape: {input_ids.shape}")
print(f"input_ids (exemplo 0): {input_ids[0].tolist()}")

logits = model(input_ids)
print(f"\nlogits shape: {logits.shape}")
print(f"Esperado: [batch={batch_size}, seq_len={seq_len}, vocab_size={config['vocab_size']}]")

# ========== 4. INTERPRETANDO OS LOGITS ==========
print("\n4. INTERPRETANDO OS LOGITS")
print("-" * 70)

probs = torch.softmax(logits[0, -1, :], dim=-1)  # distribuição sobre o próximo token, última posição
top5 = torch.topk(probs, 5)
print("Top-5 tokens mais prováveis para a próxima posição (modelo não treinado, aleatório):")
for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
    print(f"  token_id={idx:4d}  prob={prob:.4f}")
print(f"Soma de todas as probabilidades: {probs.sum().item():.6f} (deve ser ~1.0)")

# ========== 5. VERIFICANDO GRADIENTES FLUEM ATÉ O EMBEDDING ==========
print("\n5. VERIFICANDO FLUXO DE GRADIENTES")
print("-" * 70)

targets = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, config["vocab_size"]), targets.view(-1)
)
print(f"Loss (cross-entropy, pesos aleatórios): {loss.item():.4f}")
print(f"Loss esperada aproximada para pesos aleatórios: ln({config['vocab_size']}) = {torch.log(torch.tensor(float(config['vocab_size']))).item():.4f}")

loss.backward()
grad_norm_emb = model.token_embedding.weight.grad.norm().item()
print(f"\nNorma do gradiente em token_embedding.weight: {grad_norm_emb:.4f}")
print("(diferente de zero, confirma que o gradiente flui da saída até o embedding de entrada)")

# ========== 6. ESCALANDO O MODELO: COMPARANDO CONFIGURAÇÕES ==========
print("\n6. COMO OS PARÂMETROS ESCALAM COM d_model E num_layers")
print("-" * 70)

configs_teste = [
    dict(vocab_size=1000, d_model=64, num_heads=4, num_layers=4, max_seq_len=128),
    dict(vocab_size=1000, d_model=128, num_heads=4, num_layers=4, max_seq_len=128),  # d_model 2x
    dict(vocab_size=1000, d_model=64, num_heads=4, num_layers=8, max_seq_len=128),   # num_layers 2x
]

print(f"{'d_model':>10} {'num_layers':>12} {'total_params':>15}")
for cfg in configs_teste:
    m = GPTModel(**cfg, d_ff=4*cfg["d_model"])
    n = m.count_parameters()
    print(f"{cfg['d_model']:>10} {cfg['num_layers']:>12} {n:>15,}")

print("\nObservação: dobrar d_model aumenta parâmetros ~4x (termo d^2);")
print("dobrar num_layers aumenta parâmetros ~2x (termo linear em L)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Esquecer de aplicar a máscara causal / vazamento de informação futura

```python
# Errado: usar atenção sem máscara causal em um modelo autoregressivo
class MultiHeadSelfAttentionSemMascara(nn.Module):
    def forward(self, x):
        # ... calcula scores sem aplicar masked_fill
        attn = torch.softmax(scores, dim=-1)  # token pode "ver" o futuro!
        return attn @ v

# Certo: sempre aplicar a máscara causal antes do softmax em modelos GPT-like
scores = scores.masked_fill(~causal_mask, float("-inf"))
attn = torch.softmax(scores, dim=-1)
```

Sem a máscara causal (Capítulo 15), o modelo teria acesso a tokens futuros durante o treinamento — o que invalida completamente o objetivo de predição autoregressiva de próximo token. O modelo pareceria treinar muito bem (loss baixíssima), mas seria incapaz de gerar texto coerente durante a inferência real, quando os tokens futuros simplesmente não existem ainda.

### Erro 2: Weight tying com shapes incompatíveis

```python
# Errado: tentar weight tying quando lm_head tem bias ou vocab_size diferente
lm_head = nn.Linear(d_model, vocab_size, bias=True)  # tem bias
lm_head.weight = token_embedding.weight  # o bias fica sem correspondência, mas roda
# (funciona tecnicamente, mas mistura convenções -- prefira bias=False)

# Certo: lm_head sem bias, e vocab_size idêntico ao do embedding
lm_head = nn.Linear(d_model, vocab_size, bias=False)
assert lm_head.weight.shape == token_embedding.weight.shape
lm_head.weight = token_embedding.weight
```

Weight tying só é matematicamente coerente quando os dois módulos operam sobre o mesmo `vocab_size` e `d_model`. Um erro comum é mudar o `vocab_size` do tokenizer depois de já ter configurado o modelo, ou tentar aplicar weight tying entre módulos de shapes diferentes — isso gera um erro imediato do PyTorch ao tentar atribuir os pesos.

### Erro 3: Não mover positional encoding sinusoidal para o mesmo device do modelo

```python
# Errado: calcular positional encoding sem especificar device,
# causando erro ao rodar em GPU
def forward(self, input_ids):
    pos_emb = positional_encoding_sinusoidal(T, self.d_model)  # sempre em CPU!
    x = tok_emb + pos_emb  # RuntimeError se tok_emb estiver em GPU

# Certo: propagar o device de input_ids para a função
def forward(self, input_ids):
    pos_emb = positional_encoding_sinusoidal(T, self.d_model, device=input_ids.device)
    x = tok_emb + pos_emb  # ambos no mesmo device
```

Como o positional encoding sinusoidal é calculado sob demanda (não é um parâmetro registrado no modelo), é fácil esquecer de propagar o `device` correto ao criá-lo dinamicamente dentro do `forward`. Isso funciona silenciosamente em CPU, mas quebra com um erro de "tensors on different devices" assim que você move o modelo para GPU/MPS.

---

## Exercícios

### Exercício 23.1: Construindo um Modelo Pequeno
Instancie um `GPTModel` com `vocab_size=500, d_model=32, num_heads=2, num_layers=2, max_seq_len=64`. Rode um forward pass com `batch_size=2, seq_len=8` e imprima o shape de `logits`.

### Exercício 23.2: Contagem Manual de Parâmetros
Para a configuração do Exercício 23.1, calcule manualmente (usando as fórmulas da seção "Contagem de Parâmetros") o número aproximado de parâmetros total, e compare com `model.count_parameters()`.

### Exercício 23.3: Com e Sem Weight Tying
Instancie dois modelos idênticos, um com `tie_weights=True` e outro com `tie_weights=False`. Compare a contagem total de parâmetros e calcule a economia percentual gerada pelo weight tying.

### Exercício 23.4: Escalando vocab_size
Fixe `d_model=128, num_layers=4` e varie `vocab_size` entre 1.000, 10.000 e 50.000. Compute o total de parâmetros em cada caso (com e sem weight tying) e observe como o impacto de `vocab_size` muda relativamente ao resto do modelo.

### Exercício 23.5: Verificando Causalidade
Rode o modelo duas vezes com a mesma sequência de entrada, mas na segunda vez altere apenas o último token. Confirme que os logits nas posições anteriores ao último token **não mudam** (propriedade decorrente da máscara causal).

---

## Gabarito

### Exercício 23.1: Construindo um Modelo Pequeno
```python
model = GPTModel(vocab_size=500, d_model=32, num_heads=2, num_layers=2, max_seq_len=64)
input_ids = torch.randint(0, 500, (2, 8))
logits = model(input_ids)
print(f"logits shape: {logits.shape}")  # [2, 8, 500]
```

### Exercício 23.2: Contagem Manual de Parâmetros
```python
V, d, L = 500, 32, 2
d_ff = 4 * d

# Token embedding
p_tok_emb = V * d

# Por bloco: atenção (~4d^2) + ffn (~8d^2) + layernorms (4d)
p_attn = d * 3*d + 3*d + d*d + d       # W_qkv + bias, W_o + bias
p_ffn = d * d_ff + d_ff + d_ff * d + d  # linear1 + linear2, com bias
p_ln = 2 * (2*d)
p_bloco = p_attn + p_ffn + p_ln
p_blocos = p_bloco * L

# LayerNorm final
p_ln_final = 2 * d

# LM head (com weight tying = 0 parâmetros extras)
p_head = 0

total_manual = p_tok_emb + p_blocos + p_ln_final + p_head
print(f"Estimativa manual: {total_manual:,}")

model = GPTModel(vocab_size=V, d_model=d, num_heads=2, num_layers=L, max_seq_len=64)
print(f"model.count_parameters(): {model.count_parameters():,}")
```

### Exercício 23.3: Com e Sem Weight Tying
```python
cfg = dict(vocab_size=2000, d_model=64, num_heads=4, num_layers=3, max_seq_len=64)

model_tied = GPTModel(**cfg, tie_weights=True)
model_untied = GPTModel(**cfg, tie_weights=False)

n_tied = model_tied.count_parameters()
n_untied = model_untied.count_parameters()

economia = (n_untied - n_tied) / n_untied
print(f"Sem weight tying: {n_untied:,}")
print(f"Com weight tying: {n_tied:,}")
print(f"Economia: {economia:.2%}")
```

### Exercício 23.4: Escalando vocab_size
```python
d_model, num_layers = 128, 4

print(f"{'vocab_size':>12} {'sem tying':>15} {'com tying':>15} {'economia':>10}")
for V in [1000, 10000, 50000]:
    m_untied = GPTModel(vocab_size=V, d_model=d_model, num_heads=8, num_layers=num_layers, max_seq_len=128, tie_weights=False)
    m_tied = GPTModel(vocab_size=V, d_model=d_model, num_heads=8, num_layers=num_layers, max_seq_len=128, tie_weights=True)
    n_u, n_t = m_untied.count_parameters(), m_tied.count_parameters()
    print(f"{V:>12,} {n_u:>15,} {n_t:>15,} {(n_u-n_t)/n_u:>10.2%}")
# vocab_size maior -> weight tying economiza uma fração maior do total
```

### Exercício 23.5: Verificando Causalidade
```python
model = GPTModel(vocab_size=200, d_model=32, num_heads=4, num_layers=2, max_seq_len=32)
model.eval()

torch.manual_seed(0)
input_ids = torch.randint(0, 200, (1, 6))
input_ids_modificado = input_ids.clone()
input_ids_modificado[0, -1] = (input_ids_modificado[0, -1] + 1) % 200  # muda só o último token

with torch.no_grad():
    logits_original = model(input_ids)
    logits_modificado = model(input_ids_modificado)

# Posições 0 a seq_len-2 não devem mudar (não dependem do último token, por causalidade)
diff_anteriores = (logits_original[:, :-1, :] - logits_modificado[:, :-1, :]).abs().max().item()
diff_ultima = (logits_original[:, -1, :] - logits_modificado[:, -1, :]).abs().max().item()

print(f"Diferença máxima nas posições anteriores: {diff_anteriores:.2e}")  # ~0
print(f"Diferença máxima na última posição: {diff_ultima:.4f}")  # > 0, mudou como esperado
```

---

## Desafios Avançados (Opcionais)

### Fixação 23.1: KV Cache Conceitual
Durante geração autoregressiva token a token, recalcular Q, K, V para toda a sequência a cada novo token é desperdício. Pesquise a ideia de "KV cache" e escreva pseudocódigo (não precisa implementar completo) de como ele evitaria recomputação redundante.

### Fixação 23.2: Comparando Custo de Memória de Ativações
Para `batch_size=8, seq_len=512, d_model=768, num_layers=12`, estime (em MB, assumindo float32) a memória necessária apenas para armazenar as ativações intermediárias (`x` após cada bloco) durante um forward pass, sem contar gradientes.

### Fixação 23.3: Modelo com Múltiplos Positional Schemes
Modifique `GPTModel` para aceitar um parâmetro `positional_scheme` com valores `"sinusoidal"`, `"learned"` ou `"none"`. Teste o modelo com `"none"` e confirme (via alguma métrica, como perplexidade em uma tarefa sintética sensível a ordem) que ele performa pior do que com posição.

### Fixação 23.4: Inicialização Escalada por Profundidade
Pesquise a técnica usada no GPT-2 de escalar a inicialização dos pesos de saída de cada bloco por $1/\sqrt{2L}$ (onde $L$ é o número de camadas), para compensar o acúmulo de variância ao longo da pilha de residuais. Implemente essa inicialização e compare a escala das ativações finais com a inicialização padrão.

### Fixação 23.5: Contagem de FLOPs Completa
Usando a fórmula aproximada de FLOPs por token de um Transformer ($\approx 2 \times \text{num\_params}$ para o forward pass, mais custo adicional quadrático da atenção em função de `seq_len`), estime o custo total de FLOPs para processar um batch de `seq_len=1024` tokens com o maior modelo que você configurou no Exercício 23.4.

---

## Resumo

- **Pipeline completo**: `input_ids -> token embedding + positional embedding -> pilha de N blocos Transformer -> LayerNorm final -> LM head -> logits`
- **Invariante de shape**: `d_model` se mantém constante do embedding até a LayerNorm final; a única mudança de dimensão em todo o pipeline é a projeção final para `vocab_size`
- **Contagem de parâmetros**: dominada pelo termo $12 L d^2$ (pilha de blocos) quando o modelo escala — dobrar `d_model` custa ~4x mais parâmetros, dobrar `num_layers` custa ~2x mais
- **Weight tying**: compartilhar os pesos entre `token_embedding` e `lm_head` economiza $V \times d$ parâmetros e costuma melhorar a qualidade do modelo
- **Logits**: a saída bruta `[batch, seq_len, vocab_size]`, uma pontuação não normalizada para cada palavra do vocabulário em cada posição — o ponto de partida para calcular loss (Capítulo 26) ou gerar texto (Capítulos 31-32)
- **Causalidade preservada**: graças à máscara causal dentro de cada bloco, os logits em uma posição nunca dependem de tokens futuros

Próximo capítulo: **Tokenização** — antes de qualquer coisa entrar nesse pipeline, o texto bruto precisa virar uma sequência de IDs inteiros. Vamos entender como isso é feito.

---

**Próximo**: [Capítulo 24: Tokenização](24_tokenizacao.md)
