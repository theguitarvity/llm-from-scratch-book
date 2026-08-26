# Capítulo 15: Causal Masking

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Explicar por que modelos autoregressivos não podem ter acesso a tokens futuros durante o treinamento
2. Descrever o problema de vazamento de informação (information leakage) que ocorre sem máscara causal
3. Construir uma máscara triangular inferior com `torch.tril`/`torch.triu`
4. Aplicar `-inf` corretamente antes do softmax (e explicar por que não se usa zero)
5. Diferenciar causal masking (usado em decoders) de atenção sem máscara (usada em encoders)

---

## Por Que Isso Importa

Imagine treinar um modelo para prever a próxima palavra de uma frase, mas por engano você deixa ele "espiar" a própria resposta durante o treinamento. O loss despenca rapidamente, os números parecem ótimos, você comemora — e então, na hora de gerar texto de verdade (onde não existe resposta para espiar), o modelo produz um resultado que parece aleatório ou repetitivo. Se você já treinou um modelo de linguagem e viu esse padrão específico — loss de treino artificialmente baixo, geração de texto ruim — é quase certeza de que o culpado é vazamento de informação por falta de causal masking.

O motivo é conceitualmente simples, mas fácil de esquecer na hora de implementar: um modelo autoregressivo, como GPT, é treinado para prever o token na posição $i+1$ **usando apenas informação disponível até a posição $i$**. Isso espelha exatamente como o modelo vai ser usado em produção: quando você pede para o modelo gerar texto, ele produz um token de cada vez, e cada novo token só pode depender dos tokens já gerados — nunca dos que ainda vão vir, porque eles simplesmente não existem ainda.

O problema é que a self-attention, como vimos nos capítulos anteriores, **naturalmente conecta cada posição com todas as outras posições da sequência**, inclusive as futuras. Sem uma intervenção explícita, ao processar a posição 2 de uma frase de 5 tokens, o mecanismo de atenção "veria" as posições 3, 4 e 5 durante o treinamento — informação que, na hora da geração real, simplesmente não estará disponível. Treinar com esse vazamento produz um modelo que aprendeu a "colar" em vez de aprender a prever, e que falha catastroficamente assim que colocado para gerar texto de verdade. Causal masking é o mecanismo que fecha essa brecha.

---

## O Problema do Vazamento de Informação

### Setup: predição autoregressiva

Em um modelo de linguagem autoregressivo, o objetivo de treinamento é maximizar a probabilidade de cada token dado todos os anteriores:

$$P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, \ldots, x_{i-1})$$

Ou seja, para prever $x_i$, o modelo só pode usar $x_1, \ldots, x_{i-1}$ — nunca $x_i$ em diante. Isso é chamado de **fatoração autoregressiva** ou "left-to-right".

### Por que a self-attention vaza informação sem máscara

Sem intervenção, quando calculamos $\text{scores} = QK^T$ para uma sequência inteira de uma vez (o que fazemos por eficiência — processar a sequência inteira em paralelo, em vez de token por token), a posição $i$ acaba com um score não-nulo contra **todas** as posições $j$, inclusive $j > i$ (futuras). O softmax então distribui peso de atenção também para essas posições futuras, e o output da posição $i$ passa a carregar informação de tokens que, na hora de prever $x_i$, não deveriam existir.

Durante o treinamento, isso é catastrófico: o modelo aprende a "trapacear", usando a resposta certa (o próprio token futuro, ou pistas fortes dele) para "prever" ele mesmo. O loss de treino cai rapidamente e de forma enganosa, mas o modelo não aprendeu nada de generalizável — porque na hora de gerar texto novo, essa informação futura não existirá.

### Visualizando o vazamento

Considere a sequência "O gato dormia" (3 tokens). Sem máscara, a matriz de attention weights poderia ter esta forma (valores ilustrativos):

```
             K="O"   K="gato"  K="dormia"
Q="O"        0.5      0.3       0.2   <- veio de "dormia", que ainda não devia existir!
Q="gato"     0.2      0.5       0.3   <- veio de "dormia" também!
Q="dormia"   0.3      0.3       0.4
```

Ao processar a Query "O" (primeira posição), o modelo já está "olhando" 0.2 de peso para "dormia" — um token que, no momento de prever o que vem depois de "O", ainda não foi gerado e não deveria ser conhecido.

---

## Construindo a Máscara Triangular Inferior

### A ideia

Queremos permitir que a posição $i$ preste atenção apenas em posições $j \le i$ (incluindo ela mesma), e bloquear completamente $j > i$. Isso corresponde a uma **matriz triangular inferior** de 1s (posições permitidas) e 0s (posições bloqueadas):

$$\text{mask}[i,j] = \begin{cases} 1 & \text{se } j \le i \\ 0 & \text{se } j > i \end{cases}$$

Para uma sequência de 4 tokens, a máscara fica:

```
mask = [
    [1, 0, 0, 0],   # posição 0 só vê a si mesma
    [1, 1, 0, 0],   # posição 1 vê 0 e 1
    [1, 1, 1, 0],   # posição 2 vê 0, 1, 2
    [1, 1, 1, 1]    # posição 3 vê todas (0, 1, 2, 3)
]
```

### Construindo com torch.tril

```python
import torch

n = 4
mask = torch.tril(torch.ones(n, n))
print(mask)
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

`torch.tril` ("triangular lower") zera tudo acima da diagonal principal, mantendo a diagonal e abaixo dela. A alternativa equivalente usando `torch.triu` ("triangular upper") seria construir a máscara do que **deve ser bloqueado** e depois inverter:

```python
mask_bloqueio = torch.triu(torch.ones(n, n), diagonal=1)  # 1s acima da diagonal (excluindo ela)
print(mask_bloqueio)
# tensor([[0., 1., 1., 1.],
#         [0., 0., 1., 1.],
#         [0., 0., 0., 1.],
#         [0., 0., 0., 0.]])
```

Repare o parâmetro `diagonal=1`: isso exclui a diagonal principal do triângulo superior, garantindo que a posição $i$ ainda possa ver a si mesma ($j = i$ é permitido, não bloqueado).

---

## Por Que -inf e Não Zero

### A tentação (errada) de usar zero

Um erro intuitivo é pensar: "para bloquear uma posição, basta multiplicar seu score por zero". Isso está errado, e a razão está em como o softmax funciona.

```python
# ERRADO: zerar os scores bloqueados
scores_mascarados = scores * mask  # zera onde mask == 0
attn = torch.softmax(scores_mascarados, dim=-1)  # AINDA distribui peso para as posições "zeradas"!
```

O problema: $e^0 = 1$, não zero. Quando você zera um score e depois aplica softmax, essa posição ainda recebe $e^0 = 1$ no numerador — ela não desaparece da distribuição, ela só passa a competir como se tivesse tido um score neutro (nem alto, nem baixo). Isso significa que posições futuras continuariam recebendo peso de atenção positivo, só que artificialmente reduzido — o vazamento de informação persiste, apenas atenuado.

### A solução correta: -inf antes do softmax

Queremos que, depois do softmax, o peso dessas posições seja **exatamente zero** — não "pequeno", zero de verdade. Para isso, atribuímos $-\infty$ ao score antes do softmax:

$$\text{scores\_mascarados}[i,j] = \begin{cases} \text{scores}[i,j] & \text{se } j \le i \\ -\infty & \text{se } j > i \end{cases}$$

Por quê isso funciona: $e^{-\infty} = 0$ exatamente. Quando o softmax exponencia um score de $-\infty$, o resultado é literalmente zero, e essa posição contribui com zero tanto no numerador quanto — de forma desprezível — no denominador. O resultado é uma distribuição de probabilidade que nunca atribui peso a posições futuras, garantido matematicamente, não apenas aproximadamente.

```python
# CERTO: usar -inf, não zero
scores_mascarados = scores.masked_fill(mask == 0, float('-inf'))
attn = torch.softmax(scores_mascarados, dim=-1)  # peso EXATAMENTE zero nas posições bloqueadas
```

### Cuidado com NaN em linhas totalmente mascaradas

Um detalhe sutil: se uma linha inteira de `scores_mascarados` fosse `-inf` (o que não acontece em causal masking padrão, já que a diagonal sempre é permitida, mas pode acontecer em outros tipos de máscara, como padding mask mal configurada), o softmax produziria `0/0 = NaN`. Vale sempre garantir que cada posição tenha pelo menos um valor não-mascarado — em causal masking isso é automático, porque `mask[i,i] = 1` sempre.

---

## Exemplo Numérico Manual

### Setup

Scores brutos (já escalados por $\sqrt{d_k}$) para uma sequência de 3 tokens:

```
scores = [
    [1.0, 2.0, 0.5],
    [0.5, 1.5, 2.0],
    [2.0, 1.0, 0.0]
]
```

### Passo 1: Aplicar a máscara causal

```
mask = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]
]

scores_mascarados = [
    [1.0,  -inf, -inf],
    [0.5,  1.5,  -inf],
    [2.0,  1.0,  0.0]
]
```

### Passo 2: Softmax linha 0

```
scores[0,:] = [1.0, -inf, -inf]
exp(1.0) = 2.718, exp(-inf) = 0, exp(-inf) = 0
soma = 2.718
softmax = [2.718/2.718, 0/2.718, 0/2.718] = [1.0, 0.0, 0.0]
```

A posição 0 presta 100% de atenção em si mesma — correto, ela não tem mais ninguém para olhar (é a primeira posição).

### Passo 3: Softmax linha 1

```
scores[1,:] = [0.5, 1.5, -inf]
exp(0.5) = 1.649, exp(1.5) = 4.482, exp(-inf) = 0
soma = 6.131
softmax = [1.649/6.131, 4.482/6.131, 0/6.131] = [0.269, 0.731, 0.0]
```

A posição 1 distribui atenção entre si mesma e a posição 0, mas **zero** para a posição 2 (futura).

### Passo 4: Softmax linha 2

```
scores[2,:] = [2.0, 1.0, 0.0]  (nenhum -inf, porque posição 2 vê tudo)
exp(2.0) = 7.389, exp(1.0) = 2.718, exp(0.0) = 1.0
soma = 11.107
softmax = [7.389/11.107, 2.718/11.107, 1.0/11.107] = [0.665, 0.245, 0.090]
```

A última posição, tendo acesso a toda a sequência anterior, distribui atenção livremente entre todas as três.

---

## Causal Mask (Decoder) vs Sem Máscara (Encoder)

### Encoder: atenção bidirecional

Em arquiteturas encoder (como BERT), o objetivo não é prever o próximo token, mas construir uma representação rica de uma sequência **completa e já conhecida** (por exemplo, para classificação ou extração de features). Não há razão para restringir o fluxo de informação: cada posição pode e deve prestar atenção em todas as outras, inclusive posições "futuras" em relação à ordem do texto. Essa é a chamada **atenção bidirecional** — sem máscara causal.

### Decoder: atenção causal

Em arquiteturas decoder (como GPT, que é o foco deste livro), o objetivo é gerar texto token a token, e o treinamento precisa espelhar exatamente essa restrição. Por isso, decoders sempre usam causal masking em suas camadas de self-attention.

| Aspecto | Encoder (bidirecional) | Decoder (causal) |
|---|---|---|
| Máscara | Nenhuma (ou só padding) | Triangular inferior + padding |
| Cada posição vê | Toda a sequência | Apenas posições $\le i$ |
| Uso típico | Classificação, embeddings de sentença | Geração de texto autoregressiva |
| Exemplo | BERT | GPT, LLaMA |
| Este livro usa | — | Sim (LLMScratch é decoder-only) |

Como o LLMScratch, o modelo construído ao longo deste livro, é uma arquitetura decoder-only (estilo GPT), **todas** as camadas de self-attention que implementaremos a partir do Capítulo 16 usarão causal masking.

---

## Diagrama ASCII da Matriz de Máscara

Para uma sequência de 6 tokens, a máscara causal completa (1 = permitido, . = bloqueado):

```
        j=0  j=1  j=2  j=3  j=4  j=5
i=0  [   1    .    .    .    .    .  ]
i=1  [   1    1    .    .    .    .  ]
i=2  [   1    1    1    .    .    .  ]
i=3  [   1    1    1    1    .    .  ]
i=4  [   1    1    1    1    1    .  ]
i=5  [   1    1    1    1    1    1  ]
```

O padrão triangular é a "assinatura visual" de qualquer modelo autoregressivo — sempre que você vir esse formato numa visualização de atenção, sabe que está olhando para um decoder causal.

---

## Experimento: Implementando e Verificando Causal Masking

```python
import torch

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Causal Masking")
print("=" * 70)

# ========== CONSTRUINDO A MÁSCARA ==========
print("\n1. CONSTRUINDO A MÁSCARA TRIANGULAR")
print("-" * 70)

n = 5
mask = torch.tril(torch.ones(n, n))
print(f"mask shape: {mask.shape}")
print(f"mask =\n{mask}")

# ========== SEM MÁSCARA: VAZAMENTO ==========
print("\n2. SEM MÁSCARA (vazamento de informação)")
print("-" * 70)

d = 8
X = torch.randn(n, d)
W_q = torch.randn(d, d) * 0.3
W_k = torch.randn(d, d) * 0.3
W_v = torch.randn(d, d) * 0.3

Q = X @ W_q
K = X @ W_k
V = X @ W_v

scores = Q @ K.T / (d ** 0.5)
attn_sem_mascara = torch.softmax(scores, dim=-1)

print("attention_weights SEM máscara (posição 0 deveria só ver ela mesma):")
print(attn_sem_mascara)
print(f"\nPosição 0 dá peso {attn_sem_mascara[0, 1:].sum().item():.4f} para posições FUTURAS (1,2,3,4)")
print("Isso é VAZAMENTO — não deveria acontecer em um decoder causal!")

# ========== COM MÁSCARA: ERRADO (usando zero) ==========
print("\n3. TENTATIVA ERRADA: mascarar multiplicando por zero")
print("-" * 70)

scores_zerados = scores * mask
attn_errado = torch.softmax(scores_zerados, dim=-1)
print("attention_weights usando zero (ERRADO):")
print(attn_errado)
print(f"\nPosição 0 AINDA dá peso {attn_errado[0, 1:].sum().item():.4f} para posições futuras!")
print("Zero vira exp(0)=1, não elimina a contribuição.")

# ========== COM MÁSCARA: CORRETO (usando -inf) ==========
print("\n4. FORMA CORRETA: masked_fill com -inf")
print("-" * 70)

scores_mascarados = scores.masked_fill(mask == 0, float('-inf'))
print("scores após masked_fill (linha 0, deve ter -inf a partir da posição 1):")
print(scores_mascarados[0])

attn_correto = torch.softmax(scores_mascarados, dim=-1)
print("\nattention_weights COM máscara causal correta:")
print(attn_correto)
print(f"\nPosição 0 dá peso {attn_correto[0, 1:].sum().item():.10f} para posições futuras")
print("Exatamente zero, como esperado!")

# ========== VERIFICANDO A PROPRIEDADE TRIANGULAR ==========
print("\n5. VERIFICANDO QUE O RESULTADO RESPEITA A CAUSALIDADE")
print("-" * 70)

for i in range(n):
    peso_futuro = attn_correto[i, i+1:].sum().item() if i < n - 1 else 0.0
    soma_linha = attn_correto[i].sum().item()
    print(f"posição {i}: peso em posições futuras = {peso_futuro:.8f}, soma da linha = {soma_linha:.6f}")

# ========== OUTPUT FINAL ==========
print("\n6. OUTPUT FINAL COM CAUSAL MASKING")
print("-" * 70)

output = attn_correto @ V
print(f"output shape: {output.shape}")
print(f"output =\n{output}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Multiplicar por zero em vez de usar -inf

```python
# Errado — não elimina o vazamento, apenas o atenua
scores_mascarados = scores * mask
attn = torch.softmax(scores_mascarados, dim=-1)

# Certo
scores_mascarados = scores.masked_fill(mask == 0, float('-inf'))
attn = torch.softmax(scores_mascarados, dim=-1)
```

### Erro 2: Construir a máscara invertida

```python
# Errado — tril() dá 1 onde é PERMITIDO, não onde deve ser bloqueado.
# Usar mask == 1 no masked_fill bloquearia o que deveria ser visível!
mask = torch.tril(torch.ones(n, n))
scores_mascarados = scores.masked_fill(mask == 1, float('-inf'))  # inverte a lógica!

# Certo — bloquear onde mask == 0 (fora do triângulo inferior)
scores_mascarados = scores.masked_fill(mask == 0, float('-inf'))
```

### Erro 3: Esquecer de expandir a máscara para o shape do batch

```python
# Errado — máscara [n, n] aplicada diretamente pode não broadcastar
# corretamente quando scores tem shape [batch, num_heads, n, n]
mask = torch.tril(torch.ones(n, n))  # [n, n]
scores = torch.randn(2, 4, n, n)     # [batch, heads, n, n]
scores.masked_fill(mask == 0, float('-inf'))  # funciona por broadcasting, MAS confirme sempre

# Mais seguro: verificar explicitamente o broadcasting, ou usar bool
mask_bool = torch.tril(torch.ones(n, n)).bool()  # [n, n], broadcast automático em batch/heads
scores_mascarados = scores.masked_fill(~mask_bool, float('-inf'))
```

---

## Exercícios

### Exercício 15.1: Construindo a Máscara
Construa a máscara causal para `n=6` usando `torch.tril` e também usando `torch.triu` (com inversão). Confirme que os dois métodos produzem o mesmo resultado.

### Exercício 15.2: Detectando Vazamento
Escreva uma função `tem_vazamento(attn_weights)` que recebe uma matriz de attention weights `[n, n]` e retorna `True` se algum peso acima da diagonal principal for maior que um pequeno epsilon (por exemplo, `1e-6`).

### Exercício 15.3: Zero vs -inf
Compare numericamente, para a mesma matriz de scores, o resultado do softmax usando mascaramento por zero vs por -inf. Quantifique o "vazamento residual" no caso do zero.

### Exercício 15.4: Máscara para Sequência com Padding
Além da causal mask, sequências em batch frequentemente têm padding (tokens de preenchimento sem significado). Construa uma máscara combinada que bloqueie tanto posições futuras quanto posições de padding (assuma que os últimos 2 tokens de uma sequência de 6 são padding).

### Exercício 15.5: Encoder vs Decoder
Para cada um dos seguintes casos de uso, decida se a atenção deveria ser causal (decoder) ou bidirecional (encoder): (a) gerar a próxima frase de uma história; (b) classificar o sentimento de uma review já completa; (c) prever a palavra mascarada no meio de uma frase (estilo BERT); (d) um chatbot respondendo token a token.

---

## Gabarito

### Exercício 15.1: Construindo a Máscara
```python
import torch

n = 6
mask_tril = torch.tril(torch.ones(n, n))
mask_triu_invertida = 1 - torch.triu(torch.ones(n, n), diagonal=1)

print(torch.equal(mask_tril, mask_triu_invertida))  # True
```

### Exercício 15.2: Detectando Vazamento
```python
import torch

def tem_vazamento(attn_weights, epsilon=1e-6):
    n = attn_weights.shape[-1]
    mask_futuro = torch.triu(torch.ones(n, n), diagonal=1).bool()
    peso_futuro = attn_weights[..., mask_futuro]
    return bool((peso_futuro > epsilon).any())

attn_ok = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
attn_vazado = torch.tensor([[0.8, 0.2], [0.5, 0.5]])

print(tem_vazamento(attn_ok))      # False
print(tem_vazamento(attn_vazado))  # True
```

### Exercício 15.3: Zero vs -inf
```python
import torch

torch.manual_seed(0)
n = 4
scores = torch.randn(n, n)
mask = torch.tril(torch.ones(n, n))

attn_zero = torch.softmax(scores * mask, dim=-1)
attn_inf = torch.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)

vazamento_zero = attn_zero[0, 1:].sum().item()
vazamento_inf = attn_inf[0, 1:].sum().item()

print(f"Vazamento com zero: {vazamento_zero:.4f}")
print(f"Vazamento com -inf: {vazamento_inf:.10f}")
```

### Exercício 15.4: Máscara para Sequência com Padding
```python
import torch

n = 6
n_padding = 2  # últimos 2 tokens são padding

causal_mask = torch.tril(torch.ones(n, n)).bool()

padding_mask = torch.ones(n, dtype=torch.bool)
padding_mask[-n_padding:] = False  # False = é padding, deve ser bloqueado
padding_mask_2d = padding_mask.unsqueeze(0).expand(n, n)  # bloqueia colunas de padding

mascara_final = causal_mask & padding_mask_2d
print(mascara_final)
```

### Exercício 15.5: Encoder vs Decoder
```
(a) Causal (decoder) — gerar texto token a token não pode ver o futuro.
(b) Bidirecional (encoder) — a review inteira já existe, pode ser processada de uma vez.
(c) Bidirecional (encoder) — BERT usa mascaramento de token, não causal masking; o contexto dos dois lados ajuda a prever a palavra mascarada.
(d) Causal (decoder) — cada token da resposta só pode depender dos tokens anteriores já gerados.
```

---

## Desafios Avançados (Opcionais)

### Fixação 15.1: Custo de Recriar a Máscara
Meça o custo de recriar `torch.tril(torch.ones(n, n))` a cada forward pass vs registrar a máscara como um buffer fixo (`register_buffer`) calculado uma única vez. Para `n=1024`, qual é a diferença de tempo em 1000 chamadas?

### Fixação 15.2: Máscara com float16/bfloat16
Investigue o que acontece ao usar `float('-inf')` em tensores de precisão reduzida (`float16`). Existe risco de `NaN` no softmax? Pesquise por que implementações reais costumam usar um valor grande e finito (como `-1e9`) em vez de `-inf` nesses casos.

### Fixação 15.3: Sliding Window Causal Mask
Implemente uma variante da máscara causal em que cada posição só pode ver as últimas $k$ posições anteriores (uma "janela deslizante" causal), em vez de todo o histórico. Compare com a máscara triangular completa.

### Fixação 15.4: Verificando Gradientes Através da Máscara
Faça um `backward()` através de uma atenção com causal masking e confirme que nenhum gradiente flui para as posições de Key/Value que foram mascaradas (ou seja, que `-inf` de fato bloqueia tanto o forward quanto o backward corretamente).

### Fixação 15.5: Prefix-LM (Máscara Híbrida)
Algumas arquiteturas usam uma máscara híbrida: bidirecional para um "prefixo" (por exemplo, uma pergunta ou contexto) e causal para o restante (a resposta sendo gerada). Implemente essa máscara híbrida para uma sequência onde os primeiros 3 tokens formam o prefixo bidirecional e o restante é gerado causalmente.

---

## Resumo

- **Vazamento de informação**: sem máscara, a self-attention conecta cada posição com posições futuras, invalidando o treinamento autoregressivo
- **Máscara triangular inferior**: `torch.tril(torch.ones(n,n))` marca exatamente quais posições cada Query pode ver
- **-inf, nunca zero**: aplicar `-inf` antes do softmax garante peso exatamente zero; multiplicar por zero apenas atenua o vazamento sem eliminá-lo
- **masked_fill é a ferramenta certa**: `scores.masked_fill(mask == 0, float('-inf'))` é o padrão de implementação
- **Causal (decoder) vs bidirecional (encoder)**: decoders autoregressivos (GPT, LLaMA) sempre usam causal masking; encoders (BERT) tipicamente não usam
- **A diagonal nunca é mascarada**: toda posição sempre pode ver a si mesma, garantindo que nenhuma linha do softmax fique inteiramente em `-inf`

Próximo capítulo: **Self-Attention Completo** — juntando Q/K/V, scores, scaling, softmax, causal masking e weighted sum em uma única implementação coesa.

---

**Próximo**: [Capítulo 16: Self-Attention Completo](16_self_attention.md)
