# Capítulo 26: Logits e Cross-Entropy Loss

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o que são logits e por que a última camada do modelo não aplica softmax diretamente
2. Converter logits em uma distribuição de probabilidade usando softmax
3. Calcular cross-entropy loss manualmente e entender sua conexão com maximum likelihood
4. Usar `nn.CrossEntropyLoss` corretamente, evitando o erro clássico de aplicar softmax duas vezes
5. Interpretar perplexidade como métrica intuitiva de qualidade do modelo de linguagem

---

## Por Que Isso Importa

Depois de passar o texto por embeddings, blocos transformer, atenção multi-cabeça e MLPs, o modelo produz um vetor de números por posição — um número para cada palavra possível do vocabulário. Esses números não significam nada por si só: podem ser `[2.3, -1.1, 5.7, 0.02, ...]`, positivos, negativos, arbitrariamente grandes. Como transformamos isso em "o modelo acha que a próxima palavra é X com 73% de confiança"? E, mais importante para treinar o modelo: como transformamos isso em um número único que dizemos ao otimizador "minimize isso"?

É aqui que entram logits, softmax e cross-entropy loss — o trio que converte a saída crua da rede em uma medida de erro que pode ser derivada e otimizada.

Um erro extremamente comum ao implementar isso do zero é aplicar softmax manualmente e depois passar o resultado para `nn.CrossEntropyLoss`, que já inclui um softmax (via `log_softmax`) internamente. O resultado não é um erro que trava o programa — é pior: o treinamento roda, a loss cai, mas converge para um modelo pior do que deveria, porque o gradiente está sendo calculado sobre uma composição errada de funções. Esse é exatamente o tipo de bug silencioso que consome horas de debugging quando você não entende profundamente o que cada peça faz matematicamente.

Também é neste capítulo que aparece a perplexidade, a métrica que todo paper de linguagem cita. Quando você lê "nosso modelo atingiu perplexidade 20 no conjunto de validação", isso tem um significado concreto e intuitivo: em média, o modelo está tão confuso quanto se tivesse que escolher uniformemente entre 20 opções para cada próximo token. Entender de onde vem esse número — literalmente `exp(loss)` — tira a mística da métrica e te dá uma ferramenta de diagnóstico real.

---

## Logits: A Saída Bruta da Última Camada

Depois do último bloco transformer, o modelo aplica uma projeção linear final que mapeia da dimensão de embedding `d_model` para o tamanho do vocabulário `vocab_size`:

$$\text{logits} = H W_{\text{head}}$$

Onde $H \in \mathbb{R}^{[batch, seq\_len, d\_model]}$ é a saída do último bloco transformer (já normalizada, com todas as camadas de atenção e MLP aplicadas), e $W_{\text{head}} \in \mathbb{R}^{[d\_model, vocab\_size]}$ é a matriz de projeção para o vocabulário (frequentemente compartilhada com a matriz de embedding de entrada — técnica chamada *weight tying*).

O resultado tem shape:

$$\text{logits} \in \mathbb{R}^{[batch, seq\_len, vocab\_size]}$$

Para cada posição da sequência, existe um vetor de tamanho `vocab_size` — um número real (positivo, negativo, qualquer magnitude) para cada palavra possível do vocabulário. Esses números são chamados **logits** porque, historicamente, representam o log-odds não normalizado de cada classe — mas na prática, o que importa é que eles são *comparáveis entre si*: um logit maior significa que o modelo atribui mais "peso" àquela palavra, mas os valores absolutos não têm significado probabilístico direto.

Um erro comum de iniciante é olhar para um logit de `5.7` e achar que isso "é uma probabilidade alta". Não é. Logits não são probabilidades — não somam 1, podem ser negativos, não têm limite superior. Eles são a entrada crua para a função que *vai* produzir probabilidades: o softmax.

---

## Softmax: De Logits para Probabilidades

O softmax converte um vetor de números reais arbitrários em uma distribuição de probabilidade válida — valores entre 0 e 1 que somam exatamente 1:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

Onde $z \in \mathbb{R}^V$ são os logits para uma posição, e $V$ é `vocab_size`.

Duas propriedades importantes:

1. **Exponenciação preserva ordem**: se $z_i > z_j$, então $\text{softmax}(z)_i > \text{softmax}(z)_j$. O token com maior logit continua sendo o mais provável.
2. **Exponenciação amplifica diferenças**: um logit ligeiramente maior que os outros pode virar uma probabilidade dominante, porque $e^x$ cresce rapidamente. Isso é o que dá ao modelo a capacidade de ser "confiante" quando tem certeza.

No caso de um language model, aplicamos softmax sobre a última dimensão (`vocab_size`), independentemente para cada posição da sequência:

```python
probs = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
```

Cada `probs[b, t, :]` é uma distribuição de probabilidade sobre todo o vocabulário — a resposta do modelo à pergunta "qual é o próximo token, dado tudo que veio antes até a posição t?".

---

## Cross-Entropy Loss: Medindo o Erro

Termos a distribuição predita pelo modelo (`probs`), mas precisamos comparar com a resposta certa: o token real que veio a seguir no texto. A resposta certa pode ser representada como uma distribuição "one-hot" — probabilidade 1 no token correto, 0 em todos os outros.

A cross-entropy entre a distribuição predita $p$ e a distribuição verdadeira $q$ (one-hot) é definida como:

$$H(q, p) = -\sum_{i=1}^{V} q_i \log(p_i)$$

Como $q$ é one-hot (vale 1 apenas no índice do token correto $y$, e 0 em todos os outros), a soma colapsa para um único termo:

$$\text{loss} = -\log(p_y)$$

Onde $p_y$ é a probabilidade que o modelo atribuiu ao token correto. Essa simplificação é o motivo pelo qual, na prática, calcular cross-entropy loss para classificação é simplesmente "pegar o log da probabilidade do rótulo correto e negar".

### Por Que $-\log(p_{\text{correto}})$?

Vale a pena entender de onde vem essa fórmula, e não apenas memorizá-la.

**Intuição da penalização.** Considere o gráfico de $-\log(p)$ para $p \in (0, 1]$:

- Quando $p_y \to 1$ (modelo tem certeza absoluta e acerta), $-\log(p_y) \to 0$. Loss quase zero — sem penalidade.
- Quando $p_y \to 0$ (modelo tem certeza absoluta e erra), $-\log(p_y) \to \infty$. Loss explode — penalidade catastrófica.
- Quando $p_y = 0.5$, $-\log(0.5) \approx 0.693$. Penalidade moderada.

Essa curvatura é exatamente o que queremos: um modelo que erra com baixa confiança (ex: atribuiu 30% ao token errado, mas distribuiu o resto razoavelmente) é penalizado pouco. Um modelo que erra com *alta confiança* (atribuiu 95% a um token errado) é penalizado com muito mais força que um erro proporcional simples (como diferença absoluta) capturaria. Isso incentiva o modelo a nunca ficar "cegamente confiante" em previsões erradas — um comportamento crucial durante o treinamento.

**Conexão com Maximum Likelihood Estimation (MLE).** Treinar um modelo de linguagem é, estatisticamente, tentar encontrar os parâmetros $\theta$ que maximizam a probabilidade dos dados observados:

$$\theta^* = \arg\max_\theta \prod_{t} p_\theta(x_t \mid x_{<t})$$

Produtos de números pequenos (probabilidades) são numericamente instáveis e difíceis de derivar, então aplicamos log (que é monotônico, então não muda o argmax):

$$\theta^* = \arg\max_\theta \sum_{t} \log p_\theta(x_t \mid x_{<t})$$

Como otimizadores em deep learning minimizam por convenção, negamos a expressão:

$$\theta^* = \arg\min_\theta \left(-\sum_{t} \log p_\theta(x_t \mid x_{<t})\right)$$

Essa é exatamente a soma de cross-entropy losses por posição. **Minimizar cross-entropy loss é matematicamente idêntico a maximizar a verossimilhança (likelihood) dos dados de treino sob o modelo.** Não é uma escolha arbitrária de "função de erro que parece razoável" — é a consequência direta de tratar o problema como estimação estatística de máxima verossimilhança.

### Cross-Entropy Batched: Média sobre Todas as Posições

Na prática, temos `batch_size × seq_len` predições simultâneas (uma por posição, por exemplo do batch). A loss total é a média (ou soma, dependendo da convenção) das cross-entropies individuais:

$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log p_{y_n}$$

Onde $N = batch\_size \times seq\_len$ é o número total de previsões de token no batch.

---

## Exemplo Numérico Manual

Vamos calcular a loss à mão para um caso minúsculo: vocabulário de tamanho 4, uma única posição de predição.

**Setup:**

```
Vocabulário: ["gato", "cachorro", "sofá", "correu"]
Token correto (target): "correu" (índice 3)

Logits do modelo: [1.2, 0.3, -0.5, 2.1]
```

**Passo 1 — Softmax:**

```
exp(1.2)  = 3.320
exp(0.3)  = 1.350
exp(-0.5) = 0.607
exp(2.1)  = 8.166

soma = 3.320 + 1.350 + 0.607 + 8.166 = 13.443

probs = [3.320/13.443, 1.350/13.443, 0.607/13.443, 8.166/13.443]
      = [0.247, 0.100, 0.045, 0.607]
```

Verificação: $0.247 + 0.100 + 0.045 + 0.607 = 0.999 \approx 1$ (erro de arredondamento).

**Passo 2 — Cross-entropy:**

O token correto é "correu", índice 3, com probabilidade predita $p_y = 0.607$.

```
loss = -log(0.607) = 0.499
```

**Interpretação:** o modelo atribuiu 60.7% de probabilidade ao token correto — uma predição razoavelmente boa, e a loss de 0.499 reflete isso (baixa, mas não zero). Se o modelo tivesse atribuído 99% de probabilidade ao token correto, a loss cairia para $-\log(0.99) \approx 0.01$. Se tivesse atribuído apenas 5%, a loss subiria para $-\log(0.05) \approx 3.0$ — quase 6x maior, ilustrando a penalização não-linear.

---

## Implementação Manual vs `nn.CrossEntropyLoss`

### Implementação Manual (para entender)

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[1.2, 0.3, -0.5, 2.1]])  # [1, vocab_size]
target = torch.tensor([3])  # índice do token correto

# Passo 1: softmax manual
probs = F.softmax(logits, dim=-1)

# Passo 2: pegar a probabilidade do token correto
p_correct = probs[0, target[0]]

# Passo 3: negativo do log
loss_manual = -torch.log(p_correct)
print(loss_manual)  # tensor(0.4991)
```

### `nn.CrossEntropyLoss` (o que se usa na prática)

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss_builtin = loss_fn(logits, target)
print(loss_builtin)  # tensor(0.4991) — mesmo valor
```

**Ponto crucial**: `nn.CrossEntropyLoss` espera **logits crus**, não probabilidades. Internamente, ela aplica `log_softmax` (uma versão numericamente estável de `log(softmax(x))`) seguido de `nll_loss` (negative log-likelihood). Você **nunca** deve aplicar softmax antes de passar para `nn.CrossEntropyLoss`.

---

## O Erro Clássico: Softmax Duplo

```python
# ERRADO — aplica softmax duas vezes
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, target)  # cross_entropy aplica softmax de novo!
```

O que acontece matematicamente: `F.cross_entropy` (e `nn.CrossEntropyLoss`) calculam `softmax(softmax(logits))`. Isso não gera um erro de execução — o código roda, produz um número, o treinamento "funciona" no sentido de que a loss diminui ao longo do tempo. Mas a superfície de otimização está distorcida: aplicar softmax duas vezes achata ainda mais as diferenças entre logits (porque a segunda aplicação recebe valores já entre 0 e 1, com soma 1, então os gradientes ficam menores e mais uniformes), resultando em convergência mais lenta e um modelo final pior.

Esse bug é particularmente insidioso porque **não quebra o código** — ele apenas degrada silenciosamente a qualidade do treinamento, e sem comparar contra uma implementação de referência, é fácil nunca perceber.

---

## Perplexidade

A perplexidade é definida como a exponencial da cross-entropy loss:

$$\text{PPL} = e^{\mathcal{L}}$$

Onde $\mathcal{L}$ é a loss média (em nats, ou seja, usando log natural).

**Por que essa métrica é tão usada?** Porque loss em nats (ex: "loss = 3.2") não tem uma interpretação intuitiva imediata. Perplexidade converte isso em algo palpável: **"quantas opções, em média, o modelo estava efetivamente confuso entre, ao prever cada token"**.

- Se PPL = 1: o modelo está sempre 100% confiante e sempre certo (caso ideal impossível na prática).
- Se PPL = `vocab_size`: o modelo está tão confuso quanto escolher uniformemente ao acaso entre todas as palavras do vocabulário (pior caso possível).
- Se PPL = 20: o modelo se comporta, em média, como se estivesse escolhendo uniformemente entre 20 opções plausíveis para cada próximo token — bem melhor que aleatório, mas longe de certeza.

Verificação matemática rápida: se o modelo atribuísse probabilidade uniforme $1/V$ a cada uma de $V$ palavras, a loss seria $-\log(1/V) = \log(V)$, e a perplexidade seria $e^{\log(V)} = V$ — exatamente o tamanho do vocabulário, confirmando a intuição de "confuso entre V opções".

---

## Experimento: Logits, Softmax e Loss na Prática

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Logits, Softmax, Cross-Entropy e Perplexidade")
print("=" * 70)

# ========== 1. CONFIGURAÇÃO ==========
vocab_size = 6
seq_len = 4
batch_size = 2

print(f"\nConfiguração:")
print(f"  vocab_size: {vocab_size}")
print(f"  seq_len: {seq_len}")
print(f"  batch_size: {batch_size}")

# ========== 2. SIMULANDO A SAÍDA DO MODELO (LOGITS) ==========
print("\n1. LOGITS (saída bruta da camada final)")
print("-" * 70)

logits = torch.randn(batch_size, seq_len, vocab_size) * 2
print(f"logits shape: {logits.shape}")
print(f"logits[0, 0, :] = {logits[0, 0, :]}")
print("Note: valores arbitrários, positivos e negativos, sem soma fixa.")

# ========== 3. TARGETS (tokens corretos) ==========
print("\n2. TARGETS (tokens corretos, shift-by-one da sequência real)")
print("-" * 70)

targets = torch.randint(0, vocab_size, (batch_size, seq_len))
print(f"targets shape: {targets.shape}")
print(f"targets =\n{targets}")

# ========== 4. SOFTMAX MANUAL ==========
print("\n3. SOFTMAX (convertendo logits em probabilidades)")
print("-" * 70)

probs = F.softmax(logits, dim=-1)
print(f"probs shape: {probs.shape}")
print(f"probs[0, 0, :] = {probs[0, 0, :]}")
print(f"soma de probs[0, 0, :] = {probs[0, 0, :].sum():.6f} (deve ser 1.0)")

# ========== 5. CROSS-ENTROPY MANUAL ==========
print("\n4. CROSS-ENTROPY MANUAL (posição a posição)")
print("-" * 70)

losses_manual = []
for b in range(batch_size):
    for t in range(seq_len):
        p_correct = probs[b, t, targets[b, t]]
        loss_bt = -torch.log(p_correct)
        losses_manual.append(loss_bt.item())

loss_manual_mean = sum(losses_manual) / len(losses_manual)
print(f"Losses individuais (primeiras 4): {[f'{l:.4f}' for l in losses_manual[:4]]}")
print(f"Loss média manual: {loss_manual_mean:.4f}")

# ========== 6. CROSS-ENTROPY COM nn.CrossEntropyLoss ==========
print("\n5. nn.CrossEntropyLoss (implementação de referência do PyTorch)")
print("-" * 70)

# CrossEntropyLoss espera [N, vocab_size] e target [N]
# Fazemos reshape: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
logits_flat = logits.view(-1, vocab_size)
targets_flat = targets.view(-1)

print(f"logits_flat shape: {logits_flat.shape}")
print(f"targets_flat shape: {targets_flat.shape}")

loss_fn = torch.nn.CrossEntropyLoss()
loss_builtin = loss_fn(logits_flat, targets_flat)
print(f"Loss (nn.CrossEntropyLoss): {loss_builtin.item():.4f}")

print(f"\nDiferença entre manual e builtin: {abs(loss_manual_mean - loss_builtin.item()):.8f}")
print("(Deve ser essencialmente zero — mesma matemática, implementações diferentes)")

# ========== 7. O ERRO DE SOFTMAX DUPLO ==========
print("\n6. DEMONSTRANDO O ERRO DE SOFTMAX DUPLO")
print("-" * 70)

loss_double_softmax = loss_fn(probs.view(-1, vocab_size), targets_flat)
print(f"Loss correta (logits crus):      {loss_builtin.item():.4f}")
print(f"Loss com softmax duplo (ERRADO): {loss_double_softmax.item():.4f}")
print("Note como o valor muda — o gradiente para backprop também seria diferente,")
print("distorcendo o treinamento sem quebrar o código.")

# ========== 8. PERPLEXIDADE ==========
print("\n7. PERPLEXIDADE")
print("-" * 70)

perplexity = torch.exp(loss_builtin)
print(f"Loss: {loss_builtin.item():.4f}")
print(f"Perplexidade: {perplexity.item():.4f}")
print(f"(vocab_size = {vocab_size}, perplexidade máxima possível ~= {vocab_size})")

# Caso extremo: modelo perfeito
print("\nCaso hipotético — modelo com certeza absoluta e sempre certo:")
loss_perfect = torch.tensor(0.0001)  # loss quase zero
print(f"  loss = {loss_perfect.item():.4f} -> perplexidade = {torch.exp(loss_perfect).item():.4f}")

# Caso extremo: modelo uniforme (aleatório)
print("Caso hipotético — modelo uniforme (probabilidade 1/vocab_size para tudo):")
loss_uniform = torch.tensor(float(torch.log(torch.tensor(vocab_size, dtype=torch.float32))))
print(f"  loss = {loss_uniform.item():.4f} -> perplexidade = {torch.exp(loss_uniform).item():.4f}")
print(f"  (perplexidade == vocab_size, como esperado)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Saída esperada (aproximada, com seed 42):

```
Loss (nn.CrossEntropyLoss): ~1.9x
Perplexidade: ~6.9x
```

---

## Erros Comuns

### Erro 1: Softmax Duplo

```python
# ERRADO
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, target)  # aplica softmax de novo internamente!

# CERTO
loss = F.cross_entropy(logits, target)  # passa logits crus
```

### Erro 2: Shape Errado em `CrossEntropyLoss`

`nn.CrossEntropyLoss` espera `logits` no formato `[N, num_classes]` e `target` no formato `[N]` (índices inteiros, não one-hot). Para sequências, é preciso "achatar" o batch e a dimensão temporal:

```python
# ERRADO — passa tensores 3D diretamente
logits = model(x)  # [batch, seq_len, vocab_size]
loss = F.cross_entropy(logits, targets)  # ERRO: shapes incompatíveis

# CERTO — achata batch e seq_len em uma única dimensão
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # [batch*seq_len, vocab_size]
    targets.view(-1)              # [batch*seq_len]
)
```

### Erro 3: Target como One-Hot em Vez de Índices

```python
# ERRADO — CrossEntropyLoss não espera one-hot
target_onehot = torch.zeros(vocab_size)
target_onehot[3] = 1.0
loss = F.cross_entropy(logits, target_onehot)  # ERRO de shape/tipo

# CERTO — target é o índice inteiro do token correto
target = torch.tensor([3])
loss = F.cross_entropy(logits, target)
```

### Erro 4: Calcular Perplexidade a Partir de Loss em Base Errada

```python
# ERRADO — se a loss foi calculada com log base 2, exp() está errado
loss_log2 = -torch.log2(p_correct)
ppl_errada = torch.exp(loss_log2)  # matematicamente inconsistente

# CERTO — cross_entropy do PyTorch usa log natural (ln), então exp() é consistente
loss = F.cross_entropy(logits, target)  # usa log natural internamente
ppl = torch.exp(loss)
```

---

## Exercícios

### Exercício 26.1: Softmax Manual
Dado `logits = [2.0, 1.0, 0.1]`, calcule o softmax manualmente (com calculadora ou papel) e verifique que a soma dá 1.

### Exercício 26.2: Cross-Entropy de um Único Exemplo
Com `logits = [0.5, 2.3, -1.0, 0.2]` e `target = 1`, calcule a loss manualmente (softmax + `-log(p_correto)`).

### Exercício 26.3: Implementar Cross-Entropy do Zero
Escreva uma função `minha_cross_entropy(logits, target)` que replica `F.cross_entropy` sem usar nenhuma função de loss do PyTorch (pode usar `torch.exp`, `torch.log`, `torch.sum`).

### Exercício 26.4: Perplexidade em Diferentes Cenários
Calcule a perplexidade para losses de `0.1`, `1.0`, `3.0` e `5.0`. Compare com o tamanho de um vocabulário de 50.000 tokens (típico de um LLM real) — o que isso te diz sobre modelos bem treinados?

### Exercício 26.5: Detectar o Bug de Softmax Duplo
Você recebe um trecho de código de um colega com a loss estranhamente alta e não convergindo bem. Encontre e corrija o bug:

```python
def training_step(model, x, y):
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    loss = F.cross_entropy(probs.view(-1, probs.size(-1)), y.view(-1))
    return loss
```

---

## Gabarito

### Exercício 26.1: Softmax Manual
```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(probs)
# Manual: exp([2.0, 1.0, 0.1]) = [7.389, 2.718, 1.105]
# soma = 11.212
# probs = [0.659, 0.242, 0.099]
```

### Exercício 26.2: Cross-Entropy de um Único Exemplo
```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[0.5, 2.3, -1.0, 0.2]])
target = torch.tensor([1])

probs = F.softmax(logits, dim=-1)
print(f"probs: {probs}")

p_correct = probs[0, 1]
loss = -torch.log(p_correct)
print(f"loss manual: {loss.item():.4f}")

loss_builtin = F.cross_entropy(logits, target)
print(f"loss builtin: {loss_builtin.item():.4f}")  # deve bater
```

### Exercício 26.3: Implementar Cross-Entropy do Zero
```python
import torch

def minha_cross_entropy(logits, target):
    # logits: [N, vocab_size], target: [N]
    exp_logits = torch.exp(logits)
    soma = exp_logits.sum(dim=-1, keepdim=True)
    probs = exp_logits / soma

    N = logits.size(0)
    p_correct = probs[torch.arange(N), target]
    loss = -torch.log(p_correct)
    return loss.mean()

logits = torch.tensor([[1.2, 0.3, -0.5, 2.1], [0.1, 0.2, 0.3, 0.4]])
target = torch.tensor([3, 1])

print(minha_cross_entropy(logits, target))
print(torch.nn.functional.cross_entropy(logits, target))  # deve bater
```

### Exercício 26.4: Perplexidade em Diferentes Cenários
```python
import torch

for loss_val in [0.1, 1.0, 3.0, 5.0]:
    ppl = torch.exp(torch.tensor(loss_val))
    print(f"loss={loss_val} -> perplexidade={ppl.item():.2f}")

# loss=0.1 -> ppl=1.11   (modelo quase certo)
# loss=1.0 -> ppl=2.72   (confuso entre ~3 opções)
# loss=3.0 -> ppl=20.09  (confuso entre ~20 opções)
# loss=5.0 -> ppl=148.41 (bem confuso)
#
# Comparado a vocab_size=50000: um modelo bem treinado com ppl~20-30
# está extremamente longe do "aleatório" (ppl=50000), mas ainda tem
# incerteza real sobre qual token vem a seguir — o que é esperado,
# já que a linguagem natural é genuinamente ambígua em muitos pontos.
```

### Exercício 26.5: Detectar o Bug de Softmax Duplo
```python
# Bug: softmax é aplicado manualmente, e F.cross_entropy aplica de novo.

# Correção:
def training_step(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return loss
```

---

## Desafios Avançados (Opcionais)

### Fixação 26.1: Label Smoothing
Implemente cross-entropy com label smoothing (em vez de target one-hot puro, distribua uma pequena fração de probabilidade entre as classes erradas). Compare a loss e o gradiente com a versão sem smoothing.

### Fixação 26.2: Estabilidade Numérica do Softmax
Implemente softmax "ingênuo" (`exp(x) / sum(exp(x))`) e teste com logits grandes (ex: `[1000, 1001, 1002]`). Observe o overflow. Implemente a versão estável (subtraindo o máximo antes de exponenciar) e compare.

### Fixação 26.3: Cross-Entropy com Padding Mask
Em batches com sequências de tamanhos diferentes, tokens de padding não devem contribuir para a loss. Implemente uma versão de cross-entropy que ignora posições marcadas com um `ignore_index` (dica: veja o parâmetro `ignore_index` de `nn.CrossEntropyLoss`).

### Fixação 26.4: Relação entre Loss e Acurácia Top-1
Para um mesmo batch, calcule tanto a cross-entropy loss quanto a acurácia (fração de vezes em que `argmax(probs) == target`). Varie a "confiança" dos logits sintéticos e observe como as duas métricas se relacionam (ou não).

### Fixação 26.5: Perplexidade por Token vs por Caractere
Se você trocasse a tokenização de BPE (subpalavras) para caracteres individuais, a perplexidade mudaria mesmo com a "mesma qualidade" de modelo? Raciocine sobre por que perplexidades de modelos com vocabulários diferentes não são diretamente comparáveis.

---

## Resumo

- **Logits**: saída bruta da última camada linear, shape `[batch, seq_len, vocab_size]`, sem interpretação probabilística direta
- **Softmax**: converte logits em distribuição de probabilidade válida (soma 1, todos positivos)
- **Cross-entropy loss**: $-\log(p_{\text{correto}})$ — penaliza fortemente erros confiantes, pouco erros incertos
- **MLE**: minimizar cross-entropy é equivalente a maximizar a verossimilhança dos dados sob o modelo
- **`nn.CrossEntropyLoss`**: espera logits crus (não probabilidades) — aplicar softmax antes é um bug silencioso comum
- **Perplexidade**: `exp(loss)`, interpretável como "número efetivo de opções que confundem o modelo"

Próximo capítulo: **Backpropagação e Gradientes** — como o erro medido pela loss se propaga de volta por toda a rede para atualizar cada parâmetro.

---

**Próximo**: [Capítulo 27: Backpropagação e Gradientes](27_backpropagacao.md)
