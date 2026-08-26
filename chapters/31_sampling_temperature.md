# Capítulo 31: Sampling, Temperature e Top-K/P

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender a diferença entre greedy decoding e sampling estocástico
2. Aplicar temperature para controlar aleatoriedade na geração
3. Implementar top-k sampling do zero
4. Implementar top-p (nucleus) sampling do zero
5. Escolher a estratégia de amostragem certa para cada caso de uso

---

## Por Que Isso Importa

Depois do forward pass de uma LLM, você tem um vetor de logits — um número por token do vocabulário, indicando "quão provável" o modelo acha que aquele token vem em seguida. A pergunta que este capítulo responde é enganosamente simples: **dado esse vetor de logits, qual token eu realmente escolho?**

A resposta ingênua é "o de maior probabilidade". Isso se chama *greedy decoding*, e é determinístico: o mesmo prompt sempre produz a mesma saída. Parece ótimo, mas na prática produz texto estranhamente repetitivo e monótono. Peça para um modelo em modo greedy continuar "Era uma vez..." e ele tende a cair em loops — "...uma vez, uma vez, uma vez" — ou a sempre escolher o caminho mais "seguro" e genérico, porque o token mais provável no passo 1 nem sempre leva à melhor frase completa.

Se você já usou ChatGPT ou Claude e pediu a mesma pergunta duas vezes, provavelmente recebeu respostas diferentes. Isso não é acidente nem bug — é sampling estocástico deliberado, ajustado por um parâmetro de **temperature** e por filtros como **top-k** e **top-p**. Esses três mecanismos são o painel de controle entre "o modelo sempre responde igual e sem graça" e "o modelo delira e produz texto sem nexo". Entender exatamente como eles transformam a distribuição de probabilidade é entender por que a API da OpenAI ou da Anthropic expõe parâmetros chamados `temperature`, `top_p` e `top_k` — e por que ajustá-los muda tanto o comportamento do modelo.

Este capítulo constrói cada uma dessas estratégias do zero, com PyTorch puro, para que você veja exatamente como cada uma transforma um vetor de logits em uma escolha de token.

---

## Greedy Decoding: O Ponto de Partida

Greedy decoding é a estratégia mais simples possível: em cada passo, escolha o token com maior probabilidade (equivalentemente, maior logit, já que softmax preserva ordem).

$$\text{token}_t = \arg\max_i \; \text{logits}[i]$$

Em código:

```python
next_token = torch.argmax(logits, dim=-1)
```

**Vantagens**: determinístico (reprodutível), simples, rápido.

**Desvantagens**:
- Repetição: uma vez que o modelo entra em um "sulco" de alta probabilidade, ele tende a ficar nele, gerando frases como "eu acho que eu acho que eu acho que...".
- Sub-ótimo globalmente: o token de maior probabilidade *agora* pode levar a uma sequência de baixa probabilidade *no total*. Greedy é uma escolha gulosa (myopic) — ele nunca olha à frente. (Beam search tenta mitigar isso mantendo múltiplas hipóteses, mas isso é assunto de outro capítulo.)
- Sem diversidade: útil para tarefas com resposta "certa" (tradução literal, código determinístico), péssimo para criatividade.

---

## Sampling Estocástico

Em vez de sempre pegar o argmax, podemos **amostrar** um token da distribuição de probabilidade completa. Se o token A tem probabilidade 0.6 e o token B tem 0.3, amostrar significa que A sai 60% das vezes e B sai 30% das vezes — não que A sempre vence.

Isso é literalmente jogar um dado viciado, onde os pesos do dado são as probabilidades do softmax.

```python
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

`torch.multinomial(probs, num_samples=1)` faz exatamente isso: recebe um vetor de probabilidades (não-normalizado ou normalizado, mas aqui usamos normalizado) e sorteia um índice de acordo com essas probabilidades.

O problema: sampling puro da distribuição completa é arriscado. Mesmo tokens com probabilidade muito baixa (0.0001) têm *alguma* chance de ser escolhidos, e como o vocabulário tem dezenas de milhares de tokens, a soma dessas "caudas" pequenas pode ser significativa. Ocasionalmente você amostra um token completamente aleatório e sem sentido, e o modelo "descarrila" — porque agora ele precisa continuar uma sequência que começa com um erro.

É aqui que entram temperature, top-k e top-p: formas de moldar a distribuição *antes* de amostrar, para reduzir esse risco sem eliminar a aleatoriedade útil.

---

## Temperature: Achatando ou Afiando a Distribuição

A temperature (T) é aplicada **antes** do softmax, dividindo os logits:

$$P(\text{token}_i) = \text{softmax}\left(\frac{\text{logits}_i}{T}\right) = \frac{e^{\text{logits}_i / T}}{\sum_j e^{\text{logits}_j / T}}$$

O efeito de T:

- **T = 1.0**: softmax normal, sem alteração.
- **T < 1.0** (ex: 0.5): divide por um número menor que 1, o que **amplia** as diferenças entre logits. A distribuição fica mais "afiada" — concentrada nos tokens já favoritos. No limite T → 0, isso se aproxima de greedy decoding (o argmax domina completamente).
- **T > 1.0** (ex: 2.0): divide por um número maior que 1, o que **reduz** as diferenças entre logits. A distribuição fica mais "achatada" — mais uniforme, dando mais chance a tokens que antes eram improváveis. No limite T → ∞, a distribuição se aproxima de uniforme (escolha totalmente aleatória entre todos os tokens).

A intuição física: pense em temperature literal. Um sólido "frio" (T baixo) tem átomos travados em posições fixas — comportamento rígido e previsível. Um gás "quente" (T alto) tem átomos se movendo caoticamente em qualquer direção — comportamento errático. É exatamente essa metáfora que a fórmula do softmax com temperature, importada da física estatística (distribuição de Boltzmann), carrega.

### Exemplo Numérico Manual

Considere logits para 4 tokens candidatos: `["gato", "cachorro", "elefante", "xadrez"]`

```
logits = [2.0, 1.0, 0.5, -1.0]
```

**T = 1.0 (baseline)**

```
exp(2.0/1)  = 7.389
exp(1.0/1)  = 2.718
exp(0.5/1)  = 1.649
exp(-1.0/1) = 0.368
soma = 12.124

probs = [7.389/12.124, 2.718/12.124, 1.649/12.124, 0.368/12.124]
      = [0.609, 0.224, 0.136, 0.030]
```

**T = 0.5 (mais determinístico)**

```
logits/T = [4.0, 2.0, 1.0, -2.0]

exp(4.0) = 54.598
exp(2.0) = 7.389
exp(1.0) = 2.718
exp(-2.0) = 0.135
soma = 64.840

probs = [54.598/64.840, 7.389/64.840, 2.718/64.840, 0.135/64.840]
      = [0.842, 0.114, 0.042, 0.002]
```

Note: "gato" pula de 60.9% para 84.2% — a distribuição ficou muito mais concentrada.

**T = 2.0 (mais aleatório)**

```
logits/T = [1.0, 0.5, 0.25, -0.5]

exp(1.0)  = 2.718
exp(0.5)  = 1.649
exp(0.25) = 1.284
exp(-0.5) = 0.607
soma = 6.258

probs = [2.718/6.258, 1.649/6.258, 1.284/6.258, 0.607/6.258]
      = [0.434, 0.264, 0.205, 0.097]
```

Note: "gato" caiu para 43.4%, e "xadrez" (o mais improvável) subiu de 3.0% para 9.7% — quase 3x mais chance de ser escolhido, mesmo sendo semanticamente estranho na lista.

| Token | T=0.5 | T=1.0 | T=2.0 |
|---|---|---|---|
| gato | 0.842 | 0.609 | 0.434 |
| cachorro | 0.114 | 0.224 | 0.264 |
| elefante | 0.042 | 0.136 | 0.205 |
| xadrez | 0.002 | 0.030 | 0.097 |

Essa tabela é o resumo visual de tudo: T baixo empurra a massa de probabilidade para o topo; T alto a espalha para a cauda.

---

## Top-K Sampling

Temperature muda a *forma* da distribuição, mas ainda amostra do vocabulário inteiro. Top-K resolve isso de forma mais direta: **restrinja a amostragem aos K tokens mais prováveis**, zere a probabilidade de todo o resto, renormalize, e amostre apenas dentro desse conjunto.

Algoritmo:

1. Ordene os logits, pegue os K maiores.
2. Zere (ou defina como `-inf`) todos os logits fora do top-K.
3. Aplique softmax nos logits resultantes (os `-inf` viram probabilidade 0).
4. Amostre com `torch.multinomial`.

```python
def top_k_sampling(logits, k):
    values, indices = torch.topk(logits, k)
    # Cria uma máscara: tudo que não está no top-k vira -inf
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, indices, values)
    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

**Exemplo numérico**: usando os mesmos logits `[2.0, 1.0, 0.5, -1.0]` com K=2:

```
top-2 valores: [2.0, 1.0] (índices 0 e 1: "gato" e "cachorro")
logits filtrados = [2.0, 1.0, -inf, -inf]

softmax([2.0, 1.0, -inf, -inf]):
exp(2.0) = 7.389
exp(1.0) = 2.718
exp(-inf) = 0
soma = 10.107

probs = [7.389/10.107, 2.718/10.107, 0, 0]
      = [0.731, 0.269, 0, 0]
```

Agora "elefante" e "xadrez" têm probabilidade exatamente zero — não importa quão "quente" a temperature seja, eles nunca serão escolhidos. K funciona como um limite rígido de quantos candidatos entram na loteria.

**Limitação de Top-K**: K é fixo, mas a "forma" da distribuição varia token a token. Às vezes o modelo está muito confiante (um token domina com 95%, o resto é ruído) — nesse caso K=40 inclui 39 tokens praticamente irrelevantes. Outras vezes o modelo está genuinamente incerto entre muitos candidatos plausíveis (uma distribuição quase uniforme entre 100 tokens) — nesse caso K=40 corta candidatos que mereciam entrar. K fixo não se adapta a essa variação de "confiança".

---

## Top-P (Nucleus Sampling)

Top-P, também chamado *nucleus sampling*, resolve exatamente essa limitação: em vez de fixar o número de tokens, fixamos a **massa de probabilidade cumulativa**. Escolhemos o menor conjunto de tokens cuja soma de probabilidades ultrapassa um limiar P (ex: 0.9), e amostramos apenas dentro desse conjunto ("núcleo").

Algoritmo:

1. Ordene os tokens por probabilidade, do maior para o menor.
2. Some as probabilidades acumuladas.
3. Encontre o ponto de corte onde a soma acumulada ultrapassa P.
4. Zere tudo fora desse conjunto, renormalize, amostre.

```python
def top_p_sampling(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Máscara: True para tokens que devem ser removidos (depois do corte)
    sorted_mask = cumulative_probs - sorted_probs > p
    # (subtraímos sorted_probs para sempre manter ao menos 1 token)

    sorted_probs[sorted_mask] = 0.0
    # Devolve para a ordem original do vocabulário
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_indices, sorted_probs)
    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(new_probs, num_samples=1)
    return next_token
```

**Exemplo numérico** com os probs de T=1.0: `[0.609, 0.224, 0.136, 0.030]` (já ordenados), P=0.9:

```
cumulativo: [0.609, 0.833, 0.969, 0.999]

Procuramos o menor prefixo cuja soma >= 0.9:
0.609 < 0.9 → inclui "gato", continua
0.833 < 0.9 → inclui "cachorro", continua
0.969 >= 0.9 → inclui "elefante", PARA

núcleo = {gato, cachorro, elefante}, exclui "xadrez"

Renormalizando sobre o núcleo (soma = 0.609+0.224+0.136 = 0.969):
probs = [0.609/0.969, 0.224/0.969, 0.136/0.969, 0]
      = [0.628, 0.231, 0.140, 0]
```

Compare com top-K=2, que teria incluído apenas gato/cachorro. Aqui, top-P=0.9 incluiu 3 tokens porque a distribuição original era "espalhada" o suficiente para precisar de 3 tokens para atingir 90% de massa.

**Por que top-P é mais adaptativo**: se em outro passo de geração a distribuição fosse muito mais confiante — digamos `[0.95, 0.03, 0.01, 0.01]` — o núcleo com P=0.9 conteria **apenas 1 token** (0.95 já ultrapassa 0.9 sozinho), efetivamente virando greedy decoding automaticamente naquele passo. Já com uma distribuição quase uniforme `[0.27, 0.26, 0.24, 0.23]`, o núcleo precisaria incluir 4 tokens para atingir 90%. O **tamanho do conjunto muda dinamicamente** conforme a "confiança" (entropia) da distribuição — exatamente o que top-K fixo não consegue fazer.

---

## Combinando as Estratégias

Na prática, sistemas de produção (GPT, Claude, Llama) costumam combinar tudo: aplicar temperature primeiro, depois top-K (opcional, como corte grosseiro de segurança), depois top-P (corte fino adaptativo), e só então amostrar. A ordem típica é:

```
logits → / temperature → top-k (opcional) → top-p → softmax → multinomial
```

Cada camada de filtro é uma rede de segurança adicional contra amostrar um token absurdo, mantendo a variabilidade que torna o texto gerado interessante.

---

## Experimento: Comparando Estratégias de Sampling

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Sampling, Temperature e Top-K/P")
print("=" * 70)

# ========== 1. DISTRIBUIÇÃO DE EXEMPLO ==========
print("\n1. LOGITS DE EXEMPLO")
print("-" * 70)

vocab = ["gato", "cachorro", "elefante", "xadrez", "nuvem", "prédio"]
logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])

print(f"Vocabulário: {vocab}")
print(f"Logits:      {logits.tolist()}")

probs_base = F.softmax(logits, dim=-1)
print(f"Probs (T=1): {[round(p, 4) for p in probs_base.tolist()]}")

# ========== 2. GREEDY DECODING ==========
print("\n2. GREEDY DECODING")
print("-" * 70)

greedy_token = torch.argmax(logits).item()
print(f"Token escolhido (sempre o mesmo): '{vocab[greedy_token]}' (índice {greedy_token})")

# ========== 3. EFEITO DA TEMPERATURE ==========
print("\n3. EFEITO DA TEMPERATURE NA DISTRIBUIÇÃO")
print("-" * 70)

for T in [0.5, 1.0, 2.0]:
    probs_T = F.softmax(logits / T, dim=-1)
    print(f"T={T}: {[round(p, 4) for p in probs_T.tolist()]}")

print("\nObservação: T baixo concentra probabilidade no topo;")
print("T alto achata a distribuição, dando mais chance à cauda.")

# ========== 4. SAMPLING PURO (sem filtros) ==========
print("\n4. SAMPLING ESTOCÁSTICO PURO (T=1.0)")
print("-" * 70)

contagem = {t: 0 for t in vocab}
N = 1000
for _ in range(N):
    tok = torch.multinomial(probs_base, num_samples=1).item()
    contagem[vocab[tok]] += 1

print(f"Frequências observadas em {N} amostras:")
for tok, c in contagem.items():
    print(f"  {tok:12s}: {c:4d} ({c/N:.1%})")

# ========== 5. TOP-K SAMPLING ==========
print("\n5. TOP-K SAMPLING (K=3)")
print("-" * 70)

def top_k_sampling(logits, k):
    values, indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, indices, values)
    probs = F.softmax(filtered_logits, dim=-1)
    return probs

probs_topk = top_k_sampling(logits, k=3)
print(f"Probs após top-k=3: {[round(p, 4) for p in probs_topk.tolist()]}")
print("Tokens fora do top-3 têm probabilidade exatamente 0.")

contagem_k = {t: 0 for t in vocab}
for _ in range(N):
    tok = torch.multinomial(probs_topk, num_samples=1).item()
    contagem_k[vocab[tok]] += 1
print(f"\nFrequências observadas em {N} amostras (top-k=3):")
for tok, c in contagem_k.items():
    print(f"  {tok:12s}: {c:4d} ({c/N:.1%})")

# ========== 6. TOP-P SAMPLING ==========
print("\n6. TOP-P (NUCLEUS) SAMPLING (P=0.9)")
print("-" * 70)

def top_p_sampling(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_probs = sorted_probs.clone()
    sorted_probs[sorted_mask] = 0.0

    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_indices, sorted_probs)
    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
    return new_probs

probs_topp = top_p_sampling(logits, p=0.9)
print(f"Probs após top-p=0.9: {[round(p, 4) for p in probs_topp.tolist()]}")
n_incluidos = (probs_topp > 0).sum().item()
print(f"Número de tokens no núcleo: {n_incluidos}")

# Comparar com distribuição mais "confiante"
logits_confiantes = torch.tensor([5.0, 0.5, 0.3, -1.0, -2.0, -3.0])
probs_conf = top_p_sampling(logits_confiantes, p=0.9)
n_incluidos_conf = (probs_conf > 0).sum().item()
print(f"\nCom distribuição mais 'confiante' (um token dominante):")
print(f"Probs base: {[round(p,4) for p in F.softmax(logits_confiantes, dim=-1).tolist()]}")
print(f"Núcleo top-p=0.9 inclui apenas {n_incluidos_conf} token(s)")
print("Isso mostra a adaptatividade: o tamanho do núcleo muda com a confiança.")

# ========== 7. COMPARAÇÃO FINAL ==========
print("\n7. RESUMO COMPARATIVO")
print("-" * 70)
print(f"{'Estratégia':<20s}{'Determinístico?':<18s}{'Adaptativo?':<15s}")
print(f"{'Greedy':<20s}{'Sim':<18s}{'N/A':<15s}")
print(f"{'Sampling puro':<20s}{'Não':<18s}{'N/A':<15s}")
print(f"{'Top-K':<20s}{'Não':<18s}{'Não (K fixo)':<15s}")
print(f"{'Top-P':<20s}{'Não':<18s}{'Sim':<15s}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Saída esperada (resumida): você verá a distribuição afiar/achatar com T, o top-K zerar a cauda completamente, e o top-P mostrar um núcleo maior quando a distribuição é incerta e um núcleo de tamanho 1 quando um token domina.

---

## Erros Comuns

### Erro 1: Aplicar temperature depois do softmax

```python
# ❌ Errado — dividir probabilidades por T não tem a mesma semântica
probs = F.softmax(logits, dim=-1)
probs = probs / T  # Isso quebra a normalização (soma não é mais 1)!

# ✓ Certo — dividir os LOGITS por T, antes do softmax
probs = F.softmax(logits / T, dim=-1)
```

Dividir probabilidades diretamente por T não corresponde à distribuição de Boltzmann e nem sequer soma 1 depois — é matematicamente diferente de aplicar temperature corretamente.

### Erro 2: Esquecer de renormalizar depois de filtrar (top-k/top-p)

```python
# ❌ Errado — usar softmax nos logits ORIGINAIS depois de já ter
# zerado probabilidades manualmente, sem re-normalizar
probs = F.softmax(logits, dim=-1)
probs[fora_do_topk] = 0.0
next_token = torch.multinomial(probs, num_samples=1)  # soma != 1, comportamento inesperado

# ✓ Certo — ou zere os LOGITS (com -inf) antes do softmax,
# ou renormalize explicitamente depois de zerar probabilidades
filtered_logits = logits.clone()
filtered_logits[fora_do_topk] = float('-inf')
probs = F.softmax(filtered_logits, dim=-1)  # já soma 1 automaticamente
```

`torch.multinomial` não exige que o vetor some exatamente 1 (ele aceita pesos não normalizados), mas se você misturar filtragens manuais de probabilidade sem cuidado, é fácil introduzir vieses sutis — o caminho mais seguro é sempre voltar para logits com `-inf` e deixar o softmax renormalizar.

### Erro 3: Usar T=0 diretamente na fórmula

```python
# ❌ Errado — divisão por zero!
probs = F.softmax(logits / 0.0, dim=-1)  # gera inf/nan

# ✓ Certo — T=0 deveria ser tratado como um caso especial (= greedy)
if T == 0:
    next_token = torch.argmax(logits, dim=-1)
else:
    probs = F.softmax(logits / T, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
```

Muitas APIs de LLM (incluindo bibliotecas populares) tratam `temperature=0` como sinônimo de greedy decoding exatamente por esse motivo — matematicamente T→0 tende a um argmax, mas literalmente dividir por zero quebra o cálculo.

---

## Exercícios

### Exercício 31.1: Greedy vs. Argmax
Dado `logits = torch.tensor([0.1, 3.5, -1.0, 3.5])`, qual token o greedy decoding escolhe? O que acontece em caso de empate (dois logits iguais e máximos)? Teste com `torch.argmax` e explique o comportamento.

### Exercício 31.2: Temperature Extrema
Compute `F.softmax(logits / T, dim=-1)` para `logits = [1.0, 2.0, 3.0]` com T=0.01 e T=100. O que acontece com a distribuição em cada extremo? Compare com o argmax e com a distribuição uniforme.

### Exercício 31.3: Implementando Top-K do Zero
Sem usar `torch.topk`, implemente sua própria versão de top-k sampling usando `torch.sort`. Teste com `logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])` e k=2, e confirme que o resultado bate com a versão que usa `torch.topk`.

### Exercício 31.4: Top-P com P=1.0 e P Muito Pequeno
O que acontece com `top_p_sampling` quando P=1.0? E quando P=0.01? Explique por que P=1.0 equivale a sampling puro e P muito pequeno se aproxima de greedy.

### Exercício 31.5: Combinando Temperature + Top-P
Escreva uma função `sample(logits, temperature, top_p)` que aplica temperature primeiro e depois filtra por top-p antes de amostrar. Teste com `temperature=0.7, top_p=0.9` na distribuição do experimento principal e rode 500 amostras, reportando a frequência de cada token.

---

## Gabarito

### Exercício 31.1: Greedy vs. Argmax
```python
import torch

logits = torch.tensor([0.1, 3.5, -1.0, 3.5])
token = torch.argmax(logits).item()
print(token)  # 1 — argmax retorna o PRIMEIRO índice máximo em caso de empate

# Índices 1 e 3 têm o mesmo valor (3.5), mas argmax sempre
# desempata deterministicamente pegando o de menor índice.
# Isso significa que greedy decoding é reprodutível mesmo com empates,
# mas pode enviesar sistematicamente para tokens de índice menor
# no vocabulário se houver empates frequentes.
```

### Exercício 31.2: Temperature Extrema
```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, 2.0, 3.0])

probs_baixa = F.softmax(logits / 0.01, dim=-1)
print(probs_baixa)  # tensor([~0.0, ~0.0, ~1.0]) — praticamente one-hot, igual argmax

probs_alta = F.softmax(logits / 100, dim=-1)
print(probs_alta)  # tensor([~0.322, ~0.332, ~0.346]) — quase uniforme (1/3 cada)

# T -> 0: distribuição colapsa no argmax (one-hot)
# T -> infinito: distribuição tende à uniforme (1/vocab_size cada)
```

### Exercício 31.3: Implementando Top-K do Zero
```python
import torch
import torch.nn.functional as F

def top_k_sampling_manual(logits, k):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    kth_value = sorted_logits[k - 1]  # k-ésimo maior valor (limiar de corte)
    mask = logits < kth_value
    filtered_logits = logits.clone()
    filtered_logits[mask] = float('-inf')
    return F.softmax(filtered_logits, dim=-1)

logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

probs_manual = top_k_sampling_manual(logits, k=2)
print(probs_manual)  # tensor([0.7311, 0.2689, 0., 0., 0.])

# Comparação com torch.topk
def top_k_sampling(logits, k):
    values, indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, indices, values)
    return F.softmax(filtered_logits, dim=-1)

probs_topk = top_k_sampling(logits, k=2)
print(probs_topk)  # tensor([0.7311, 0.2689, 0., 0., 0.]) — idêntico
```

### Exercício 31.4: Top-P com P=1.0 e P Muito Pequeno
```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_probs = sorted_probs.clone()
    sorted_probs[sorted_mask] = 0.0
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_indices, sorted_probs)
    return new_probs / new_probs.sum(dim=-1, keepdim=True)

logits = torch.tensor([2.0, 1.0, 0.5, -1.0])

probs_p1 = top_p_sampling(logits, p=1.0)
print(probs_p1)
# Com P=1.0, a soma cumulativa SEMPRE ultrapassa 1.0 no último token
# (ou antes), então NENHUM token é zerado — o núcleo é o vocabulário
# inteiro. Isso equivale a sampling puro (softmax normal).

probs_p_pequeno = top_p_sampling(logits, p=0.01)
print(probs_p_pequeno)
# Com P muito pequeno, o primeiro token sozinho já ultrapassa 0.01
# (quase qualquer prob individual é > 0.01), então o núcleo vira
# {token mais provável} — equivalente a greedy decoding.
```

### Exercício 31.5: Combinando Temperature + Top-P
```python
import torch
import torch.nn.functional as F

def sample(logits, temperature=1.0, top_p=1.0):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs - sorted_probs > top_p
    sorted_probs = sorted_probs.clone()
    sorted_probs[sorted_mask] = 0.0

    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_indices, sorted_probs)
    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(new_probs, num_samples=1)

torch.manual_seed(42)
vocab = ["gato", "cachorro", "elefante", "xadrez", "nuvem", "prédio"]
logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])

contagem = {t: 0 for t in vocab}
for _ in range(500):
    tok = sample(logits, temperature=0.7, top_p=0.9).item()
    contagem[vocab[tok]] += 1

for tok, c in contagem.items():
    print(f"{tok:12s}: {c:4d} ({c/500:.1%})")
# "gato" e "cachorro" devem dominar (T=0.7 concentra, top_p=0.9 corta a cauda)
```

---

## Desafios Avançados (Opcionais)

### Fixação 31.1: Min-P Sampling
Pesquise (ou deduza) como funcionaria "min-p sampling": em vez de um limiar de massa cumulativa, use um limiar relativo à probabilidade do token mais provável (ex: mantenha apenas tokens com prob >= 0.1 * prob_max). Implemente e compare com top-p.

### Fixação 31.2: Repetition Penalty
Implemente uma penalidade de repetição: antes de amostrar, reduza os logits de tokens que já apareceram na sequência gerada (ex: `logits[token_ja_usado] -= penalty`). Teste em um loop de geração simulado e observe o efeito na diversidade.

### Fixação 31.3: Entropia da Distribuição Filtrada
Compute a entropia de Shannon da distribuição antes e depois de aplicar top-k e top-p. Confirme que ambos os filtros reduzem a entropia (tornam a distribuição menos incerta) e compare a magnitude da redução entre eles para a mesma distribuição base.

### Fixação 31.4: Sampling com Batch
Generalize sua função `sample` para aceitar `logits` de shape `[batch, vocab_size]` (múltiplas distribuições simultâneas) em vez de um vetor único. Cuidado com `torch.multinomial`, que já suporta batch nativamente, mas `torch.sort`/`torch.cumsum` precisam operar na dimensão correta (`dim=-1`).

### Fixação 31.5: Comparando Estatísticas de Longo Prazo
Rode 10.000 amostras de `sample(logits, temperature=1.0, top_p=1.0)` (sampling puro) e compare a frequência empírica de cada token com a probabilidade teórica do softmax. Quão perto elas ficam? O que isso te diz sobre a lei dos grandes números aplicada a sampling de LLMs?

---

## Resumo

- **Greedy decoding**: sempre escolhe o token de maior probabilidade — determinístico, mas repetitivo e mio pico (short-sighted).
- **Sampling estocástico**: amostra da distribuição de probabilidade completa via `torch.multinomial` — introduz variabilidade, mas arrisca escolher tokens da cauda irrelevante.
- **Temperature**: divide os logits por T antes do softmax — T baixo afia a distribuição (mais determinístico), T alto a achata (mais aleatório).
- **Top-K**: restringe a amostragem aos K tokens mais prováveis, zerando o resto — corte rígido, não se adapta à forma da distribuição.
- **Top-P (nucleus)**: restringe ao menor conjunto cuja probabilidade cumulativa excede P — corte adaptativo, o tamanho do núcleo varia com a confiança do modelo.
- **Combinação**: sistemas reais aplicam temperature, depois (opcionalmente) top-k, depois top-p, antes de amostrar — camadas de filtro que equilibram criatividade e coerência.

Próximo capítulo: **Geração de Texto Completa** — como transformar essas estratégias de amostragem em um loop de geração autoregressiva de ponta a ponta, com controle de contexto e condições de parada.

---

**Próximo**: [Capítulo 32: Geração de Texto Completa](32_geracao_texto.md)
