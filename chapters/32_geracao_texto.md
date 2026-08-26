# Capítulo 32: Geração de Texto Completa

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Implementar o loop de geração autoregressiva completo, do prompt ao texto final
2. Entender a diferença fundamental entre o forward pass de treino e o de inferência
3. Gerenciar o truncamento de contexto quando a sequência excede o limite do modelo
4. Reconhecer e implementar condições de parada (EOS, comprimento máximo)
5. Entender a intuição por trás de KV-caching e por que ele acelera drasticamente a inferência

---

## Por Que Isso Importa

Até agora você viu como um Transformer processa uma sequência inteira de uma vez (treino) e como escolher um único próximo token a partir de uma distribuição de probabilidade (Capítulo 31). Mas ChatGPT não gera uma resposta inteira em uma única passada — ele produz um token, depois mais um, depois mais um, cada um aparecendo na tela poucos milissegundos depois do anterior. Essa cadência é o resultado direto de como a geração de texto realmente funciona: **um loop**.

Pense em como você mesmo escreve uma frase falando alto sem saber o final dela de antemão: você fala uma palavra, ouve o que acabou de dizer, decide a próxima palavra com base em tudo que já foi dito, fala essa palavra, e repete. Você nunca "trava" toda a frase de uma vez na sua cabeça antes de abrir a boca. Um Transformer autoregressivo funciona exatamente assim: gera um token, anexa esse token ao contexto, roda o forward pass de novo sobre o contexto (agora um token mais longo), e repete até decidir parar.

Esse mecanismo simples — token, concatena, repete — é o motivo de LLMs serem lentas para gerar textos longos (cada token exige um forward pass inteiro) e é também o motivo de existirem otimizações inteligentes como o KV-cache, que este capítulo explica na intuição, mesmo sem implementá-lo em produção. Entender esse loop de ponta a ponta é o que separa "eu sei o que é atenção" de "eu sei como o ChatGPT realmente decide a próxima palavra, byte a byte".

---

## Treino vs. Inferência: Paralelo vs. Sequencial

Esse é o ponto mais frequentemente mal-entendido por quem está aprendendo Transformers, então vale isolar bem.

### Durante o treino: teacher forcing, tudo em paralelo

No Capítulo 29 (Treinamento Autoregressivo), você viu que, dado um exemplo de treino como `"O gato dormia"`, o modelo recebe a sequência inteira de entrada de uma vez:

```
input:  [O, gato, dormia]
target: [gato, dormia, <EOS>]
```

O forward pass processa as 3 posições **simultaneamente**, em uma única chamada de matriz. Graças à máscara causal (Capítulo 15), a posição `i` só pode "ver" as posições `<= i`, então mesmo processando tudo de uma vez, a predição da posição 0 não "trapaceia" olhando para a posição 1. Isso se chama **teacher forcing**: o modelo sempre recebe a sequência *correta* como entrada, mesmo que ele mesmo tivesse errado a predição no passo anterior. É extremamente eficiente porque uma única passada de matriz calcula a loss para todas as posições ao mesmo tempo.

```
Shape do forward de treino: [batch, seq_len, vocab_size]
                                        ↑
                              todas as posições calculadas juntas
```

### Durante a inferência: sequencial, um token por vez

Na hora de gerar texto novo, não existe "resposta correta" para dar de entrada — o próprio modelo tem que decidir o próximo token, e só então esse token vira parte da entrada do próximo passo. Isso é **fundamentalmente sequencial**:

```
Passo 1: input=[O]                    → prediz "gato"        → contexto=[O, gato]
Passo 2: input=[O, gato]              → prediz "dormia"      → contexto=[O, gato, dormia]
Passo 3: input=[O, gato, dormia]      → prediz "no"          → contexto=[O, gato, dormia, no]
...
```

Cada passo exige um forward pass **completo** sobre o contexto acumulado até aquele ponto. Gerar uma resposta de 100 tokens significa rodar o modelo 100 vezes, cada vez sobre um contexto um pouco maior. É por isso que a geração de texto é notavelmente mais lenta que o treino: não há paralelismo entre tokens gerados — o token N+1 literalmente não pode ser calculado antes do token N existir.

| | Treino | Inferência |
|---|---|---|
| Entrada | Sequência completa de uma vez | Cresce token a token |
| Paralelismo | Todas as posições em paralelo | Sequencial, uma posição nova por vez |
| "Resposta certa" | Sim (teacher forcing) | Não — o modelo usa sua própria saída |
| Custo por token | Amortizado (uma passada, N losses) | Uma passada inteira POR token novo |

---

## O Loop de Geração Autoregressiva

O algoritmo central deste capítulo, em pseudocódigo:

```
função generate(modelo, prompt_ids, max_novos_tokens):
    contexto = prompt_ids
    para cada passo em range(max_novos_tokens):
        contexto_truncado = truncar(contexto, max_seq_len)
        logits = modelo.forward(contexto_truncado)
        proximo_token = escolher_token(logits[:, -1, :])   # sampling do Cap. 31
        contexto = concatenar(contexto, proximo_token)
        se proximo_token == EOS_TOKEN:
            parar
    retornar contexto
```

Vamos detalhar cada peça.

### Extraindo o logit da última posição

O forward pass retorna logits para **todas** as posições da sequência: shape `[batch, seq_len, vocab_size]`. Mas para decidir o *próximo* token, só nos interessa a predição feita na **última posição** — porque é essa posição que, graças à máscara causal, já "viu" toda a sequência até aqui e está prevendo o que vem depois dela.

```python
logits = model(context)              # [batch, seq_len, vocab_size]
next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
```

Isso é fácil de esquecer: em treino, usamos os logits de **todas** as posições (porque temos um target para cada uma). Em geração, usamos só o **último**.

### Truncamento de Contexto (Sliding Window)

Todo modelo Transformer tem um `max_seq_len` fixo — o positional embedding (Capítulo 22) só foi treinado para posições até esse limite, e a complexidade quadrática da atenção (Capítulo 10) torna contextos muito longos caros. Se o contexto acumulado ultrapassa `max_seq_len`, precisamos truncar antes do forward pass.

A estratégia mais comum é uma **janela deslizante** (sliding window): manter apenas os últimos `max_seq_len` tokens, descartando os mais antigos.

```python
context_window = context[:, -max_seq_len:]  # pega só os últimos N tokens
logits = model(context_window)
```

Isso significa que, à medida que a geração avança além de `max_seq_len`, o modelo literalmente "esquece" o início do prompt — ele só enxerga a janela mais recente. Esse é um dos motivos pelos quais LLMs "perdem o fio" em conversas muito longas quando o contexto excede o limite do modelo.

### Condições de Parada

Um loop de geração precisa saber quando parar. Duas condições são padrão:

1. **Token de fim de sequência (EOS)**: um token especial no vocabulário (frequentemente índice 0 ou um ID reservado) que sinaliza "a geração terminou aqui". Quando o modelo amostra o EOS, paramos imediatamente — mesmo que `max_novos_tokens` não tenha sido atingido.

2. **Comprimento máximo**: um limite superior de segurança (`max_novos_tokens`), para garantir que o loop sempre termina, mesmo que o modelo nunca gere EOS (o que pode acontecer com um modelo mal treinado, ou deliberadamente quando você quer uma quantidade fixa de tokens).

```python
for _ in range(max_novos_tokens):
    ...
    if next_token.item() == EOS_TOKEN_ID:
        break
```

Sistemas de produção frequentemente adicionam mais condições: sequências de parada customizadas (stop sequences, ex: `"\n\nHuman:"` em prompts de chat), ou até um orçamento de tempo/latência.

---

## KV-Caching: A Intuição

Aqui está o problema de performance escondido no loop acima: a cada novo token gerado, recomputamos o forward pass sobre o contexto **inteiro**, incluindo tokens que já foram processados nos passos anteriores. Se o contexto tem 500 tokens e vamos gerar mais 100, o passo de geração do token 600 processa TODOS os 600 tokens do zero — mesmo que os primeiros 599 já tenham sido processados 599 vezes antes, em passos anteriores.

Isso é um desperdício gigantesco, porque a maior parte desse trabalho é redundante: os vetores Key (K) e Value (V) de um token que já existia no contexto **não mudam** quando um novo token é adicionado ao final (graças à máscara causal — um token antigo nunca "olha para frente" para o token novo). Só o vetor Query (Q) do último token, e a interação dele com os K/V de todos os anteriores, é realmente novo trabalho.

**KV-caching** explora exatamente isso: em vez de recalcular K e V para toda a sequência a cada passo, guardamos (fazemos cache) os K e V já computados nos passos anteriores, e em cada novo passo computamos K e V **apenas para o token novo**, concatenando-os ao cache existente.

```
Sem cache (passo N):
  recomputa Q, K, V para TODOS os N tokens do contexto
  custo: O(N) trabalho de projeção, a cada passo

Com cache (passo N):
  recomputa Q, K, V APENAS para o token novo (1 token)
  concatena o novo K, V ao cache de K, V dos passos anteriores
  custo: O(1) trabalho de projeção por passo, mais leitura do cache
```

O ganho é substancial: sem cache, gerar N tokens custa aproximadamente O(N²) trabalho de projeção total (cada um dos N passos recomputa até N vetores); com cache, custa O(N) — cada passo faz um trabalho constante, e só a etapa de atenção (que compara o Q novo contra todos os K armazenados) ainda cresce com o tamanho do contexto. Na prática, KV-caching é a diferença entre uma LLM gerar texto em tempo real ou levar minutos para uma resposta curta.

Este capítulo não implementa um KV-cache completo de produção (isso envolve gerenciamento cuidadoso de memória, batching dinâmico, e é tipicamente delegado a bibliotecas como o `transformers` da Hugging Face ou motores de inferência dedicados como vLLM). O que importa aqui é a intuição: **tokens antigos no contexto não mudam seus K/V quando você adiciona um token novo, então recomputá-los do zero a cada passo é puro desperdício.**

---

## Batch Generation

Gerar texto para um único prompt por vez é comum ao experimentar, mas sistemas em produção geralmente processam vários prompts simultaneamente para aproveitar melhor a GPU. A ideia é simples: em vez de `input_ids` com shape `[1, seq_len]`, usamos `[batch_size, seq_len]`, e cada linha do batch é um prompt diferente sendo gerado em paralelo.

```python
prompts = ["O gato", "Um cachorro", "A chuva"]
# tokenizar cada um, fazer padding até o mesmo comprimento
batch_ids = tokenize_and_pad(prompts)  # [3, seq_len]

output = model.generate(batch_ids, max_novos_tokens=20)  # [3, seq_len + 20]
```

A complicação principal do batch generation é que prompts diferentes têm comprimentos diferentes, exigindo **padding** (Capítulo 25) e, frequentemente, uma **attention mask** adicional para o modelo ignorar posições de padding. Outro detalhe sutil: se um prompt do batch já gerou seu EOS mas outro ainda não, você precisa continuar gerando para o batch inteiro (porque é uma única chamada de matriz), mas descartar/ignorar os tokens gerados depois do EOS daquele prompt específico.

---

## Qualidade de Geração: O Que Observar

Depois de gerar texto, como saber se está "bom"? Alguns sinais práticos, ligados diretamente às escolhas de sampling do Capítulo 31:

- **Repetição**: frases ou n-gramas se repetindo (`"eu acho que eu acho que eu acho que"`). Sintoma clássico de temperature baixa demais ou greedy decoding puro. Mitigado por sampling com temperature > 0, top-p, ou penalidades de repetição.
- **Degeneração / incoerência**: texto que perde o sentido gramatical ou semântico, saltando entre tópicos sem conexão. Sintoma de temperature alta demais ou ausência de filtros (top-k/top-p), permitindo que tokens muito improváveis sejam escolhidos.
- **Monotonia**: texto gramaticalmente correto mas genérico e "sem graça", sempre escolhendo o caminho mais óbvio. Sintoma de greedy decoding ou temperature muito baixa.
- **Coerência de longo prazo**: o texto mantém consistência com o que foi dito páginas atrás? Isso depende do `max_seq_len` do modelo e de quanto contexto sobrevive ao truncamento por sliding window.

Na prática, ajustar `temperature` e `top_p` é o principal instrumento de controle de qualidade disponível na hora da geração — o modelo já está treinado e fixo, mas o comportamento de saída ainda pode ser bastante moldado por esses parâmetros de decodificação.

---

## Experimento: Função generate() Completa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Loop de Geração de Texto Completo")
print("=" * 70)

# ========== 1. MODELO MINIATURA (para o experimento) ==========
print("\n1. CONSTRUINDO UM MODELO PEQUENO")
print("-" * 70)

class ModeloMiniatura(nn.Module):
    """Transformer minúsculo apenas para demonstrar o loop de geração."""
    def __init__(self, vocab_size, d_model=16, num_heads=2, max_seq_len=32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Máscara causal: posição i não pode ver posição j > i
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        logits = self.head(x)  # [batch, seq_len, vocab_size]
        return logits

VOCAB_SIZE = 20
EOS_TOKEN_ID = 0
MAX_SEQ_LEN = 16

model = ModeloMiniatura(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Modelo criado com {n_params} parâmetros")
print(f"vocab_size={VOCAB_SIZE}, max_seq_len={MAX_SEQ_LEN}, EOS_TOKEN_ID={EOS_TOKEN_ID}")

# ========== 2. FUNÇÕES DE SAMPLING (do Capítulo 31) ==========
print("\n2. FUNÇÕES DE SAMPLING (reutilizadas do Capítulo 31)")
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
    return new_probs / new_probs.sum(dim=-1, keepdim=True)

def escolher_proximo_token(logits, temperature=1.0, top_p=1.0):
    """logits: [batch, vocab_size] -> next_token: [batch, 1]"""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    probs = top_p_sampling(scaled, top_p)
    return torch.multinomial(probs, num_samples=1)

print("Funções escolher_proximo_token() e top_p_sampling() prontas.")

# ========== 3. A FUNÇÃO GENERATE() ==========
print("\n3. LOOP DE GERAÇÃO AUTOREGRESSIVA")
print("-" * 70)

@torch.no_grad()
def generate(model, prompt_ids, max_novos_tokens, max_seq_len,
             temperature=1.0, top_p=1.0, eos_token_id=None, verbose=False):
    """
    prompt_ids: [batch, seq_len_inicial]
    Retorna: [batch, seq_len_inicial + tokens_gerados]
    """
    model.eval()
    context = prompt_ids.clone()

    for passo in range(max_novos_tokens):
        # Truncamento: sliding window sobre os últimos max_seq_len tokens
        context_window = context[:, -max_seq_len:]

        # Forward pass completo sobre o contexto atual
        logits = model(context_window)              # [batch, seq_len, vocab]
        next_token_logits = logits[:, -1, :]         # [batch, vocab] -- só a última posição

        # Escolher o próximo token (estratégia de sampling)
        next_token = escolher_proximo_token(next_token_logits, temperature, top_p)  # [batch, 1]

        # Concatenar ao contexto
        context = torch.cat([context, next_token], dim=1)

        if verbose:
            print(f"  passo {passo+1:2d}: contexto_len={context.shape[1]:2d}, "
                  f"token_gerado={next_token.item()}")

        # Condição de parada: EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            if verbose:
                print(f"  EOS encontrado no passo {passo+1}, parando.")
            break

    return context

# ========== 4. TESTANDO A GERAÇÃO ==========
print("\n4. TESTE: GERANDO A PARTIR DE UM PROMPT")
print("-" * 70)

prompt = torch.tensor([[5, 8, 3]])  # prompt fictício de 3 tokens
print(f"Prompt inicial: {prompt.tolist()} (shape {tuple(prompt.shape)})")

resultado = generate(
    model, prompt,
    max_novos_tokens=10,
    max_seq_len=MAX_SEQ_LEN,
    temperature=0.8,
    top_p=0.9,
    eos_token_id=EOS_TOKEN_ID,
    verbose=True,
)

print(f"\nSequência completa gerada: {resultado.tolist()}")
print(f"Shape final: {tuple(resultado.shape)}")

# ========== 5. TRUNCAMENTO DE CONTEXTO EM AÇÃO ==========
print("\n5. DEMONSTRANDO TRUNCAMENTO (sliding window)")
print("-" * 70)

prompt_longo = torch.randint(1, VOCAB_SIZE, (1, 20))  # já maior que MAX_SEQ_LEN=16
print(f"Prompt inicial já tem {prompt_longo.shape[1]} tokens (> max_seq_len={MAX_SEQ_LEN})")

resultado_longo = generate(
    model, prompt_longo,
    max_novos_tokens=5,
    max_seq_len=MAX_SEQ_LEN,
    temperature=1.0,
    top_p=1.0,
    eos_token_id=None,  # sem EOS, gera até max_novos_tokens
)
print(f"Contexto final: {resultado_longo.shape[1]} tokens")
print("A cada passo, o forward pass só viu os últimos "
      f"{MAX_SEQ_LEN} tokens da janela, nunca a sequência inteira.")

# ========== 6. COMPARANDO ESTRATÉGIAS NO MESMO PROMPT ==========
print("\n6. COMPARANDO GREEDY vs SAMPLING NO MESMO PROMPT")
print("-" * 70)

prompt_fixo = torch.tensor([[7, 2, 9]])

torch.manual_seed(42)
saida_greedy_1 = generate(model, prompt_fixo, 8, MAX_SEQ_LEN, temperature=0.0)
torch.manual_seed(123)  # seed diferente
saida_greedy_2 = generate(model, prompt_fixo, 8, MAX_SEQ_LEN, temperature=0.0)

print(f"Greedy (seed 42):  {saida_greedy_1.tolist()}")
print(f"Greedy (seed 123): {saida_greedy_2.tolist()}")
print("-> Devem ser IDÊNTICOS: greedy não depende de aleatoriedade.")

torch.manual_seed(42)
saida_sample_1 = generate(model, prompt_fixo, 8, MAX_SEQ_LEN, temperature=1.0, top_p=0.9)
torch.manual_seed(123)
saida_sample_2 = generate(model, prompt_fixo, 8, MAX_SEQ_LEN, temperature=1.0, top_p=0.9)

print(f"\nSampling (seed 42):  {saida_sample_1.tolist()}")
print(f"Sampling (seed 123): {saida_sample_2.tolist()}")
print("-> Podem ser DIFERENTES: sampling introduz aleatoriedade controlada.")

# ========== 7. BATCH GENERATION ==========
print("\n7. GERAÇÃO EM BATCH (múltiplos prompts simultâneos)")
print("-" * 70)

batch_prompts = torch.tensor([
    [4, 8, 1],
    [2, 9, 5],
    [7, 3, 6],
])
print(f"Batch de {batch_prompts.shape[0]} prompts, shape {tuple(batch_prompts.shape)}")

torch.manual_seed(42)
resultado_batch = generate(model, batch_prompts, max_novos_tokens=6,
                            max_seq_len=MAX_SEQ_LEN, temperature=0.8, top_p=0.9)

print(f"Shape final do batch: {tuple(resultado_batch.shape)}")
for i, seq in enumerate(resultado_batch.tolist()):
    print(f"  Prompt {i}: {seq}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Usar os logits de todas as posições, não só a última

```python
# ❌ Errado — pega logits da sequência inteira e tenta amostrar de todos
logits = model(context)                    # [batch, seq_len, vocab]
next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)  # shape errado!

# ✓ Certo — extrai apenas a última posição antes de amostrar
logits = model(context)                     # [batch, seq_len, vocab]
next_token_logits = logits[:, -1, :]         # [batch, vocab]
next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
```

Esquecer o `[:, -1, :]` é um dos bugs mais comuns ao implementar geração pela primeira vez — o forward pass sempre retorna previsões para todas as posições, mas só a última é "o próximo token depois do contexto inteiro".

### Erro 2: Não truncar o contexto, estourando max_seq_len

```python
# ❌ Errado — contexto cresce sem limite, eventualmente excede
# o max_seq_len para o qual o modelo foi treinado (positional
# embeddings não existem além desse limite -> erro ou lixo)
for _ in range(max_novos_tokens):
    logits = model(context)  # context pode crescer indefinidamente
    ...
    context = torch.cat([context, next_token], dim=1)

# ✓ Certo — sempre truncar para a janela máxima antes do forward
for _ in range(max_novos_tokens):
    context_window = context[:, -max_seq_len:]
    logits = model(context_window)
    ...
    context = torch.cat([context, next_token], dim=1)  # o histórico completo é mantido,
                                                          # mas só a janela é usada no forward
```

Note que ainda vale a pena manter o `context` completo (não truncado) na variável, mesmo truncando o que é passado ao modelo — assim você preserva o texto gerado inteiro para exibir ao usuário, mesmo que o modelo só "veja" uma janela dele.

### Erro 3: Esquecer `model.eval()` e `torch.no_grad()` na geração

```python
# ❌ Errado — modelo em modo treino (dropout ativo, ruído extra)
# e construindo grafo de gradientes desnecessariamente (gasta memória)
def generate(model, prompt_ids, max_novos_tokens, ...):
    for _ in range(max_novos_tokens):
        logits = model(context)
        ...

# ✓ Certo
@torch.no_grad()
def generate(model, prompt_ids, max_novos_tokens, ...):
    model.eval()  # desativa dropout, normaliza BatchNorm/LayerNorm em modo inferência
    for _ in range(max_novos_tokens):
        logits = model(context)
        ...
```

Sem `model.eval()`, camadas como `nn.Dropout` continuam zerando ativações aleatoriamente durante a geração, tornando a saída ainda mais ruidosa do que o sampling já é por natureza. Sem `torch.no_grad()`, cada forward pass constrói um grafo computacional para backward — que nunca será usado na geração — desperdiçando memória e tempo.

---

## Exercícios

### Exercício 32.1: Contando Forward Passes
Se você gera 50 tokens novos a partir de um prompt de 10 tokens, quantas chamadas de `model.forward()` acontecem no total (sem KV-cache)? E qual é o comprimento do contexto processado em cada uma dessas chamadas (liste os 3 primeiros e os 3 últimos)?

### Exercício 32.2: Implementando a Condição de Parada por Comprimento
Modifique a função `generate()` do experimento para também parar se o comprimento total do contexto (`context.shape[1]`) ultrapassar um `max_total_len` fornecido, mesmo que `max_novos_tokens` ainda não tenha sido atingido e o EOS não tenha aparecido.

### Exercício 32.3: Batch com EOS Assíncrono
No batch generation, um prompt pode gerar EOS antes dos outros. Escreva uma versão de `generate()` que rastreia, para cada sequência do batch, se ela já "terminou" (gerou EOS), e continua gerando para as outras sem sobrescrever os tokens já finalizados da que terminou (dica: mantenha uma máscara booleana `finished` de shape `[batch]`).

### Exercício 32.4: Medindo o Custo de Recomputação
Escreva um pequeno experimento que conta quantas vezes, no total ao longo de toda a geração (sem KV-cache), cada token individual do contexto é "reprocessado" pelo forward pass (ou seja, aparece na janela de contexto de uma chamada). Para uma geração de N tokens novos a partir de um prompt de tamanho P (assumindo N+P <= max_seq_len, sem truncamento), qual é a fórmula fechada para o total de "reprocessamentos"?

### Exercício 32.5: Sliding Window e Perda de Contexto
Configure `max_seq_len=8` e gere 20 tokens a partir de um prompt de 5 tokens. Verifique empiricamente que, a partir de um certo passo, os primeiros tokens do prompt original deixam de aparecer na janela de contexto passada ao modelo. Em qual passo isso acontece?

---

## Gabarito

### Exercício 32.1: Contando Forward Passes
```python
# 50 tokens novos = 50 chamadas de forward (uma por token gerado, sequencialmente)
#
# Comprimento do contexto em cada chamada (prompt inicial = 10 tokens):
# chamada 1:  contexto tem 10 tokens (o prompt original)
# chamada 2:  contexto tem 11 tokens
# chamada 3:  contexto tem 12 tokens
# ...
# chamada 48: contexto tem 57 tokens
# chamada 49: contexto tem 58 tokens
# chamada 50: contexto tem 59 tokens
#
# Fórmula geral: chamada k tem comprimento (10 + k - 1) tokens, para k=1..50
# Total processado ao longo de toda a geração (sem cache):
# soma_{k=1}^{50} (10 + k - 1) = 50*10 + soma_{k=0}^{49} k = 500 + 1225 = 1725 "tokens-forward"
```

### Exercício 32.2: Parada por Comprimento Total
```python
@torch.no_grad()
def generate(model, prompt_ids, max_novos_tokens, max_seq_len,
             temperature=1.0, top_p=1.0, eos_token_id=None,
             max_total_len=None):
    model.eval()
    context = prompt_ids.clone()

    for _ in range(max_novos_tokens):
        context_window = context[:, -max_seq_len:]
        logits = model(context_window)
        next_token_logits = logits[:, -1, :]
        next_token = escolher_proximo_token(next_token_logits, temperature, top_p)
        context = torch.cat([context, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # Nova condição: comprimento total absoluto
        if max_total_len is not None and context.shape[1] >= max_total_len:
            break

    return context
```

### Exercício 32.3: Batch com EOS Assíncrono
```python
@torch.no_grad()
def generate_batch_com_eos(model, prompt_ids, max_novos_tokens, max_seq_len,
                            temperature, top_p, eos_token_id):
    model.eval()
    context = prompt_ids.clone()
    batch_size = context.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool)

    for _ in range(max_novos_tokens):
        if finished.all():
            break

        context_window = context[:, -max_seq_len:]
        logits = model(context_window)
        next_token_logits = logits[:, -1, :]
        next_token = escolher_proximo_token(next_token_logits, temperature, top_p)  # [batch, 1]

        # Para sequências já finalizadas, força o próximo token a ser EOS
        # (efetivamente "congela" a sequência, só preenchendo com padding/EOS)
        next_token = torch.where(
            finished.unsqueeze(1), torch.full_like(next_token, eos_token_id), next_token
        )

        context = torch.cat([context, next_token], dim=1)

        # Atualiza quais sequências acabaram de terminar
        just_finished = (next_token.squeeze(1) == eos_token_id)
        finished = finished | just_finished

    return context
```

### Exercício 32.4: Fórmula de Reprocessamento
```python
# Para N tokens gerados a partir de um prompt de P tokens (sem truncamento):
# Na chamada k (k=1..N), o contexto tem (P + k - 1) tokens, todos reprocessados.
#
# Total de "tokens-forward" processados ao longo de toda a geração:
# soma_{k=1}^{N} (P + k - 1) = N*P + soma_{k=0}^{N-1} k = N*P + N*(N-1)/2
#
# Verificação empírica:
def contar_reprocessamentos(P, N):
    total = 0
    context_len = P
    for _ in range(N):
        total += context_len
        context_len += 1
    return total

P, N = 10, 50
formula = N*P + N*(N-1)//2
empirico = contar_reprocessamentos(P, N)
print(formula, empirico)  # 1725 1725 -- batem
# Isso é O(N^2 + N*P): cresce quadraticamente com o número de tokens gerados,
# exatamente o custo que o KV-cache elimina.
```

### Exercício 32.5: Sliding Window e Perda de Contexto
```python
import torch

MAX_SEQ_LEN = 8
prompt = torch.arange(1, 6).unsqueeze(0)  # tokens [1,2,3,4,5], shape [1,5]

context = prompt.clone()
for passo in range(1, 21):
    novo_token = torch.tensor([[100 + passo]])  # token fictício "gerado"
    context = torch.cat([context, novo_token], dim=1)

    janela = context[:, -MAX_SEQ_LEN:]
    tem_token_1 = (janela == 1).any().item()

    if not tem_token_1 and passo > 1:
        janela_anterior = context[:, -(MAX_SEQ_LEN+1):-1] if context.shape[1] > MAX_SEQ_LEN else None
        print(f"No passo {passo}, o token original '1' saiu da janela de contexto.")
        break

# Com prompt de 5 tokens e MAX_SEQ_LEN=8, cabem 3 tokens novos antes do
# token '1' (posição 0 do contexto) sair da janela dos últimos 8.
# No passo 4, o contexto tem 9 tokens; a janela dos últimos 8 já não inclui
# mais o token '1'.
```

---

## Desafios Avançados (Opcionais)

### Fixação 32.1: Implementando um KV-Cache Simplificado
Implemente uma versão simplificada de KV-cache para uma única camada de atenção (não o modelo inteiro): mantenha `K_cache` e `V_cache` como tensores que crescem a cada passo, e no forward de um novo token, compute apenas o Q/K/V dele, concatene K e V ao cache, e use o Q novo contra o cache completo. Compare o tempo de execução (usando `time.perf_counter()`) contra a versão sem cache, gerando 50 tokens.

### Fixação 32.2: Repetition Penalty no Loop de Geração
Integre uma penalidade de repetição (reduzir logits de tokens já presentes no contexto) dentro da função `generate()`, aplicada antes do sampling em cada passo. Compare a taxa de tokens repetidos com e sem a penalidade, em uma geração de 100 tokens.

### Fixação 32.3: Stop Sequences Customizadas
Além do EOS, implemente suporte a "stop sequences" — sequências de tokens específicas (ex: `[7, 2]`) que, se aparecerem no final do contexto gerado, interrompem a geração. Teste com uma stop sequence de 2 tokens.

### Fixação 32.4: Streaming de Tokens
Modifique `generate()` para ser um gerador Python (`yield` a cada token novo, em vez de retornar só no final), simulando como uma API de streaming (como a da OpenAI/Anthropic) entrega tokens ao cliente à medida que são gerados, em vez de esperar a resposta inteira.

### Fixação 32.5: Beam Search (Preview)
Pesquise o conceito de beam search: em vez de manter uma única sequência sendo gerada, mantenha as `B` sequências parciais de maior probabilidade acumulada a cada passo, expandindo cada uma e mantendo só as `B` melhores no total. Implemente uma versão simples com B=3 para um modelo pequeno e compare a saída com greedy decoding (B=1).

---

## Resumo

- **Loop autoregressivo**: gerar um token, concatenar ao contexto, repetir — cada iteração exige um forward pass completo sobre o contexto acumulado.
- **Treino vs. inferência**: treino processa a sequência inteira em paralelo com teacher forcing; inferência é sequencial, um token por vez, usando a própria saída do modelo como entrada seguinte.
- **Truncamento (sliding window)**: quando o contexto excede `max_seq_len`, mantemos apenas os últimos N tokens — o modelo "esquece" o início de contextos muito longos.
- **Condições de parada**: token EOS (parada natural) e comprimento máximo (limite de segurança) garantem que o loop sempre termina.
- **KV-caching**: tokens antigos não mudam seus K/V quando um novo token é adicionado; cachear evita recomputação redundante e reduz o custo de O(N²) para O(N) ao longo da geração.
- **Batch generation**: gerar para múltiplos prompts em paralelo aproveita melhor a GPU, mas exige padding e rastreamento individual de EOS por sequência.

Com isso, você tem todas as peças — atenção, blocos Transformer, treinamento, sampling e o loop de geração — prontas para montar um modelo completo, funcional, de ponta a ponta.

Próximo capítulo: **Projeto LLMScratch — Modelo Completo** — onde juntamos tudo isso em uma implementação única, treinável e capaz de gerar texto de verdade.

---

**Próximo**: [Capítulo 33: Projeto LLMScratch — Modelo Completo](33_projeto_james_jr.md)
