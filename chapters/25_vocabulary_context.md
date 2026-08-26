# Capítulo 25: Vocabulary e Context Windows

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o trade-off entre tamanho de vocabulário e número de parâmetros do modelo
2. Explicar o que é a context window (`max_seq_len`) e por que ela é limitada
3. Calcular o custo quadrático de atenção em função do comprimento de sequência
4. Descrever, em alto nível, técnicas modernas para contextos longos (sliding window, atenção esparsa)
5. Escolher `vocab_size` e `max_seq_len` na prática para um projeto pequeno

---

## Por Que Isso Importa

No capítulo anterior, aprendemos a transformar texto em token IDs usando BPE. Mas deixamos duas perguntas em aberto: quantos tokens diferentes o vocabulário deveria ter? E quantos tokens de contexto o modelo deveria conseguir processar de uma vez?

Essas não são perguntas triviais — são decisões de arquitetura com consequências reais em custo. Pense assim: `vocab_size` é como decidir quantas palavras diferentes vão caber no dicionário da sua biblioteca — um dicionário maior deixa cada "entrada" mais específica (menos ambiguidade, sequências mais curtas), mas o dicionário em si ocupa mais espaço nas prateleiras. Já `max_seq_len` é como decidir quantas páginas de um livro o leitor consegue ter abertas na mesa ao mesmo tempo, revisitando qualquer uma delas instantaneamente — quanto mais páginas abertas, mais rico o contexto, mas também mais mesa você precisa (e a mesa cresce em área, não em comprimento — como veremos, o custo é quadrático).

Essas duas escolhas — `vocab_size` e `max_seq_len` — aparecem literalmente na primeira linha do `CONFIG` de qualquer projeto de LLM, incluindo o LLMScratch que vamos construir no Capítulo 33. Elas não são detalhes de implementação: são decisões que determinam quantos parâmetros seu modelo tem, quanta memória de GPU ele consome, e quão rápido ele treina. Entender esse trade-off agora evita que você escolha valores arbitrários mais tarde e se surpreenda com um modelo que não cabe na memória — ou que treina uma ordem de magnitude mais devagar do que precisaria.

---

## O Trade-off do Vocabulary Size

### Onde o vocab_size Aparece na Arquitetura

Relembrando a arquitetura fim a fim do Capítulo 23, o `vocab_size` afeta diretamente duas camadas do modelo:

**1. A tabela de embeddings de entrada** (Capítulo 07): uma matriz $E \in \mathbb{R}^{[\text{vocab\_size}, d]}$, onde $d$ é a dimensão de embedding. Cada linha é o vetor denso que representa um token.

**2. A cabeça de saída (output head)**: a camada linear final que projeta o vetor de dimensão $d$ de volta para um vetor de tamanho `vocab_size` — os *logits*, um escore para cada token possível do vocabulário (veremos isso em detalhe no Capítulo 26). Essa camada é $W_{\text{out}} \in \mathbb{R}^{[d, \text{vocab\_size}]}$.

Repare que essas duas matrizes têm exatamente `vocab_size * d` parâmetros **cada uma**. Em modelos pequenos e médios, isso frequentemente é uma fração enorme do total de parâmetros do modelo inteiro — maior até do que todos os blocos transformer somados.

### O Trade-off

**Vocabulário maior:**
- Sequências mais curtas (cada token "carrega" mais informação — ex: "correu" é 1 token ao invés de 6).
- Embedding table e output head maiores → mais parâmetros, mais memória, mais compute na camada final (o cálculo de logits é uma multiplicação de matriz $[n, d] \times [d, \text{vocab\_size}]$ — cresce linearmente com `vocab_size`).

**Vocabulário menor:**
- Sequências mais longas (cada palavra vira mais tokens, como vimos no Capítulo 24 com tokenização por caractere).
- Embedding table e output head menores → menos parâmetros nessas duas camadas.
- Mas sequências mais longas custam mais caro na atenção — como veremos a seguir, esse custo cresce **quadraticamente**, não linearmente.

Ou seja: reduzir `vocab_size` economiza parâmetros nas pontas do modelo (embedding e output head), mas pode aumentar drasticamente o custo computacional no meio (todas as camadas de atenção), porque as sequências ficam mais longas. Não existe resposta universal — é sempre um equilíbrio específico ao problema.

Modelos de produção reais usam vocabulários na faixa de 32.000 a 100.000+ tokens (GPT-2 usa ~50.257, LLaMA 2 usa 32.000, GPT-4 usa ~100.000). Projetos educacionais pequenos, como o nosso, usam vocabulários bem menores (na ordem de 1.000), porque o corpus de treino também é pequeno — não faz sentido ter um vocabulário de 50.000 tokens para um corpus de algumas páginas de texto.

---

## Context Window: O que É e Por Que É Limitada

### Definição

A **context window** (também chamada `max_seq_len` ou `block_size`) é o número máximo de tokens que o modelo consegue processar de uma vez em uma única passada. Se o seu `max_seq_len` é 128, o modelo literalmente não consegue "ver" o token 129 ao processar o token 200 — ele simplesmente não existe na janela que o modelo enxerga.

Isso não é um limite arbitrário de engenharia que alguém poderia simplesmente aumentar de graça. É uma consequência direta de como a atenção funciona.

### O Custo Quadrático da Atenção

Lembre-se do Capítulo 10: a atenção computa uma matriz de scores $\text{scores} = QK^T \in \mathbb{R}^{[n, n]}$, onde $n$ é o comprimento de sequência. Essa matriz tem **$n^2$ entradas** — toda posição precisa computar um score contra toda outra posição.

Isso significa:
- **Tempo de computação**: cresce com $O(n^2 \cdot d)$ (o produto $QK^T$ envolve $n^2$ produtos escalares, cada um de dimensão $d$).
- **Memória**: a matriz de attention weights sozinha ocupa $O(n^2)$ de memória — e isso é *por cabeça de atenção, por camada, por exemplo no batch*.

Vamos tornar isso concreto. Se dobrarmos `max_seq_len` de 128 para 256:
- O comprimento de sequência dobra (2x).
- A matriz de atenção vai de $128^2 = 16.384$ entradas para $256^2 = 65.536$ entradas — **4x mais**, não 2x.

Isso é o que "quadrático" significa na prática: dobrar o contexto quadruplica o custo de memória e tempo da atenção. É por isso que aumentar a context window não é uma decisão barata — é o motivo pelo qual modelos de produção investem enormes esforços de engenharia (Capítulo 34 toca nisso) só para conseguir contextos de 32K, 128K ou mais tokens.

### Memória Total da Matriz de Atenção

Para sermos precisos, o tamanho em memória da matriz de attention weights, para um único exemplo, é:

$$\text{memória} = \text{num\_heads} \times n^2 \times \text{bytes\_por\_elemento}$$

Onde `num_heads` é o número de cabeças de atenção (Capítulo 17) e `bytes_por_elemento` é 4 para float32 ou 2 para float16/bfloat16. Multiplique ainda pelo `batch_size` e pelo número de camadas transformer se quiser o total do forward pass inteiro (embora, tipicamente, a matriz de uma camada seja liberada da memória antes da próxima ser computada, durante inferência — durante treino, muitas vezes são todas mantidas para o backward pass).

---

## Técnicas Modernas para Contextos Longos (Visão Geral)

O custo quadrático é uma barreira física real, e uma parte ativa da pesquisa em LLMs busca contorná-la. Sem entrar em detalhes de implementação (isso foge do escopo deste livro introdutório), vale saber que essas direções existem:

- **Sliding window attention**: cada token só presta atenção a uma janela fixa de tokens vizinhos (ex: os últimos 4096), ao invés de toda a sequência. Reduz o custo de $O(n^2)$ para $O(n \cdot w)$, onde $w$ é o tamanho da janela — linear em $n$.
- **Atenção esparsa (sparse attention)**: ao invés de computar todos os $n^2$ pares, computa-se apenas um subconjunto estratégico de pares (por exemplo, um padrão que mistura atenção local com alguns tokens "globais" que todos podem ver).
- **Flash Attention**: não reduz o número de operações matemáticas, mas reorganiza o cálculo para ser muito mais eficiente em memória de GPU, evitando materializar a matriz $n \times n$ inteira de uma vez.
- **Compressão/resumo de contexto**: técnicas que resumem ou descartam partes antigas do contexto à medida que a conversa cresce, ao invés de manter tudo.

Essas técnicas são o motivo pelo qual modelos como GPT-4 e Claude conseguem processar dezenas de milhares (ou centenas de milhares) de tokens de contexto, apesar do custo quadrático fundamental da atenção "vanilla" que implementamos nos capítulos anteriores. O LLMScratch, sendo educacional, usa atenção completa (sem essas otimizações) — é mais simples de entender, e para os `max_seq_len` pequenos que usaremos, o custo é perfeitamente administrável.

---

## Escolhendo vocab_size e max_seq_len na Prática

Para um projeto pequeno como o LLMScratch (Capítulo 33), a escolha é guiada por restrições práticas, não por benchmarks de produção:

- **Corpus pequeno** (algumas páginas a alguns capítulos de texto): um `vocab_size` de **1.000** já é suficiente para cobrir bem o vocabulário de um texto em português de tamanho modesto, treinado com BPE. Usar 50.000 seria um desperdício — a maioria dos tokens do vocabulário nunca apareceria no treino, e o embedding table ficaria enorme e majoritariamente não-treinado.
- **Hardware modesto** (CPU ou GPU de laptop, ex: Apple Silicon MPS): um `max_seq_len` de **128** mantém o custo de atenção administrável ($128^2 = 16.384$ entradas por cabeça, por camada — trivial para hardware moderno), enquanto ainda permite ao modelo enxergar frases inteiras e algum contexto entre frases.
- **Regra prática geral**: comece pequeno em ambos os parâmetros. Se o modelo estiver truncando frases importantes (perdendo contexto necessário), aumente `max_seq_len`. Se o modelo estiver gerando muitos tokens `<unk>` ou fragmentando palavras comuns demais, aumente `vocab_size`. Ambos são fáceis de ajustar e retreinar em um projeto pequeno — a decisão só fica cara em escala de produção.

No Capítulo 33, usaremos exatamente `vocab_size=1000` e `max_seq_len=128` como parte do `CONFIG` do LLMScratch. Vamos agora quantificar concretamente o que essas escolhas custam em parâmetros e memória.

---

## Experimento: Custo de Parâmetros e Memória

```python
import torch

print("=" * 70)
print("EXPERIMENTO: Vocab Size, Context Window e Custo Computacional")
print("=" * 70)

# ========== 1. PARÂMETROS DA EMBEDDING TABLE E OUTPUT HEAD ==========
print("\n1. PARÂMETROS: EMBEDDING TABLE E OUTPUT HEAD")
print("-" * 70)

d_model = 128  # dimensão de embedding, fixa para esta comparação

vocab_sizes = [500, 1_000, 5_000, 10_000, 50_000, 100_000]

print(f"{'vocab_size':>12} | {'embedding_params':>18} | {'output_head_params':>19} | {'total':>12}")
print("-" * 70)

for vs in vocab_sizes:
    embedding_params = vs * d_model
    output_head_params = d_model * vs  # W_out: [d_model, vocab_size]
    total = embedding_params + output_head_params
    print(f"{vs:>12,} | {embedding_params:>18,} | {output_head_params:>19,} | {total:>12,}")

print(f"\n(d_model fixo em {d_model} para esta comparação)")
print("Note: embedding + output head crescem LINEARMENTE com vocab_size,")
print("mas em modelos pequenos podem dominar o total de parâmetros.")


# ========== 2. COMPARANDO COM O RESTO DO MODELO ==========
print("\n2. FRAÇÃO DO MODELO OCUPADA POR EMBEDDING + OUTPUT HEAD")
print("-" * 70)

def params_bloco_transformer(d_model, d_ff, num_heads=4):
    """Estimativa simplificada de parâmetros de 1 bloco transformer."""
    # Atenção: W_Q, W_K, W_V, W_O -> 4 matrizes [d_model, d_model]
    params_atencao = 4 * (d_model * d_model)
    # MLP: expansão d_model -> d_ff -> d_model
    params_mlp = 2 * (d_model * d_ff)
    # LayerNorms (gamma + beta), desprezível, mas incluímos por completude
    params_ln = 2 * (2 * d_model)
    return params_atencao + params_mlp + params_ln

num_layers = 4
d_ff = d_model * 4
params_por_bloco = params_bloco_transformer(d_model, d_ff)
params_transformer_total = params_por_bloco * num_layers

print(f"Config do corpo do modelo: d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}")
print(f"Parâmetros por bloco transformer: {params_por_bloco:,}")
print(f"Parâmetros totais do corpo (todas as camadas): {params_transformer_total:,}")

print(f"\n{'vocab_size':>12} | {'embed+head':>12} | {'corpo':>12} | {'% do total em embed+head':>26}")
print("-" * 70)
for vs in vocab_sizes:
    embed_head = 2 * vs * d_model
    total = embed_head + params_transformer_total
    fracao = 100 * embed_head / total
    print(f"{vs:>12,} | {embed_head:>12,} | {params_transformer_total:>12,} | {fracao:>25.1f}%")

print("\nObservação: com vocab_size pequeno, embedding+head é uma fração")
print("pequena do total. Com vocab_size grande, pode DOMINAR o modelo inteiro.")


# ========== 3. CUSTO DE MEMÓRIA DA MATRIZ DE ATENÇÃO ==========
print("\n3. MEMÓRIA DA MATRIZ DE ATENÇÃO PARA DIFERENTES seq_len")
print("-" * 70)

seq_lens = [32, 64, 128, 256, 512, 1024]
num_heads = 4
bytes_por_float32 = 4

print(f"{'seq_len':>10} | {'entradas (n^2)':>16} | {'memoria/head (KB)':>18} | {'memoria total (KB)':>19}")
print("-" * 70)

for n in seq_lens:
    entradas = n * n
    memoria_por_head_bytes = entradas * bytes_por_float32
    memoria_por_head_kb = memoria_por_head_bytes / 1024
    memoria_total_kb = memoria_por_head_kb * num_heads
    print(f"{n:>10} | {entradas:>16,} | {memoria_por_head_kb:>18,.1f} | {memoria_total_kb:>19,.1f}")

print(f"\n(assumindo {num_heads} cabeças de atenção, float32, 1 exemplo, 1 camada)")
print("Note o crescimento QUADRÁTICO: dobrar seq_len quadruplica a memória.")


# ========== 4. VERIFICANDO COM TENSORES REAIS ==========
print("\n4. VERIFICAÇÃO COM TENSORES REAIS (PyTorch)")
print("-" * 70)

torch.manual_seed(42)

for n in [128, 256]:
    Q = torch.randn(num_heads, n, d_model // num_heads)
    K = torch.randn(num_heads, n, d_model // num_heads)
    scores = Q @ K.transpose(-2, -1)  # [num_heads, n, n]
    memoria_bytes = scores.element_size() * scores.nelement()
    print(f"seq_len={n:>4}: scores.shape={tuple(scores.shape)}, "
          f"memoria real={memoria_bytes / 1024:.1f} KB")

razao = (256 ** 2) / (128 ** 2)
print(f"\nRazão de memória (256 vs 128): {razao:.1f}x")
print("Confirma: dobrar seq_len -> 4x a memória da matriz de atenção.")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_vocab_context.py
```

Você deve observar dois padrões centrais: (1) os parâmetros de embedding + output head crescem **linearmente** com `vocab_size`, mas dominam o total quando o corpo do modelo é pequeno; (2) a memória da matriz de atenção cresce **quadraticamente** com `seq_len` — dobrar o contexto sempre quadruplica esse custo específico.

---

## Erros Comuns

### Erro 1: Escolher vocab_size desproporcional ao corpus

```python
# ❌ Errado: vocab_size gigante para um corpus pequeno
CONFIG = {
    "vocab_size": 50_000,  # a maioria nunca vai aparecer no treino!
    ...
}
# Resultado: embedding table enorme, maioria das linhas nunca recebe
# gradiente significativo, desperdício de memória e parâmetros.

# ✓ Certo: dimensionar vocab_size ao tamanho real do corpus
CONFIG = {
    "vocab_size": 1_000,  # adequado para um corpus de poucas páginas
    ...
}
```

### Erro 2: Aumentar max_seq_len sem considerar o custo quadrático

```python
# ❌ Errado: aumentar seq_len "só um pouco" sem fazer as contas
max_seq_len = 128
# "vou aumentar para 1024, deve ser só 8x mais lento"
max_seq_len = 1024  # na verdade é 64x mais memória na atenção! (8^2)

# ✓ Certo: lembrar que o custo é quadrático, não linear
fator_seq = 1024 / 128  # 8x mais longo
fator_memoria_atencao = fator_seq ** 2  # 64x mais memória
print(f"Aumentar seq_len em {fator_seq}x custa {fator_memoria_atencao}x em memória de atenção")
```

### Erro 3: Truncar sequências silenciosamente sem avisar

```python
# ❌ Errado: truncar sem logging, perdendo informação sem perceber
tokens = tokens[:max_seq_len]  # o resto do texto simplesmente desaparece

# ✓ Certo: ao menos logar/avisar quando há truncamento
if len(tokens) > max_seq_len:
    print(f"Aviso: sequência truncada de {len(tokens)} para {max_seq_len} tokens")
    tokens = tokens[:max_seq_len]
```

---

## Exercícios

### Exercício 25.1: Parâmetros da Embedding Table
Calcule o número de parâmetros de uma embedding table com `vocab_size=2000` e `d_model=64`.

### Exercício 25.2: Memória da Matriz de Atenção
Para `seq_len=256`, `num_heads=8`, float32, calcule a memória (em KB) da matriz de attention weights de uma única camada, um único exemplo.

### Exercício 25.3: Razão de Custo
Se você aumentar `max_seq_len` de 64 para 512, por qual fator a memória da matriz de atenção aumenta?

### Exercício 25.4: Fração do Modelo
Dado `vocab_size=1000`, `d_model=128`, e um corpo transformer com 500.000 parâmetros, calcule que fração do total de parâmetros do modelo é ocupada por embedding + output head.

### Exercício 25.5: Escolha Justificada
Para um corpus de 10 páginas de texto em português, proponha um `vocab_size` e um `max_seq_len` razoáveis, justificando cada escolha em 1-2 frases.

---

## Gabarito

### Exercício 25.1: Parâmetros da Embedding Table
```python
vocab_size = 2000
d_model = 64
params = vocab_size * d_model
print(f"Parâmetros: {params:,}")  # 128,000
```

### Exercício 25.2: Memória da Matriz de Atenção
```python
seq_len = 256
num_heads = 8
bytes_float32 = 4

entradas_por_head = seq_len * seq_len
memoria_bytes = entradas_por_head * bytes_float32 * num_heads
memoria_kb = memoria_bytes / 1024
print(f"Memória: {memoria_kb:.1f} KB")  # 2048.0 KB (2 MB)
```

### Exercício 25.3: Razão de Custo
```python
seq_len_antigo = 64
seq_len_novo = 512

fator_seq = seq_len_novo / seq_len_antigo  # 8x
fator_memoria = fator_seq ** 2  # 64x

print(f"Sequência {fator_seq}x mais longa -> memória de atenção {fator_memoria}x maior")
# 64x
```

### Exercício 25.4: Fração do Modelo
```python
vocab_size = 1000
d_model = 128
params_corpo = 500_000

embed_head = 2 * vocab_size * d_model  # embedding + output head
total = embed_head + params_corpo
fracao = 100 * embed_head / total

print(f"Embedding+head: {embed_head:,}")   # 256,000
print(f"Total: {total:,}")                  # 756,000
print(f"Fração: {fracao:.1f}%")             # 33.9%
```

### Exercício 25.5: Escolha Justificada
```python
# Resposta esperada (exemplo):
vocab_size = 800
# Justificativa: 10 páginas de texto têm um vocabulário relativamente
# restrito; um vocab_size próximo de 500-1000 tokens de BPE cobre bem
# palavras comuns e ainda decompõe palavras raras em subpartes, sem
# desperdiçar parâmetros em uma embedding table majoritariamente inútil.

max_seq_len = 128
# Justificativa: frases e pequenos parágrafos raramente passam de 100-150
# tokens; 128 dá folga para a maioria dos exemplos de treino sem pagar o
# custo quadrático de uma janela muito maior, que seria desnecessária
# para um corpus deste tamanho.

print(f"vocab_size={vocab_size}, max_seq_len={max_seq_len}")
```

---

## Desafios Avançados (Opcionais)

### Fixação 25.1: Ponto de Equilíbrio
Escreva código que encontra o `vocab_size` no qual embedding+output head passam a ocupar exatamente 50% dos parâmetros totais do modelo, dado um corpo transformer de tamanho fixo. Como esse ponto muda se você aumentar `num_layers`?

### Fixação 25.2: Custo Total de um Forward Pass
Estenda o experimento para estimar o custo total (em FLOPs, não só memória) de um forward pass completo, considerando `num_layers`, `d_model`, `seq_len` e `vocab_size`. Qual termo domina quando `seq_len` é muito maior que `d_model`? E quando é muito menor?

### Fixação 25.3: Sliding Window Simulado
Implemente uma versão simplificada de sliding window attention (uma máscara adicional que zera pesos de atenção fora de uma janela de tamanho `w` ao redor de cada posição) e compare a memória economizada versus atenção completa, para `seq_len=1024` e `w=128`.

### Fixação 25.4: Vocab Size e Perplexidade
(Requer treinar um modelo pequeno — adiantando os Capítulos 26-29) Treine o mesmo modelo com `vocab_size=200` e `vocab_size=2000` no mesmo corpus. Compare o número de épocas necessárias para convergir e a perplexidade final. O que muda?

### Fixação 25.5: Batch Size e Memória Total
Generalize o cálculo de memória da matriz de atenção do experimento para incluir `batch_size` e `num_layers`. Para que combinação de `batch_size`, `seq_len` e `num_layers` a memória total da atenção ultrapassaria 1 GB (float32)?

---

## Resumo

- **vocab_size** controla o tamanho da embedding table e da output head — parâmetros crescem linearmente, mas podem dominar modelos pequenos.
- **Vocabulário maior** = sequências mais curtas, mas mais parâmetros nas camadas de entrada/saída.
- **Vocabulário menor** = sequências mais longas, o que empurra custo para as camadas de atenção.
- **Context window (max_seq_len)**: limite de tokens que o modelo processa de uma vez, imposto pelo custo quadrático $O(n^2)$ da atenção.
- **Dobrar seq_len quadruplica** o custo de tempo e memória da atenção — a razão pela qual contextos longos são caros.
- **Técnicas modernas** (sliding window, atenção esparsa, Flash Attention) existem para driblar esse custo quadrático em modelos de produção com contextos muito longos.
- **Na prática**, para um projeto pequeno como o LLMScratch, valores modestos (`vocab_size=1000`, `max_seq_len=128`) equilibram bem capacidade expressiva e custo computacional.

Próximo capítulo: **Logits e Cross-Entropy Loss** — como transformamos a saída bruta do modelo em uma medida de erro que podemos otimizar.

---

**Próximo**: [Capítulo 26: Logits e Cross-Entropy Loss](26_logits_e_loss.md)
