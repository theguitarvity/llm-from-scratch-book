# Capítulo 24: Tokenização

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o que é tokenização e por que é o primeiro passo de qualquer pipeline de LLM
2. Comparar tokenização por caractere, por palavra e por subpalavra (subword)
3. Explicar o algoritmo Byte-Pair Encoding (BPE) passo a passo
4. Implementar um tokenizer BPE simplificado do zero, com treino, encode e decode
5. Reconhecer o papel de tokens especiais (`<pad>`, `<unk>`, `<bos>`, `<eos>`)

---

## Por Que Isso Importa

Até agora, neste livro, todo exemplo começou com uma matriz de embeddings já pronta: `X = [[0.1, -0.2, ...], ...]`. Convenientemente, pulamos uma pergunta enorme: como um texto como `"O gato dormia"` vira aquela matriz de números?

A resposta tem duas etapas, e este capítulo cobre a primeira delas. Primeiro, o texto precisa virar uma sequência de **inteiros** — os chamados *token IDs*. Só depois esses inteiros são usados como índices para buscar vetores na tabela de embeddings (Capítulo 07). Sem essa primeira etapa, a rede neural nem sabe por onde começar: redes neurais multiplicam matrizes, não leem letras.

Pense em tokenização como o trabalho de um bibliotecário que organiza um catálogo. Ele não guarda "o livro sobre gatos" — ele atribui um número de catálogo (`4271`) e a partir daí todo o sistema (empréstimos, buscas, prateleiras) trabalha com aquele número. A tokenização faz exatamente isso com pedaços de texto: cada "pedaço" (token) ganha um número inteiro fixo, definido de antemão em um vocabulário.

A escolha de *como* dividir o texto em pedaços — por letra, por palavra inteira, ou por algo no meio do caminho — tem consequências profundas em todo o resto do pipeline: o tamanho da tabela de embeddings, o comprimento das sequências que a atenção processa (lembre-se do custo O(n²) do Capítulo 10), e a capacidade do modelo de lidar com palavras que nunca viu no treino. É essa tensão que vamos explorar agora, terminando com a solução que praticamente todo modelo moderno usa: Byte-Pair Encoding.

---

## Estratégias de Tokenização

Existem três estratégias clássicas. Vamos comparar todas com a mesma frase de exemplo: `"o gato correu"`.

### Tokenização por Caractere

Cada caractere vira um token.

```
"o gato correu" -> ['o', ' ', 'g', 'a', 't', 'o', ' ', 'c', 'o', 'r', 'r', 'e', 'u']
```

- **Vocabulário**: pequeníssimo (26 letras + pontuação + espaço ≈ 50-100 símbolos).
- **Sequências**: muito longas — 13 tokens para 3 palavras curtas.
- **Palavras novas**: nunca é um problema. Qualquer palavra é só uma sequência de caracteres já conhecidos.

O problema é o comprimento. Como a atenção custa O(n²) em tempo e memória (você viu isso no Capítulo 10 e vai formalizar no próximo capítulo), sequências 5-8x mais longas tornam o treinamento proibitivamente caro para textos reais.

### Tokenização por Palavra

Cada palavra vira um token.

```
"o gato correu" -> ['o', 'gato', 'correu']
```

- **Vocabulário**: enorme. Um vocabulário que cobre um idioma decentemente precisa de centenas de milhares de entradas (considerando plurais, conjugações, nomes próprios).
- **Sequências**: curtas — ótimo para o custo de atenção.
- **Palavras novas**: catastrófico. Qualquer palavra fora do vocabulário vira `<unk>` (unknown), e o modelo perde toda a informação sobre ela. Um nome próprio novo, um neologismo, um erro de digitação — tudo vira o mesmo token genérico.

### Tokenização por Subpalavra (Subword)

O meio-termo: pedaços menores que palavras, mas maiores que caracteres — normalmente sílabas ou fragmentos frequentes.

```
"o gato correu" -> ['o', 'gato', 'corr', 'eu']
```

Palavras comuns (`gato`) permanecem como um único token. Palavras raras ou desconhecidas (`correu`, ou hipoteticamente `tokenização`) são quebradas em pedaços conhecidos: `token`, `iza`, `ção`. Isso dá ao modelo o melhor dos dois mundos:

| Estratégia | Vocab size | Comprimento de sequência | Lida com palavra nova? |
|---|---|---|---|
| Caractere | Muito pequeno | Muito longo | Sim, trivialmente |
| Palavra | Muito grande | Curto | Não — vira `<unk>` |
| Subpalavra | Médio (ajustável) | Médio | Sim — quebra em pedaços conhecidos |

É por isso que **todo modelo moderno relevante** (GPT, LLaMA, Gemini, etc.) usa alguma variante de tokenização por subpalavra. E o algoritmo mais usado para *construir* esse vocabulário de subpalavras é o **Byte-Pair Encoding (BPE)**.

---

## Byte-Pair Encoding (BPE): O Algoritmo

BPE nasceu como algoritmo de compressão de dados (1994) e foi adaptado para NLP em 2015. A ideia central é simples: **comece granular, e mescle iterativamente os pares mais frequentes.**

### O Algoritmo, Passo a Passo

1. **Inicialize** o vocabulário com todos os caracteres únicos do corpus de treino.
2. **Represente** cada palavra do corpus como uma sequência de caracteres (mais um marcador de fim de palavra, geralmente).
3. **Conte** a frequência de todos os pares de tokens adjacentes em todo o corpus.
4. **Mescle** o par mais frequente em um único novo token. Adicione esse novo token ao vocabulário.
5. **Repita** os passos 3-4 até atingir o `vocab_size` desejado (ou até não haver mais pares repetidos).

O resultado é um vocabulário onde os tokens mais "úteis" — porque aparecem com frequência — acabam sendo pedaços maiores (sílabas comuns, até palavras inteiras), enquanto pedaços raros continuam fragmentados em caracteres individuais.

### Exemplo Manual: 3 Iterações de BPE

Vamos rodar o algoritmo à mão em um mini-corpus. Usamos `_` para marcar fim de palavra (convenção comum, evita mesclar através de fronteiras de palavras).

Corpus de treino (com contagem de frequência de cada palavra):

```
"gato_"   x 5
"gata_"   x 3
"gado_"   x 2
```

**Estado inicial**: cada palavra é uma sequência de caracteres.

```
g a t o _    (freq 5)
g a t a _    (freq 3)
g a d o _    (freq 2)
```

Vocabulário inicial (caracteres únicos): `{g, a, t, o, d, _}`

#### Iteração 1: contar pares adjacentes

Contamos todos os pares (bigrama de tokens) ponderados pela frequência da palavra:

```
(g,a): aparece em "gato_"(5) + "gata_"(3) + "gado_"(2) = 10
(a,t): aparece em "gato_"(5) + "gata_"(3) = 8
(t,o): aparece em "gato_"(5) = 5
(t,a): aparece em "gata_"(3) = 3
(o,_): aparece em "gato_"(5) + "gado_"(2) = 7
(a,_): aparece em "gata_"(3) = 3
(a,d): aparece em "gado_"(2) = 2
(d,o): aparece em "gado_"(2) = 2
```

O par mais frequente é **(g, a)** com contagem 10. Mesclamos em um novo token `ga`.

```
Novo vocabulário: {g, a, t, o, d, _, ga}

ga t o _    (freq 5)
ga t a _    (freq 3)
ga d o _    (freq 2)
```

#### Iteração 2: contar pares de novo

```
(ga,t): 5 + 3 = 8
(t,o): 5
(t,a): 3
(o,_): 5 + 2 = 7
(a,_): 3
(ga,d): 2
(d,o): 2
```

O par mais frequente agora é **(ga, t)** com contagem 8. Mesclamos em `gat`.

```
Novo vocabulário: {g, a, t, o, d, _, ga, gat}

gat o _    (freq 5)
gat a _    (freq 3)
ga d o _   (freq 2)
```

#### Iteração 3: contar pares de novo

```
(gat,o): 5
(o,_): 5 + 2 = 7
(gat,a): 3
(a,_): 3
(ga,d): 2
(d,o): 2
```

O par mais frequente é **(o, _)** com contagem 7. Mesclamos em `o_`.

```
Novo vocabulário: {g, a, t, o, d, _, ga, gat, o_}

gat o_    (freq 5)
gat a _   (freq 3)
ga d o_   (freq 2)
```

Repare o padrão: em 3 iterações, `"gato_"` já foi reduzido de 5 tokens (`g a t o _`) para 2 tokens (`gat o_`). Se continuássemos, `gat` provavelmente se fundiria ainda mais (por exemplo, com `o_` e `a_`, formando `gato_` e `gata_` como tokens únicos, já que são as palavras mais frequentes do corpus). O algoritmo aprende, de forma puramente estatística, que "gato" é uma unidade que vale a pena guardar inteira — sem que ninguém tenha lhe dito o que é um "gato".

---

## Experimento: Implementando BPE do Zero

Vamos implementar um tokenizer BPE completo: treino (aprender os merges a partir de um corpus), encode (texto -> IDs) e decode (IDs -> texto).

```python
import random
from collections import defaultdict, Counter

random.seed(42)

print("=" * 70)
print("EXPERIMENTO: Tokenizador BPE do Zero")
print("=" * 70)

# ========== 1. CORPUS DE TREINO ==========
print("\n1. CORPUS DE TREINO")
print("-" * 70)

corpus = [
    "o gato correu",
    "o gato dormiu",
    "a gata correu",
    "o gado pastou",
    "a gata dormiu",
]

print("Frases de treino:")
for frase in corpus:
    print(f"  {frase!r}")


# ========== 2. PRÉ-PROCESSAMENTO: PALAVRAS -> SEQUÊNCIAS DE CARACTERES ==========
print("\n2. PRÉ-PROCESSAMENTO")
print("-" * 70)

def contar_palavras(corpus):
    """Conta a frequência de cada palavra no corpus."""
    contagem = Counter()
    for frase in corpus:
        for palavra in frase.split():
            contagem[palavra] += 1
    return contagem

freq_palavras = contar_palavras(corpus)
print(f"Frequência de palavras: {dict(freq_palavras)}")

# Cada palavra vira uma tupla de caracteres + marcador de fim de palavra "_"
def palavra_para_simbolos(palavra):
    return tuple(list(palavra) + ["_"])

vocab_palavras = {
    palavra_para_simbolos(p): freq
    for p, freq in freq_palavras.items()
}

print("\nPalavras representadas como sequências de símbolos:")
for simbolos, freq in vocab_palavras.items():
    print(f"  {simbolos} (freq={freq})")


# ========== 3. TREINO DO BPE ==========
print("\n3. TREINO DO BPE (aprendendo merges)")
print("-" * 70)

def contar_pares(vocab_palavras):
    """Conta a frequência de cada par de símbolos adjacentes."""
    pares = defaultdict(int)
    for simbolos, freq in vocab_palavras.items():
        for i in range(len(simbolos) - 1):
            par = (simbolos[i], simbolos[i + 1])
            pares[par] += freq
    return pares

def aplicar_merge(vocab_palavras, par):
    """Aplica um merge em todas as palavras do vocabulário."""
    novo_vocab = {}
    a, b = par
    novo_simbolo = a + b
    for simbolos, freq in vocab_palavras.items():
        nova_seq = []
        i = 0
        while i < len(simbolos):
            if i < len(simbolos) - 1 and simbolos[i] == a and simbolos[i + 1] == b:
                nova_seq.append(novo_simbolo)
                i += 2
            else:
                nova_seq.append(simbolos[i])
                i += 1
        novo_vocab[tuple(nova_seq)] = freq
    return novo_vocab

NUM_MERGES = 10
merges = []  # lista ordenada de merges aprendidos (a, b) -> ab

vocab_atual = dict(vocab_palavras)

for passo in range(NUM_MERGES):
    pares = contar_pares(vocab_atual)
    if not pares:
        print(f"  Parando na iteração {passo}: sem mais pares para mesclar.")
        break

    melhor_par = max(pares, key=pares.get)
    freq_melhor = pares[melhor_par]

    merges.append(melhor_par)
    vocab_atual = aplicar_merge(vocab_atual, melhor_par)

    print(f"  Merge {passo + 1}: {melhor_par} (freq={freq_melhor}) -> "
          f"{''.join(melhor_par)!r}")

print(f"\nTotal de merges aprendidos: {len(merges)}")


# ========== 4. VOCABULÁRIO FINAL ==========
print("\n4. VOCABULÁRIO FINAL (token -> ID)")
print("-" * 70)

# Coleta todos os símbolos únicos que aparecem no vocabulário final
simbolos_finais = set()
for simbolos in vocab_atual.keys():
    simbolos_finais.update(simbolos)

# Tokens especiais primeiro (convenção comum)
tokens_especiais = ["<pad>", "<unk>", "<bos>", "<eos>"]
todos_tokens = tokens_especiais + sorted(simbolos_finais)

token_to_id = {tok: i for i, tok in enumerate(todos_tokens)}
id_to_token = {i: tok for tok, i in token_to_id.items()}

print(f"Tamanho do vocabulário final: {len(token_to_id)} tokens")
print(f"Vocabulário: {token_to_id}")


# ========== 5. FUNÇÃO DE ENCODE ==========
print("\n5. ENCODE (texto -> IDs)")
print("-" * 70)

def aplicar_merges_em_palavra(palavra, merges):
    """Aplica a lista de merges aprendidos, na ordem, a uma única palavra."""
    simbolos = list(palavra_para_simbolos(palavra))
    for par in merges:
        a, b = par
        novo_simbolo = a + b
        nova_seq = []
        i = 0
        while i < len(simbolos):
            if i < len(simbolos) - 1 and simbolos[i] == a and simbolos[i + 1] == b:
                nova_seq.append(novo_simbolo)
                i += 2
            else:
                nova_seq.append(simbolos[i])
                i += 1
        simbolos = nova_seq
    return simbolos

def encode(texto, merges, token_to_id):
    ids = []
    tokens_legiveis = []
    for palavra in texto.split():
        simbolos = aplicar_merges_em_palavra(palavra, merges)
        for s in simbolos:
            tok_id = token_to_id.get(s, token_to_id["<unk>"])
            ids.append(tok_id)
            tokens_legiveis.append(s if s in token_to_id else "<unk>")
    return ids, tokens_legiveis

texto_teste = "o gato correu"
ids, tokens_legiveis = encode(texto_teste, merges, token_to_id)

print(f"Texto: {texto_teste!r}")
print(f"Tokens: {tokens_legiveis}")
print(f"IDs:    {ids}")

# Testando com uma palavra NUNCA vista no treino
texto_novo = "o gatorrado"
ids_novo, tokens_novo = encode(texto_novo, merges, token_to_id)
print(f"\nTexto com palavra nova: {texto_novo!r}")
print(f"Tokens: {tokens_novo}")
print(f"IDs:    {ids_novo}")
print("-> Repare: mesmo 'gatorrado' nunca tendo aparecido no treino,")
print("   o BPE consegue quebrá-lo em pedaços conhecidos (ou individuais).")


# ========== 6. FUNÇÃO DE DECODE ==========
print("\n6. DECODE (IDs -> texto)")
print("-" * 70)

def decode(ids, id_to_token):
    tokens = [id_to_token[i] for i in ids]
    texto = "".join(tokens)
    texto = texto.replace("_", " ").strip()
    return texto

texto_reconstruido = decode(ids, id_to_token)
print(f"IDs originais:        {ids}")
print(f"Texto reconstruído:   {texto_reconstruido!r}")
print(f"Texto original:       {texto_teste!r}")
print(f"Reconstrução correta: {texto_reconstruido == texto_teste}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_bpe.py
```

Você deve observar que, com apenas 5 frases curtas, o BPE aprende rapidamente que `gat`, `o_`, `a_` são unidades frequentes, e que uma palavra nunca vista (`gatorrado`) ainda consegue ser codificada — em pedaços parcialmente conhecidos, ao invés de virar um único `<unk>` que descarta toda a informação.

---

## Tokenizers em Produção

O BPE que implementamos acima é didático — opera sobre caracteres Unicode e é lento em corpora grandes. Bibliotecas de produção fazem variações mais sofisticadas:

- **tiktoken** (OpenAI): usado pelos modelos GPT. Implementa BPE a nível de *bytes* (não caracteres), o que garante que absolutamente qualquer sequência Unicode — mesmo emoji ou texto corrompido — sempre possa ser codificada, sem nunca precisar de `<unk>`. É escrito em Rust para ser extremamente rápido.
- **SentencePiece** (Google): usado por modelos como LLaMA e T5. Trata o texto como uma sequência bruta de caracteres (inclusive espaços), o que o torna independente de idioma — não assume que espaços separam palavras, o que é importante para idiomas como japonês e chinês. Suporta tanto BPE quanto um algoritmo alternativo chamado *Unigram Language Model*.

Ambos resolvem o mesmo problema central que resolvemos aqui manualmente: aprender, a partir de dados, um vocabulário de subpalavras que equilibra tamanho de vocabulário e comprimento de sequência. A diferença é engenharia — velocidade, robustez a Unicode, tratamento de espaços — não o princípio.

---

## Tokens Especiais

Além dos tokens "de conteúdo" aprendidos pelo BPE, todo tokenizer reserva alguns IDs para tokens de controle:

- **`<pad>`** (padding): usado para preencher sequências mais curtas até um comprimento fixo, quando processamos um batch com sequências de tamanhos diferentes. Não carrega significado — o modelo aprende a ignorá-lo (frequentemente combinado com uma máscara de atenção).
- **`<unk>`** (unknown): representa qualquer token fora do vocabulário. Com BPE a nível de bytes, isso é raro; com BPE a nível de caractere (como o nosso experimento, se um caractere nunca visto aparecer), ainda pode acontecer.
- **`<bos>`** (beginning of sequence): marca o início de uma sequência. Ajuda o modelo a saber "aqui começa um texto novo".
- **`<eos>`** (end of sequence): marca o fim de uma sequência. Crucial durante geração de texto (Capítulo 32) — é o sinal que diz ao modelo "pare de gerar aqui".

Esses tokens ocupam IDs fixos e reservados no vocabulário (no nosso experimento, colocamos eles nas posições 0-3), e o restante do pipeline (embedding, loss, geração) trata eles de forma especial em vários pontos — por exemplo, ignorando `<pad>` no cálculo da loss (Capítulo 26).

---

## Erros Comuns

### Erro 1: Esquecer o marcador de fim de palavra no BPE

```python
# ❌ Errado: sem marcador de fim de palavra, "o" no fim de "gato"
# pode se misturar com "o" no início de outra palavra durante os merges
simbolos = list("gato")  # ['g', 'a', 't', 'o']

# ✓ Certo: marcador explícito impede merges através de fronteiras de palavra
simbolos = list("gato") + ["_"]  # ['g', 'a', 't', 'o', '_']
```

### Erro 2: Aplicar merges fora de ordem no encode

```python
# ❌ Errado: aplicar os merges em ordem arbitrária (ex: conjunto/dict sem ordem)
# pode produzir uma tokenização diferente da que seria produzida no treino
for par in set(merges):  # sets não garantem ordem!
    ...

# ✓ Certo: os merges devem ser aplicados na MESMA ordem em que foram
# aprendidos durante o treino — essa ordem é parte do "modelo" do tokenizer
for par in merges:  # lista, preserva ordem de aprendizado
    ...
```

### Erro 3: Não tratar palavras fora do vocabulário

```python
# ❌ Errado: KeyError quando um símbolo não existe no vocabulário
tok_id = token_to_id[simbolo]  # explode se simbolo for desconhecido

# ✓ Certo: fallback para <unk>
tok_id = token_to_id.get(simbolo, token_to_id["<unk>"])
```

---

## Exercícios

### Exercício 24.1: Contar Pares Manualmente
Dado o mini-corpus `["ab_", "ab_", "ac_"]` (com "_" marcando fim de palavra e cada palavra já convertida em lista de caracteres), conte manualmente a frequência de cada par adjacente e diga qual seria mesclado primeiro.

### Exercício 24.2: Simular 2 Merges
Usando o mesmo corpus do Exercício 24.1, aplique 2 iterações de BPE manualmente (conte pares, mescle o mais frequente, repita) e escreva o estado final de cada palavra.

### Exercício 24.3: Implementar `contar_pares`
Sem olhar o experimento, implemente uma função `contar_pares(vocab_palavras)` que recebe um dicionário `{tupla_de_simbolos: frequencia}` e devolve a contagem de todos os pares adjacentes.

### Exercício 24.4: Vocab Size vs Número de Merges
Se o vocabulário inicial (só caracteres) tem 30 símbolos, e você faz 20 merges, qual é o tamanho final do vocabulário (sem contar tokens especiais)? Generalize a fórmula.

### Exercício 24.5: Tokens Especiais
Escreva um trecho de código que codifica a frase `"o gato correu"` cercando-a com `<bos>` e `<eos>`, usando o `encode()` do experimento.

---

## Gabarito

### Exercício 24.1: Contar Pares Manualmente
```python
from collections import defaultdict

corpus = [tuple("ab_"), tuple("ab_"), tuple("ac_")]

pares = defaultdict(int)
for palavra in corpus:
    for i in range(len(palavra) - 1):
        pares[(palavra[i], palavra[i+1])] += 1

print(dict(pares))
# {('a','b'): 2, ('b','_'): 2, ('a','c'): 1, ('c','_'): 1}
# ('a','b') e ('b','_') empatam com freq=2; qualquer critério de desempate
# consistente (ex: ordem alfabética) escolheria um deles primeiro.
```

### Exercício 24.2: Simular 2 Merges
```python
# Estado inicial: [a,b,_] [a,b,_] [a,c,_]
# Merge 1: mesclar ('a','b') (freq 2) -> "ab"
#   Estado: [ab,_] [ab,_] [a,c,_]
# Merge 2: contar pares de novo:
#   ('ab','_'): 2, ('a','c'): 1, ('c','_'): 1
#   Mesclar ('ab','_') -> "ab_"
#   Estado final: [ab_] [ab_] [a,c,_]
print("ab_ / ab_ / a c _")
```

### Exercício 24.3: Implementar `contar_pares`
```python
from collections import defaultdict

def contar_pares(vocab_palavras):
    pares = defaultdict(int)
    for simbolos, freq in vocab_palavras.items():
        for i in range(len(simbolos) - 1):
            par = (simbolos[i], simbolos[i + 1])
            pares[par] += freq
    return pares

# Teste
vocab = {("g","a","t","o","_"): 5, ("g","a","t","a","_"): 3}
print(dict(contar_pares(vocab)))
```

### Exercício 24.4: Vocab Size vs Número de Merges
```python
vocab_inicial = 30  # só caracteres
num_merges = 20

vocab_final = vocab_inicial + num_merges
print(f"Vocabulário final: {vocab_final}")  # 50

# Fórmula geral: vocab_size_final = vocab_size_caracteres + num_merges
# Cada merge cria exatamente 1 novo token, sem remover nenhum antigo
# (os símbolos originais continuam existindo para casos não mesclados).
```

### Exercício 24.5: Tokens Especiais
```python
def encode_com_especiais(texto, merges, token_to_id):
    ids, tokens = encode(texto, merges, token_to_id)
    ids_finais = [token_to_id["<bos>"]] + ids + [token_to_id["<eos>"]]
    tokens_finais = ["<bos>"] + tokens + ["<eos>"]
    return ids_finais, tokens_finais

ids, tokens = encode_com_especiais("o gato correu", merges, token_to_id)
print(tokens)
print(ids)
```

---

## Desafios Avançados (Opcionais)

### Fixação 24.1: BPE a Nível de Bytes
Modifique o experimento para operar sobre bytes UTF-8 (`texto.encode("utf-8")`) ao invés de caracteres Unicode. Qual vantagem isso traz para textos com emojis ou caracteres raros?

### Fixação 24.2: Vocabulário Ótimo
Treine o tokenizer com `NUM_MERGES` variando de 0 a 50 sobre um corpus maior (ex: um parágrafo de texto real). Plote o comprimento médio de sequência (em tokens) por frase versus `NUM_MERGES`. Em que ponto os ganhos começam a diminuir?

### Fixação 24.3: Comparando com um Tokenizer Real
Instale `tiktoken` (`pip install tiktoken`) e compare a tokenização de uma frase em português com o seu BPE caseiro. Quantos tokens cada um usa?

### Fixação 24.4: Desempate Determinístico
No `contar_pares`, quando dois pares empatam em frequência, `max()` do Python escolhe de forma implícita (baseado na ordem de inserção do dicionário). Modifique o código para ter um critério de desempate explícito e documentado (ex: ordem alfabética do par).

### Fixação 24.5: Merges Reversos (Decode Puro por Concatenação)
Prove, com um teste automatizado, que para qualquer frase do corpus de treino, `decode(encode(frase))` sempre reconstrói a frase original exatamente. Depois teste com uma frase totalmente fora do corpus — em que casos a reconstrução falha ou perde informação (ex: espaços múltiplos, pontuação colada)?

---

## Resumo

- **Tokenização**: converte texto bruto em uma sequência de inteiros (token IDs) — o primeiro passo antes de qualquer embedding.
- **Caractere vs palavra vs subpalavra**: trade-off entre tamanho de vocabulário, comprimento de sequência e capacidade de lidar com palavras novas.
- **BPE**: aprende um vocabulário de subpalavras iterativamente, mesclando o par de tokens adjacentes mais frequente até atingir o `vocab_size` desejado.
- **Ordem dos merges importa**: o encode deve aplicar os merges na mesma ordem em que foram aprendidos no treino.
- **Tokenizers de produção** (tiktoken, SentencePiece) resolvem o mesmo problema com engenharia mais robusta — bytes ao invés de caracteres, alta performance, independência de idioma.
- **Tokens especiais** (`<pad>`, `<unk>`, `<bos>`, `<eos>`) são reservados no vocabulário para controlar padding, palavras desconhecidas e limites de sequência.

Próximo capítulo: **Vocabulary e Context Windows** — como o tamanho do vocabulário e o comprimento máximo de sequência afetam o número de parâmetros e o custo computacional do modelo.

---

**Próximo**: [Capítulo 25: Vocabulary e Context Windows](25_vocabulary_context.md)
