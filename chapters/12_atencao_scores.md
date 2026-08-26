# Capítulo 12: Attention Scores

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Explicar por que o produto escalar (dot product) é usado como medida de similaridade entre Query e Key
2. Interpretar geometricamente o dot product em termos de ângulo e magnitude de vetores
3. Computar a forma matricial $QK^T$ e rastrear seus shapes corretamente
4. Analisar a complexidade $O(n^2)$ da matriz de scores e seu impacto em sequências longas
5. Visualizar e interpretar uma matriz de scores bruta, antes da normalização por softmax

---

## Por Que Isso Importa

No Capítulo 11 você aprendeu que Query e Key são duas "lentes" diferentes aplicadas ao mesmo embedding, cada uma capturando um aspecto diferente da informação do token. Mas como, exatamente, o modelo decide que a Query do token "estava" deve "combinar" fortemente com a Key do token "rato" e fracamente com a Key do token "sofá"? A resposta é um cálculo simples: o **produto escalar** entre os dois vetores.

Se você já trabalhou com sistemas de busca (search engines, recomendação, embeddings de texto tipo `sentence-transformers`), provavelmente já usou similaridade de cosseno ou dot product para encontrar "documentos parecidos com a query do usuário". O attention score é literalmente a mesma ideia, só que aplicada internamente, token contra token, dentro do próprio modelo — e recalculada em toda camada, para cada novo contexto.

Isso importa na prática porque a matriz de scores $QK^T$ é o coração computacional de qualquer Transformer, e também seu maior gargalo. Se você já tentou rodar um modelo tipo GPT com uma sequência de 8.000 ou 32.000 tokens e viu a memória da GPU estourar, a causa raiz normalmente está aqui: essa matriz cresce com o **quadrado** do comprimento da sequência. Entender por que isso acontece — e por que o dot product foi escolhido em vez de, digamos, uma distância euclidiana ou uma rede neural completa para medir similaridade — é essencial tanto para debugar performance quanto para entender por que técnicas como FlashAttention e atenção esparsa existem.

---

## O Dot Product Como Medida de Similaridade

### Definição

Dados dois vetores $q, k \in \mathbb{R}^d$, o produto escalar (dot product) é:

$$q \cdot k = \sum_{i=1}^{d} q_i k_i$$

Esse único número resume "o quanto $q$ e $k$ apontam na mesma direção, ponderado por suas magnitudes".

### Interpretação geométrica

O dot product se relaciona com o ângulo $\theta$ entre os vetores por:

$$q \cdot k = \|q\| \, \|k\| \cos(\theta)$$

Isso nos diz três coisas importantes:

1. **Se $q$ e $k$ apontam na mesma direção** ($\theta \approx 0°$, $\cos\theta \approx 1$), o dot product é grande e positivo — alta similaridade.
2. **Se $q$ e $k$ são ortogonais** ($\theta = 90°$, $\cos\theta = 0$), o dot product é zero — nenhuma relação direcional.
3. **Se $q$ e $k$ apontam em direções opostas** ($\theta \approx 180°$, $\cos\theta \approx -1$), o dot product é grande e negativo — "anti-similaridade".

Note que o dot product **não é** exatamente a similaridade de cosseno — ele também é proporcional às normas $\|q\|$ e $\|k\|$. Isso significa que um vetor Key com norma muito grande (por exemplo, porque a rede aprendeu a "gritar mais alto" para chamar atenção) pode dominar o score mesmo com um ângulo apenas moderadamente favorável. Esse é, inclusive, um dos motivos pelos quais o scaling do Capítulo 13 é necessário — a magnitude dos vetores afeta diretamente a escala dos scores.

### Por que não usar distância euclidiana?

Poderíamos, em princípio, medir "similaridade" como o negativo da distância euclidiana: $-\|q - k\|^2$. Expandindo:

$$-\|q-k\|^2 = -\|q\|^2 - \|k\|^2 + 2(q \cdot k)$$

Repare que o termo $2(q \cdot k)$ aparece dentro dessa expressão — a distância euclidiana já contém o dot product, só que somado a termos extras ($\|q\|^2$ e $\|k\|^2$) que dependem apenas de cada vetor isoladamente, não da relação entre eles. Usar o dot product puro é computacionalmente mais barato (evita calcular normas explicitamente a cada par) e, empiricamente, funciona tão bem quanto — por isso a arquitetura Transformer original optou por ele.

### Por que não uma rede neural para medir similaridade?

Uma alternativa mais expressiva seria treinar uma pequena MLP que recebe $[q; k]$ (concatenados) e produz um score de compatibilidade — essa abordagem, chamada de "additive attention", foi de fato usada antes do Transformer (Bahdanau et al., 2014). Mas ela é mais cara computacionalmente: para cada par $(i, j)$ seria necessário um forward pass completo pela MLP. O dot product, por outro lado, pode ser calculado para **todos os pares simultaneamente** através de uma única multiplicação de matrizes, que é extremamente otimizada em GPUs. Essa eficiência computacional é uma das razões centrais pelas quais o Transformer venceu arquiteturas anteriores.

---

## Forma Matricial: QK^T

### Do par ao lote completo

Calcular scores[i,j] par a par, com um loop, seria correto mas lento:

```python
# Lento — só para entender o conceito
for i in range(n):
    for j in range(n):
        scores[i, j] = torch.dot(Q[i], K[j])
```

Em vez disso, calculamos todos os pares de uma vez com multiplicação de matrizes:

$$\text{scores} = Q K^T$$

Com shapes:

$$Q \in \mathbb{R}^{[n, d]}, \quad K^T \in \mathbb{R}^{[d, n]} \quad\Rightarrow\quad \text{scores} \in \mathbb{R}^{[n, n]}$$

Cada entrada $\text{scores}[i,j]$ é exatamente $Q[i] \cdot K[j]$ — a linha $i$ de $Q$ multiplicada pela coluna $j$ de $K^T$ (que é a linha $j$ de $K$).

Com batch dimension, em PyTorch:

$$Q \in \mathbb{R}^{[batch, n, d]}, \quad K \in \mathbb{R}^{[batch, n, d]}$$

```python
scores = Q @ K.transpose(-2, -1)  # [batch, n, n]
```

Usamos `transpose(-2, -1)` (em vez de `.T`) porque `.T` inverteria **todas** as dimensões, incluindo o batch — o que quebraria o resultado. `transpose(-2, -1)` troca só as duas últimas dimensões, preservando o batch.

### Lendo a matriz de scores

A matriz $\text{scores} \in \mathbb{R}^{[n,n]}$ tem uma leitura específica:

- **Linha $i$**: todos os scores da Query da posição $i$ contra todas as Keys.
- **Coluna $j$**: o quanto a Key da posição $j$ "atraiu" cada uma das Queries.
- **Diagonal $\text{scores}[i,i]$**: o quanto o token presta atenção em si mesmo (Q[i] · K[i]).

Importante: **a matriz de scores geralmente não é simétrica** ($\text{scores}[i,j] \neq \text{scores}[j,i]$), porque $Q[i] \cdot K[j]$ usa a Query de $i$ com a Key de $j$, enquanto $\text{scores}[j,i]$ usa a Query de $j$ com a Key de $i$ — e como $W_Q \neq W_K$ (Capítulo 11), esses valores diferem em geral.

---

## Exemplo Numérico Manual

Vamos calcular scores manualmente com números pequenos, sem PyTorch, para fixar a mecânica.

### Setup

Sequência de 3 tokens, $d = 2$ (dimensão pequena para facilitar as contas):

```
Q = [
    [1.0, 0.0],   # posição 0
    [0.0, 1.0],   # posição 1
    [1.0, 1.0]    # posição 2
]

K = [
    [1.0, 1.0],   # posição 0
    [1.0, 0.0],   # posição 1
    [0.0, 1.0]    # posição 2
]
```

### Passo 1: scores[0,:] — Query da posição 0

```
scores[0,0] = Q[0] · K[0] = 1.0*1.0 + 0.0*1.0 = 1.0
scores[0,1] = Q[0] · K[1] = 1.0*1.0 + 0.0*0.0 = 1.0
scores[0,2] = Q[0] · K[2] = 1.0*0.0 + 0.0*1.0 = 0.0
```

### Passo 2: scores[1,:] — Query da posição 1

```
scores[1,0] = Q[1] · K[0] = 0.0*1.0 + 1.0*1.0 = 1.0
scores[1,1] = Q[1] · K[1] = 0.0*1.0 + 1.0*0.0 = 0.0
scores[1,2] = Q[1] · K[2] = 0.0*0.0 + 1.0*1.0 = 1.0
```

### Passo 3: scores[2,:] — Query da posição 2

```
scores[2,0] = Q[2] · K[0] = 1.0*1.0 + 1.0*1.0 = 2.0
scores[2,1] = Q[2] · K[1] = 1.0*1.0 + 1.0*0.0 = 1.0
scores[2,2] = Q[2] · K[2] = 1.0*0.0 + 1.0*1.0 = 1.0
```

### Matriz completa

```
scores = [
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [2.0, 1.0, 1.0]
]
```

Repare que a matriz **não é simétrica**: $\text{scores}[0,1] = 1.0$ mas $\text{scores}[1,0] = 1.0$ (coincidência nesse caso pequeno), enquanto $\text{scores}[0,2] = 0.0$ é bem diferente de $\text{scores}[2,0] = 2.0$. Isso confirma o que discutimos: a relação de atenção não é simétrica.

Note também que essa é apenas a matriz de scores **bruta** — ainda não passamos por scaling nem softmax (isso é assunto do Capítulo 13). Esses valores ainda não são probabilidades; são apenas medidas cruas de similaridade, que podem ser qualquer número real.

---

## Complexidade O(n²)

### De onde vem o quadrado

A matriz de scores tem shape $[n, n]$ — ou seja, $n^2$ entradas, cada uma exigindo $d$ multiplicações e somas para o dot product. O custo total de calcular $QK^T$ é:

$$O(n^2 \cdot d)$$

Para uma sequência de comprimento fixo $n$, dobrar $n$ **quadruplica** o custo de calcular scores (e a memória necessária para armazenar a matriz). Isso é fundamentalmente diferente de arquiteturas recorrentes (RNN/LSTM), cujo custo cresce linearmente com $n$ ($O(n \cdot d^2)$), mas que sofrem com o problema de propagação de informação ao longo de muitos passos, que motivou o Capítulo 10.

### Por que isso importa na prática

Considere um modelo com $d = 768$ (tamanho do GPT-2 pequeno) processando sequências de diferentes comprimentos:

| Comprimento (n) | Entradas em scores (n²) | Memória aproximada (float32, 1 cabeça) |
|---|---|---|
| 512 | 262.144 | ~1 MB |
| 2.048 | 4.194.304 | ~16 MB |
| 8.192 | 67.108.864 | ~256 MB |
| 32.768 | 1.073.741.824 | ~4 GB |

E isso é **por batch, por camada, por cabeça de atenção** — em um modelo real com múltiplas camadas e múltiplas cabeças (Capítulo 17), a memória total explode rapidamente. Essa é a razão prática pela qual modelos com contexto muito longo (dezenas ou centenas de milhares de tokens) exigem técnicas especiais — FlashAttention (que evita materializar a matriz $n \times n$ inteira na memória), atenção esparsa (que calcula scores só para um subconjunto de pares), ou atenção linear (que reformula o cálculo para evitar a explosão quadrática).

Para os propósitos deste livro, vamos trabalhar com sequências curtas ($n \le 32$ nos experimentos), onde o custo $O(n^2)$ é irrelevante — mas é importante internalizar que essa é uma limitação fundamental da arquitetura, não um detalhe de implementação.

---

## Experimento: Calculando e Visualizando Scores

```python
import torch

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Attention Scores")
print("=" * 70)

# ========== CONFIGURAÇÃO ==========
n = 5       # seq_len
d = 4       # dimensão de Q e K

print(f"\nConfiguracao:")
print(f"  n (seq_len): {n}")
print(f"  d (dim de Q, K): {d}")

# ========== GERANDO Q E K ==========
print("\n1. Q E K ALEATÓRIOS")
print("-" * 70)

Q = torch.randn(n, d)
K = torch.randn(n, d)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")

# ========== CALCULANDO SCORES VIA LOOP (LENTO, PARA COMPARAR) ==========
print("\n2. SCORES VIA LOOP EXPLÍCITO (referência)")
print("-" * 70)

scores_loop = torch.zeros(n, n)
for i in range(n):
    for j in range(n):
        scores_loop[i, j] = torch.dot(Q[i], K[j])

print(f"scores_loop shape: {scores_loop.shape}")
print(f"scores_loop =\n{scores_loop}")

# ========== CALCULANDO SCORES VIA MATMUL (RÁPIDO) ==========
print("\n3. SCORES VIA Q @ K.T (matricial)")
print("-" * 70)

scores_matmul = Q @ K.T
print(f"scores_matmul shape: {scores_matmul.shape}")
print(f"scores_matmul =\n{scores_matmul}")

# Verificar equivalência
print(f"\nLoop e matmul são iguais? {torch.allclose(scores_loop, scores_matmul, atol=1e-5)}")

# ========== SIMETRIA (OU FALTA DELA) ==========
print("\n4. VERIFICANDO (A)SIMETRIA")
print("-" * 70)

diff = scores_matmul - scores_matmul.T
print(f"scores - scores.T (deveria ser ~zero SE simétrico) =\n{diff}")
print(f"É simétrica? {torch.allclose(scores_matmul, scores_matmul.T, atol=1e-5)}")

# ========== COM BATCH E TRANSPOSE(-2,-1) ==========
print("\n5. COM DIMENSÃO DE BATCH")
print("-" * 70)

batch_size = 2
Q_batch = torch.randn(batch_size, n, d)
K_batch = torch.randn(batch_size, n, d)

scores_batch = Q_batch @ K_batch.transpose(-2, -1)
print(f"Q_batch shape: {Q_batch.shape}")
print(f"K_batch shape: {K_batch.shape}")
print(f"scores_batch shape: {scores_batch.shape}  (deve ser [batch, n, n])")

# ========== COMPLEXIDADE QUADRÁTICA ==========
print("\n6. CRESCIMENTO QUADRÁTICO DA MATRIZ DE SCORES")
print("-" * 70)

for seq_len in [8, 16, 32, 64, 128]:
    n_entradas = seq_len * seq_len
    memoria_mb = (n_entradas * 4) / (1024 ** 2)  # float32 = 4 bytes
    print(f"n={seq_len:>4}  ->  scores tem {n_entradas:>7} entradas  "
          f"(~{memoria_mb:.4f} MB, 1 cabeça, float32)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Usar `.T` em tensores com batch

```python
# Errado — .T inverte TODAS as dimensões, incluindo batch
Q = torch.randn(2, 5, 4)  # [batch, n, d]
K = torch.randn(2, 5, 4)
scores = Q @ K.T  # ERRO de shape ou resultado incorreto

# Certo — transpose(-2, -1) troca só as duas últimas dimensões
scores = Q @ K.transpose(-2, -1)  # [2, 5, 5]
```

### Erro 2: Achar que scores já são probabilidades

```python
# Errado — tratar scores brutos como pesos de atenção
scores = Q @ K.transpose(-2, -1)
output = scores @ V  # scores não somam 1! Isso não é atenção correta.

# Certo — scores precisam passar por softmax primeiro (Capítulo 13)
scores = Q @ K.transpose(-2, -1)
weights = torch.softmax(scores / (d ** 0.5), dim=-1)
output = weights @ V
```

### Erro 3: Confundir scores[i,j] com scores[j,i]

```python
# Errado — assumir que a matriz é simétrica e usar qualquer ordem
relevancia_de_i_para_j = scores[j, i]  # Isso é o oposto do que se quer!

# Certo — scores[i,j] é "o quanto a Query de i valoriza a Key de j"
relevancia_de_i_para_j = scores[i, j]
```

---

## Exercícios

### Exercício 12.1: Cálculo Manual
Dados $Q = [[2, 0], [0, 2]]$ e $K = [[1, 1], [1, -1]]$, calcule a matriz de scores $QK^T$ manualmente (na mão) e depois confirme com PyTorch.

### Exercício 12.2: Verificando Assimetria
Gere $Q$ e $K$ aleatórios de shape `[4, 6]` com `torch.randn`. Calcule `scores = Q @ K.T`. Verifique programaticamente que `scores` não é simétrica.

### Exercício 12.3: Interpretação Geométrica
Crie dois vetores unitários (norma 1) apontando na mesma direção e calcule o dot product. Depois crie dois vetores ortogonais e calcule novamente. Confirme que os valores batem com a fórmula $q \cdot k = \|q\|\|k\|\cos\theta$.

### Exercício 12.4: Custo de Memória
Escreva uma função que recebe `seq_len` e `num_heads` e retorna a memória estimada (em MB, float32) da matriz de scores completa (todas as cabeças).

### Exercício 12.5: Scores com Batch
Gere `Q` e `K` de shape `[3, 10, 8]` (batch=3). Calcule os scores com `transpose(-2,-1)` e confirme o shape resultante.

---

## Gabarito

### Exercício 12.1: Cálculo Manual
```python
import torch

# Manual:
# scores[0,0] = Q[0]·K[0] = 2*1 + 0*1 = 2
# scores[0,1] = Q[0]·K[1] = 2*1 + 0*(-1) = 2
# scores[1,0] = Q[1]·K[0] = 0*1 + 2*1 = 2
# scores[1,1] = Q[1]·K[1] = 0*1 + 2*(-1) = -2
# resultado esperado: [[2, 2], [2, -2]]

Q = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
K = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
scores = Q @ K.T
print(scores)  # tensor([[ 2.,  2.], [ 2., -2.]])
```

### Exercício 12.2: Verificando Assimetria
```python
import torch
torch.manual_seed(0)

Q = torch.randn(4, 6)
K = torch.randn(4, 6)
scores = Q @ K.T

is_symmetric = torch.allclose(scores, scores.T, atol=1e-6)
print(f"scores é simétrica? {is_symmetric}")  # False (na grande maioria dos casos aleatórios)
```

### Exercício 12.3: Interpretação Geométrica
```python
import torch

# Mesma direção
a = torch.tensor([1.0, 0.0])
b = torch.tensor([1.0, 0.0])
print(torch.dot(a, b))  # 1.0 -> cos(0) = 1, normas=1

# Ortogonais
c = torch.tensor([1.0, 0.0])
d = torch.tensor([0.0, 1.0])
print(torch.dot(c, d))  # 0.0 -> cos(90) = 0
```

### Exercício 12.4: Custo de Memória
```python
def memoria_scores_mb(seq_len, num_heads, bytes_por_elemento=4):
    n_entradas = seq_len * seq_len * num_heads
    return (n_entradas * bytes_por_elemento) / (1024 ** 2)

print(memoria_scores_mb(2048, 12))  # GPT-2 small: ~192 MB só de scores
```

### Exercício 12.5: Scores com Batch
```python
import torch

Q = torch.randn(3, 10, 8)
K = torch.randn(3, 10, 8)
scores = Q @ K.transpose(-2, -1)
print(scores.shape)  # torch.Size([3, 10, 10])
```

---

## Desafios Avançados (Opcionais)

### Fixação 12.1: Normas e Dominância
Crie um vetor Key com norma muito maior que as demais (por exemplo, multiplicado por 100). Observe como isso afeta desproporcionalmente os scores relacionados a ele, mesmo com ângulo apenas moderadamente favorável.

### Fixação 12.2: Scores como Matriz de Adjacência
Trate a matriz de scores (após um threshold, por exemplo `scores > 0`) como uma matriz de adjacência de um grafo direcionado. Visualize esse grafo com `networkx` para uma sequência pequena.

### Fixação 12.3: Comparando Custos — Loop vs Matmul
Use `time.time()` para medir o tempo de calcular scores via loop duplo (`torch.dot`) vs via `Q @ K.T`, para `n=200`. Quantas vezes mais rápido é o matmul?

### Fixação 12.4: Atenção Esparsa Simples
Implemente uma versão em que cada posição só calcula scores contra suas 3 posições vizinhas mais próximas (janela local), em vez de todas as $n$ posições. Compare a economia de memória.

### Fixação 12.5: Dot Product vs Cosine Similarity
Modifique o cálculo de scores para normalizar Q e K antes do dot product (ou seja, usar similaridade de cosseno pura, sem influência de magnitude). Compare os padrões de atenção resultantes com os do dot product bruto.

---

## Resumo

- **Dot product mede similaridade**: $q \cdot k = \|q\|\|k\|\cos\theta$ — combina alinhamento direcional e magnitude
- **QK^T calcula tudo de uma vez**: forma matricial evita loops explícitos e é altamente otimizada em GPU
- **Shapes**: $Q \in [n,d]$, $K \in [n,d]$ resultam em $\text{scores} \in [n,n]$
- **Assimetria é esperada**: $\text{scores}[i,j] \neq \text{scores}[j,i]$ em geral, porque $W_Q \neq W_K$
- **Complexidade O(n²)**: a matriz de scores cresce quadraticamente com o comprimento da sequência — o principal gargalo de memória em Transformers
- **Scores brutos não são probabilidades**: ainda precisam de scaling e softmax antes de serem usados como pesos

Próximo capítulo: **Scaling por sqrt(d_k) e Softmax** — por que dividir os scores antes do softmax é crucial para o treinamento funcionar.

---

**Próximo**: [Capítulo 13: Scaling por sqrt(d_k) e Softmax](13_scaling_softmax.md)
