# Capítulo 14: Weighted Sum de V

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Explicar como attention weights (que somam 1) combinam os vetores Value em uma saída contextualizada
2. Interpretar o output de atenção como uma "média ponderada dinâmica" no espaço de embeddings
3. Rastrear os shapes exatos de cada etapa, da matriz de pesos até o output final
4. Comparar attention com mean pooling simples, entendendo por que pesos dinâmicos são mais expressivos que pesos fixos
5. Visualizar como pequenas mudanças nos pesos de atenção alteram o vetor de saída

---

## Por Que Isso Importa

Você já passou pelos Capítulos 11, 12 e 13: projetou Q, K, V; calculou scores via $QK^T$; escalou por $\sqrt{d_k}$ e normalizou com softmax. Agora você tem, para cada posição, uma distribuição de probabilidade sobre todas as posições da sequência — os attention weights. Mas uma distribuição de probabilidade sozinha não é uma representação de texto; ela é só uma lista de números que somam 1. O passo final, e talvez o mais subestimado, é usar esses pesos para efetivamente **combinar informação** — e é aqui que a "mágica" da atenção se torna tangível.

Pense em como você resume o que várias pessoas disseram em uma reunião. Você não dá o mesmo peso a cada frase dita: pondera mentalmente pela relevância de quem falou e do que foi dito, e forma uma síntese que reflete mais fortemente os pontos mais importantes. É exatamente isso que a weighted sum faz matematicamente: pega os "conteúdos" (Values) de cada token e os mistura, dando mais peso para tokens mais relevantes (segundo os attention weights calculados) e menos peso para os irrelevantes.

Isso importa na prática porque é este passo — não o cálculo de scores, nem o softmax — que efetivamente produz a "saída" da camada de atenção: um novo vetor para cada posição, que agora **carrega informação contextual** de toda a sequência, não apenas do próprio token. Se você já debugou uma implementação de atenção onde o output "parece errado" mesmo com scores e softmax corretos, o bug quase sempre está aqui: uma multiplicação de matriz na ordem errada, ou um `V` que não corresponde ao `K` usado para calcular os pesos.

---

## A Combinação: attention_weights @ V

### Definição

Dado o attention_weights $A \in \mathbb{R}^{[n,n]}$ (já normalizado por softmax, cada linha somando 1) e Value $V \in \mathbb{R}^{[n,d_v]}$, o output é:

$$\text{output} = A \cdot V$$

Com shapes:

$$A \in \mathbb{R}^{[n,n]}, \quad V \in \mathbb{R}^{[n, d_v]} \quad\Rightarrow\quad \text{output} \in \mathbb{R}^{[n, d_v]}$$

Com batch:

$$A \in \mathbb{R}^{[batch, n, n]}, \quad V \in \mathbb{R}^{[batch, n, d_v]} \quad\Rightarrow\quad \text{output} \in \mathbb{R}^{[batch, n, d_v]}$$

Note algo importante: **o output tem o mesmo shape de sequência que a entrada** ($n$ posições), mas cada posição da saída agora é uma combinação de várias posições da entrada. Isso é o que torna a atenção uma operação de "mistura de informação através da sequência".

### Lendo a fórmula linha a linha

Para uma posição específica $i$, o output é a média ponderada dos Values de todas as posições:

$$\text{output}[i] = \sum_{j=1}^{n} A[i,j] \cdot V[j]$$

Onde $A[i,j]$ é o peso de atenção que a posição $i$ dá à posição $j$, e $V[j] \in \mathbb{R}^{d_v}$ é o vetor de Value da posição $j$. Como $\sum_j A[i,j] = 1$ (propriedade do softmax), essa soma é literalmente uma **combinação convexa** dos vetores $V[j]$ — ou seja, o resultado sempre fica "dentro" do espaço convexo formado pelos Values, nunca extrapola para fora dele.

---

## Interpretação: Média Ponderada no Espaço de Embeddings

### Por que isso é uma "média ponderada dinâmica"

Uma média ponderada tradicional (como uma média aritmética simples, ou uma média ponderada por pesos fixos) usa os mesmos pesos para todo input. A weighted sum de atenção, por outro lado, **recalcula os pesos a cada forward pass**, com base no conteúdo específico daquela sequência. Os pesos $A[i,:]$ não são parâmetros fixos do modelo — são a **saída** de um cálculo (Q, K, scores, softmax) que depende inteiramente da entrada atual.

Isso significa que, para a mesma posição $i$ em duas sequências diferentes, os pesos de atenção (e portanto o output) podem ser completamente diferentes, porque o conteúdo ao redor mudou. É essa propriedade — pesos condicionados no conteúdo — que dá à atenção sua expressividade.

### Geometricamente: interpolação no espaço vetorial

Se você imaginar cada $V[j]$ como um ponto em um espaço $d_v$-dimensional, o output[i] é um ponto **dentro do fecho convexo** (convex hull) desses pontos — uma espécie de "centro de massa ponderado" que se desloca conforme os pesos mudam. Quando um peso $A[i,j]$ domina (próximo de 1), o output se aproxima de $V[j]$; quando os pesos são uniformes, o output se aproxima do centroide simples de todos os $V$.

---

## Comparação com Mean Pooling

### Mean pooling: pesos fixos e uniformes

Uma alternativa mais simples e muito usada antes da popularização da atenção é o **mean pooling**: simplesmente tirar a média aritmética de todos os vetores da sequência, sem levar em conta relevância nenhuma.

$$\text{output}_{mean}[i] = \frac{1}{n}\sum_{j=1}^{n} V[j] \qquad \text{(mesmo valor para TODO } i\text{!)}$$

Repare no detalhe crucial: no mean pooling, o output **não depende de $i$** — é o mesmo vetor para todas as posições, porque os pesos são sempre $1/n$, iguais para todo mundo, independente de conteúdo ou posição.

### Attention: pesos aprendidos e dependentes do contexto

$$\text{output}_{attn}[i] = \sum_{j=1}^{n} A[i,j] \cdot V[j] \qquad \text{(diferente para cada } i\text{, e depende do conteúdo)}$$

A tabela abaixo resume a diferença:

| Propriedade | Mean Pooling | Attention |
|---|---|---|
| Pesos | Fixos, $1/n$ para todos | Dinâmicos, calculados via Q·K |
| Depende da posição $i$? | Não — mesmo output para todo $i$ | Sim — cada posição tem sua própria combinação |
| Depende do conteúdo? | Não — ignora o que está em cada token | Sim — pondera por relevância semântica |
| Capacidade de "focar" | Nenhuma | Alta — pode concentrar quase todo peso em 1-2 tokens |
| Parâmetros aprendidos | Nenhum | W_Q, W_K, W_V |

### Por que isso importa: um exemplo concreto

Considere a frase "O gato, que estava com fome porque não comia há dois dias, finalmente encontrou comida". Para entender o que "finalmente encontrou" se refere, o modelo precisa conectar fortemente com "gato" (o sujeito) e possivelmente "fome" (a motivação), ignorando praticamente detalhes como "dois dias". Mean pooling misturaria tudo igualmente — "dois dias" teria o mesmo peso que "gato" — diluindo o sinal relevante em meio a informação irrelevante. Attention, por ser capaz de aprender pesos que concentram força em "gato" e "fome" e quase zero em "dois dias", produz uma representação muito mais informativa para prever a continuação do texto.

---

## Exemplo Numérico Manual

Vamos calcular a weighted sum passo a passo com números pequenos.

### Setup

Sequência de 3 tokens, $d_v = 2$:

```
V = [
    [1.0, 0.0],    # posição 0 ("O")
    [0.0, 1.0],    # posição 1 ("gato")
    [2.0, 2.0]     # posição 2 ("dormia")
]

attention_weights (já calculados via softmax, linha i = pesos da Query i) =
[
    [0.7, 0.2, 0.1],   # posição 0 presta MUITA atenção em si mesma
    [0.1, 0.8, 0.1],   # posição 1 presta MUITA atenção em si mesma
    [0.3, 0.3, 0.4]    # posição 2 distribui atenção mais uniformemente
]
```

### Passo 1: output[0]

```
output[0] = 0.7 * V[0] + 0.2 * V[1] + 0.1 * V[2]
          = 0.7 * [1.0, 0.0] + 0.2 * [0.0, 1.0] + 0.1 * [2.0, 2.0]
          = [0.7, 0.0] + [0.0, 0.2] + [0.2, 0.2]
          = [0.9, 0.4]
```

### Passo 2: output[1]

```
output[1] = 0.1 * V[0] + 0.8 * V[1] + 0.1 * V[2]
          = [0.1, 0.0] + [0.0, 0.8] + [0.2, 0.2]
          = [0.3, 1.0]
```

### Passo 3: output[2]

```
output[2] = 0.3 * V[0] + 0.3 * V[1] + 0.4 * V[2]
          = [0.3, 0.0] + [0.0, 0.3] + [0.8, 0.8]
          = [1.1, 1.1]
```

### Matriz final de output

```
output = [
    [0.9, 0.4],
    [0.3, 1.0],
    [1.1, 1.1]
]
```

Compare com o mean pooling simples (pesos $1/3$ para todos): $\text{output}_{mean} = \frac{1}{3}(V[0]+V[1]+V[2]) = \frac{1}{3}[3.0, 3.0] = [1.0, 1.0]$ — o **mesmo vetor** seria usado para as três posições, perdendo toda a informação de que a posição 0 se identifica mais consigo mesma, a posição 1 também, e a posição 2 distribui mais igualmente.

---

## Experimento: Weighted Sum e Sensibilidade aos Pesos

```python
import torch

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Weighted Sum de V")
print("=" * 70)

# ========== SETUP ==========
print("\n1. SETUP")
print("-" * 70)

n = 3
d_v = 2

V = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [2.0, 2.0]
])

attention_weights = torch.tensor([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.3, 0.3, 0.4]
])

print(f"V shape: {V.shape}")
print(f"attention_weights shape: {attention_weights.shape}")
print(f"Soma de cada linha de attention_weights: {attention_weights.sum(dim=1)}")

# ========== WEIGHTED SUM ==========
print("\n2. CALCULANDO OUTPUT = attention_weights @ V")
print("-" * 70)

output = attention_weights @ V
print(f"output shape: {output.shape}")
print(f"output =\n{output}")

# ========== COMPARANDO COM MEAN POOLING ==========
print("\n3. COMPARANDO COM MEAN POOLING")
print("-" * 70)

mean_pooling_weights = torch.ones(n, n) / n
output_mean = mean_pooling_weights @ V

print(f"mean_pooling_weights =\n{mean_pooling_weights}")
print(f"output com mean pooling (IGUAL para toda posição) =\n{output_mean}")
print(f"\noutput com attention (DIFERENTE por posição) =\n{output}")

# ========== SENSIBILIDADE: MUDANDO OS PESOS ==========
print("\n4. SENSIBILIDADE DO OUTPUT AOS PESOS DE ATENÇÃO")
print("-" * 70)

pesos_variantes = [
    torch.tensor([1.0, 0.0, 0.0]),   # 100% no token 0
    torch.tensor([0.0, 1.0, 0.0]),   # 100% no token 1
    torch.tensor([0.0, 0.0, 1.0]),   # 100% no token 2
    torch.tensor([0.33, 0.33, 0.34]),  # quase uniforme
]

for pesos in pesos_variantes:
    out = pesos @ V
    print(f"pesos={pesos.numpy()}  ->  output={out.numpy().round(3)}")

print("\nRepare: quando um peso domina (1.0), o output converge para o V daquele token.")
print("Quando os pesos são uniformes, o output converge para o centroide dos V.")

# ========== FORWARD COMPLETO: DE SCORES ATÉ OUTPUT ==========
print("\n5. PIPELINE COMPLETO (scores -> softmax -> weighted sum)")
print("-" * 70)

d = 4
X = torch.randn(n, d)
W_q = torch.randn(d, d) * 0.5
W_k = torch.randn(d, d) * 0.5
W_v = torch.randn(d, d) * 0.5

Q = X @ W_q
K = X @ W_k
V_full = X @ W_v

scores = Q @ K.T / (d ** 0.5)
attn = torch.softmax(scores, dim=-1)
output_full = attn @ V_full

print(f"X shape: {X.shape}")
print(f"Q, K, V shape: {Q.shape}, {K.shape}, {V_full.shape}")
print(f"scores shape: {scores.shape}")
print(f"attn (attention_weights) shape: {attn.shape}, soma por linha: {attn.sum(dim=-1)}")
print(f"output final shape: {output_full.shape}")
print(f"output final =\n{output_full}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Multiplicar na ordem errada (V @ attention_weights)

```python
# Errado — ordem trocada, shapes incompatíveis ou resultado semanticamente errado
output = V @ attention_weights  # [n, d_v] @ [n, n] -> ERRO de shape (a menos que d_v == n, por coincidência)

# Certo — attention_weights primeiro, V depois
output = attention_weights @ V  # [n, n] @ [n, d_v] = [n, d_v]
```

### Erro 2: Usar scores brutos em vez dos pesos normalizados

```python
# Errado — usar scores antes do softmax na weighted sum
output = scores @ V  # scores não somam 1, output não é uma combinação convexa

# Certo — sempre aplicar softmax antes de combinar com V
attn = torch.softmax(scores / (d_k ** 0.5), dim=-1)
output = attn @ V
```

### Erro 3: Confundir dimensão de V com dimensão de Q/K no output

```python
# Errado — assumir que o output tem a dimensão de Q/K (d_k)
# quando d_v é diferente de d_k
d_k, d_v = 32, 64
attn = torch.softmax(torch.randn(5, 5), dim=-1)  # [n, n]
V = torch.randn(5, d_v)  # [n, d_v]
output = attn @ V  # [5, 64] -- o output segue d_v, NÃO d_k!

# Certo: sempre lembrar que output.shape[-1] == V.shape[-1] (== d_v), não d_k
```

---

## Exercícios

### Exercício 14.1: Cálculo Manual
Dado `attention_weights = [[1.0, 0.0], [0.5, 0.5]]` e `V = [[3.0, 1.0], [1.0, 5.0]]`, calcule `output` manualmente e confira com PyTorch.

### Exercício 14.2: Mean Pooling como Caso Especial
Mostre que mean pooling é um caso especial de weighted sum, usando uma matriz de pesos apropriada. Implemente e confirme que os resultados batem.

### Exercício 14.3: Output em Função de um Peso Variando
Fixe `V = [[0.0, 0.0], [10.0, 10.0]]` (2 tokens). Varie o peso do primeiro token de 0.0 a 1.0 em passos de 0.1 (peso do segundo = 1 - peso do primeiro) e plote (ou imprima) como o output se move entre os dois pontos.

### Exercício 14.4: Verificando Combinação Convexa
Gere `V` aleatório de shape `[5, 3]` e `attention_weights` aleatório (após softmax) de shape `[1, 5]`. Verifique que cada coordenada do output está entre o mínimo e o máximo da coordenada correspondente em V.

### Exercício 14.5: Pipeline Completo em Batch
Implemente o pipeline completo (Q, K, V -> scores -> softmax -> weighted sum) para um batch de tamanho 4, seq_len 6, d_model 16, e confirme todos os shapes intermediários.

---

## Gabarito

### Exercício 14.1: Cálculo Manual
```python
import torch

# Manual:
# output[0] = 1.0*[3,1] + 0.0*[1,5] = [3.0, 1.0]
# output[1] = 0.5*[3,1] + 0.5*[1,5] = [2.0, 3.0]

attn = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
V = torch.tensor([[3.0, 1.0], [1.0, 5.0]])
output = attn @ V
print(output)  # tensor([[3., 1.], [2., 3.]])
```

### Exercício 14.2: Mean Pooling como Caso Especial
```python
import torch

n = 4
V = torch.randn(n, 3)

pesos_mean = torch.ones(n, n) / n
output_mean_via_attn = pesos_mean @ V
output_mean_direto = V.mean(dim=0, keepdim=True).expand(n, -1)

print(torch.allclose(output_mean_via_attn, output_mean_direto))  # True
```

### Exercício 14.3: Output em Função de um Peso Variando
```python
import torch

V = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

for w in torch.arange(0.0, 1.01, 0.1):
    pesos = torch.tensor([w, 1 - w])
    out = pesos @ V
    print(f"peso_token0={w:.1f}  ->  output={out.numpy().round(2)}")
# O output se move linearmente de [10,10] (w=0) até [0,0] (w=1)
```

### Exercício 14.4: Verificando Combinação Convexa
```python
import torch
torch.manual_seed(1)

V = torch.randn(5, 3)
pesos = torch.softmax(torch.randn(1, 5), dim=-1)
output = pesos @ V  # [1, 3]

for coord in range(3):
    minimo = V[:, coord].min().item()
    maximo = V[:, coord].max().item()
    valor = output[0, coord].item()
    dentro = minimo <= valor <= maximo
    print(f"coord {coord}: min={minimo:.3f}, max={maximo:.3f}, output={valor:.3f}, dentro_do_intervalo={dentro}")
```

### Exercício 14.5: Pipeline Completo em Batch
```python
import torch

batch, n, d = 4, 6, 16
X = torch.randn(batch, n, d)

W_q = torch.randn(d, d) * 0.1
W_k = torch.randn(d, d) * 0.1
W_v = torch.randn(d, d) * 0.1

Q = X @ W_q
K = X @ W_k
V = X @ W_v

scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
attn = torch.softmax(scores, dim=-1)
output = attn @ V

print(f"X: {X.shape}")            # [4, 6, 16]
print(f"Q, K, V: {Q.shape}")      # [4, 6, 16]
print(f"scores: {scores.shape}")  # [4, 6, 6]
print(f"attn: {attn.shape}")      # [4, 6, 6]
print(f"output: {output.shape}")  # [4, 6, 16]
```

---

## Desafios Avançados (Opcionais)

### Fixação 14.1: Centroide vs Atenção Concentrada
Compute a distância euclidiana entre o output de atenção e o centroide simples (mean pooling) para diferentes níveis de "concentração" dos pesos (use entropia como proxy). Mostre que quanto mais concentrada a atenção, mais distante o output fica do centroide.

### Fixação 14.2: d_v Diferente de d_k
Implemente um pipeline completo onde $d_k = 16$ (dimensão de Q e K) mas $d_v = 32$ (dimensão de V). Confirme que o output final tem shape `[..., 32]`, não `[..., 16]`.

### Fixação 14.3: Reconstrução Aproximada
Se você souber o `output` e a matriz `attention_weights`, é possível recuperar exatamente `V`? Investigue sob quais condições (por exemplo, `attention_weights` quadrada e invertível) isso seria possível, e por que na prática não é feito.

### Fixação 14.4: Visualizando a Trajetória do Output
Com 3 vetores V fixos formando um triângulo em 2D, varie os pesos de atenção sistematicamente (uma grade de combinações que somam 1) e plote todos os outputs resultantes. Confirme visualmente que eles preenchem o triângulo (fecho convexo).

### Fixação 14.5: Comparando com Max-Pooling
Implemente também "max pooling" (pegar o máximo elemento a elemento entre os V) como uma terceira alternativa de agregação, além de mean pooling e attention. Discuta por que max pooling não é diferenciável de forma suave (ou exige truques como soft-max) e por que attention preserva melhor gradientes.

---

## Resumo

- **Weighted sum é `attention_weights @ V`**: a etapa final que produz o output contextualizado
- **Combinação convexa**: como os pesos somam 1, o output sempre fica dentro do fecho convexo dos vetores Value
- **Output depende da posição**: diferente de mean pooling, cada posição $i$ recebe uma combinação diferente, baseada em seus próprios pesos de atenção
- **Pesos dinâmicos vs fixos**: attention aprende pesos condicionados no conteúdo; mean pooling usa pesos fixos $1/n$ para sempre
- **Shape final segue V, não Q/K**: `output.shape[-1]` é sempre `d_v`, mesmo que `d_k` seja diferente
- **Sensibilidade aos pesos**: pequenas mudanças nos pesos de atenção deslocam o output continuamente entre os vetores Value

Próximo capítulo: **Causal Masking** — como impedir que um token "veja o futuro" em modelos autoregressivos.

---

**Próximo**: [Capítulo 15: Causal Masking](15_causal_masking.md)
