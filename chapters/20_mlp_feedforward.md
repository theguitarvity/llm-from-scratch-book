# Capítulo 20: MLP e Feed-Forward Network

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o papel da sub-camada feed-forward (FFN) dentro do bloco Transformer
2. Explicar por que o FFN é aplicado posição-a-posição (a mesma rede para cada token)
3. Justificar a escolha de expandir a dimensão antes de contrair (d_model -> d_ff -> d_model)
4. Comparar GELU e ReLU e explicar por que GELU é preferida em Transformers modernos
5. Implementar um FeedForward completo em PyTorch e verificar seus shapes e contagem de parâmetros

---

## Por Que Isso Importa

Até aqui você construiu a atenção: um mecanismo que permite que cada token "converse" com todos os outros tokens da sequência, coletando informação relevante de qualquer lugar. Isso é poderoso, mas tem uma limitação importante: a atenção, na sua essência, é uma operação de **combinação linear**. O output de cada posição é uma média ponderada dos Values de outras posições. Média ponderada é uma operação relativamente "simples" — ela mistura informação, mas não transforma essa informação de maneiras muito complexas.

Depois que a atenção decide "quais tokens são relevantes e quanto", alguém ainda precisa processar essa informação coletada e extrair algo útil dela. É aqui que entra a **Feed-Forward Network** (FFN), também chamada de MLP (Multi-Layer Perceptron) dentro do bloco Transformer. Se a atenção é a fase de "comunicação entre tokens", o FFN é a fase de "processamento individual": cada token, depois de ter coletado contexto via atenção, passa por uma rede neural própria que processa aquele vetor especificamente, sem olhar para os vizinhos.

Uma analogia útil: imagine uma reunião de trabalho. A fase de atenção é a parte da reunião em que todos trocam informações — cada pessoa escuta os outros e absorve o que é relevante. A fase do FFN é o momento em que, depois da reunião, cada pessoa volta para sua mesa e processa individualmente o que ouviu, aplicando seu próprio raciocínio para decidir o que fazer com aquela informação. Sem essa segunda fase, o modelo ficaria preso a combinações lineares de embeddings originais — nunca conseguiria aprender funções verdadeiramente não-lineares e complexas sobre o que a atenção juntou.

Na prática, o FFN é responsável por boa parte da capacidade bruta do modelo: em um Transformer típico, ele contém cerca de dois terços dos parâmetros treináveis de cada bloco. Entender exatamente como ele funciona — e por que sua estrutura de "expandir e depois contrair" foi escolhida — é essencial para entender de onde vem a capacidade de um LLM.

---

## Estrutura da Feed-Forward Network

### A operação, camada por camada

O FFN dentro de um bloco Transformer é surpreendentemente simples: são apenas duas camadas lineares com uma não-linearidade no meio.

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

Onde:
- $x \in \mathbb{R}^{d_{model}}$ é o vetor de um único token (na prática processamos todos os tokens em paralelo, mas a operação é idêntica para cada um)
- $W_1 \in \mathbb{R}^{[d_{model}, d_{ff}]}$ expande a dimensão
- $\sigma$ é uma função de ativação não-linear (GELU, na maioria dos Transformers modernos)
- $W_2 \in \mathbb{R}^{[d_{ff}, d_{model}]}$ contrai de volta para a dimensão original

Em termos de shapes de tensor, para um batch completo:

```
Entrada:            X       [batch, seq_len, d_model]
Após primeira Linear: H1    [batch, seq_len, d_ff]
Após ativação:        H1'   [batch, seq_len, d_ff]      (shape não muda, só os valores)
Após segunda Linear:  Y     [batch, seq_len, d_model]
```

Note o padrão: a dimensão de entrada e saída são **idênticas** ($d_{model}$). Isso é fundamental porque o FFN precisa se encaixar dentro do bloco Transformer, que preserva shape ao longo de toda a pilha de camadas (você verá isso em detalhe no próximo capítulo). O que muda é a dimensão *interna*, $d_{ff}$, que é tipicamente muito maior que $d_{model}$.

### Por que d_ff = 4 × d_model?

Na arquitetura Transformer original ("Attention is All You Need") e na maioria dos LLMs desde então (GPT-2, GPT-3, muitos modelos Llama), a razão $d_{ff} / d_{model} = 4$ é uma convenção quase universal. Por exemplo:

| Modelo | d_model | d_ff | razão |
|---|---|---|---|
| GPT-2 small | 768 | 3072 | 4x |
| GPT-2 large | 1280 | 5120 | 4x |
| GPT-3 (todos os tamanhos) | vários | 4 × d_model | 4x |

Por que expandir antes de contrair, e por que especificamente 4x?

**Intuição da expansão**: pense na primeira camada linear ($W_1$) como projetando o vetor do token para um espaço com muito mais "eixos" disponíveis. Em um espaço de dimensão maior, existem mais direções possíveis para representar padrões — combinações de features que seriam impossíveis de separar linearmente em um espaço menor tornam-se separáveis em um espaço maior (essa é a mesma intuição por trás de kernels em SVMs: subir de dimensão facilita separar padrões complexos). A não-linearidade aplicada nesse espaço expandido consegue então capturar interações mais ricas entre as features do que seria possível diretamente em $d_{model}$ dimensões.

**Por que contrair de volta**: depois de processar no espaço expandido, precisamos "resumir" o resultado de volta para $d_{model}$ dimensões, porque é esse o shape que o resto do bloco Transformer espera (para poder somar com a conexão residual, por exemplo — veremos isso no Capítulo 21).

**Por que 4x especificamente**: não existe uma prova matemática rigorosa de que 4 é ótimo — é largamente uma escolha empírica que funcionou bem e virou convenção. É um bom compromisso entre capacidade adicional (mais parâmetros, mais poder expressivo) e custo computacional/memória (o FFN é a parte mais cara em FLOPs de cada bloco). Alguns modelos mais recentes usam razões ligeiramente diferentes (por exemplo, variantes com ativações GLU costumam usar algo próximo de 8/3 para compensar o custo extra da porta multiplicativa), mas 4x continua sendo o ponto de partida canônico.

### Contagem de parâmetros

Vale a pena fazer as contas para ver o impacto:

$$\text{parâmetros}(W_1) = d_{model} \times d_{ff}$$
$$\text{parâmetros}(W_2) = d_{ff} \times d_{model}$$
$$\text{parâmetros totais (sem bias)} = 2 \times d_{model} \times d_{ff} = 2 \times d_{model} \times 4 d_{model} = 8 \, d_{model}^2$$

Compare com a atenção multi-head, cujas quatro matrizes de projeção ($W_Q, W_K, W_V, W_O$) somam aproximadamente $4 \, d_{model}^2$ parâmetros. Ou seja, **o FFN tem o dobro de parâmetros da camada de atenção** dentro do mesmo bloco. Isso confirma a intuição de que o FFN carrega a maior parte da "capacidade bruta" de cada bloco Transformer.

### FFN é aplicado posição-a-posição

Um ponto crucial, muitas vezes mal compreendido: o FFN é aplicado **independentemente a cada posição da sequência**, usando exatamente os mesmos pesos $W_1, b_1, W_2, b_2$ para todos os tokens.

Isso significa que, ao contrário da atenção — que mistura informação entre posições diferentes — o FFN nunca olha para o token vizinho. Se você pegar o vetor do token na posição 5 e processá-lo pelo FFN isoladamente, ou processar a sequência inteira de uma vez, o resultado na posição 5 é idêntico. Matematicamente:

$$\text{FFN}(X)[i, :] = \text{FFN}(X[i, :])$$

para qualquer posição $i$. Essa propriedade é o motivo de o FFN às vezes ser chamado de "camada de processamento posição-a-posição" (*position-wise feed-forward*) na literatura original.

Do ponto de vista de implementação, isso é conveniente: podemos simplesmente aplicar `nn.Linear` a um tensor de shape `[batch, seq_len, d_model]`, e o PyTorch trata as duas primeiras dimensões (`batch` e `seq_len`) como dimensões "de lote" — a transformação linear é aplicada independentemente a cada vetor de tamanho `d_model` na última dimensão, exatamente como se fosse `[batch * seq_len, d_model]` e depois remodelado de volta.

### GELU vs. ReLU: por que a escolha da ativação importa

A função ReLU (Rectified Linear Unit) é definida como:

$$\text{ReLU}(x) = \max(0, x)$$

É simples, rápida, e funcionou muito bem em redes convolucionais e nos primeiros Transformers. Mas ela tem duas características problemáticas quando aplicada em larga escala em modelos de linguagem profundos:

1. **Não é suave**: em $x=0$ há uma "quina" — a derivada muda abruptamente de 0 para 1. Isso pode causar instabilidades sutis de otimização, especialmente em redes muito profundas com muitas camadas empilhadas.
2. **"Neurônios mortos" (dying ReLU)**: se um neurônio recebe entradas consistentemente negativas durante o treinamento, seu gradiente se torna permanentemente zero (porque a derivada de ReLU para $x<0$ é exatamente 0). Esse neurônio nunca mais é atualizado — ele "morre" e deixa de contribuir para a capacidade da rede.

A GELU (Gaussian Error Linear Unit), usada em GPT-2, BERT, GPT-3 e na maioria dos LLMs modernos, resolve ambos os problemas:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

onde $\Phi(x)$ é a função de distribuição cumulativa (CDF) da normal padrão. Na prática, costuma-se usar uma aproximação eficiente:

$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right]\right)$$

Intuitivamente, a GELU pondera $x$ pela probabilidade de uma variável aleatória gaussiana ser menor que $x$ — em vez de simplesmente "cortar" valores negativos como ReLU, ela os atenua suavemente. Para $x$ muito negativo, GELU se aproxima de 0 (como ReLU), mas para valores próximos de 0 a transição é suave, e curiosamente GELU permite pequenos valores negativos passarem (diferente de ReLU, que zera tudo abaixo de 0). Essa suavidade elimina a "quina" em $x=0$ e reduz drasticamente o problema de neurônios mortos, porque o gradiente nunca é exatamente zero para $x < 0$ (exceto no limite).

```
     ReLU                          GELU
      |                              |
      |    /                        |     _/
      |   /                         |    /
      |  /                          |   /
______|_/______            ________|__/________
      |                     _______/
   (quina abrupta)          (transição suave, permite
                              pequenos valores negativos)
```

Essa suavidade adicional se traduz empiricamente em treinamento mais estável e, em muitos benchmarks, em melhor desempenho final — motivo pelo qual praticamente todo LLM moderno relevante (GPT-2/3/4, BERT, T5, Llama em suas variantes com ativação GLU) usa alguma forma de GELU ou uma prima próxima (como SiLU/Swish).

---

## Experimento: Feed-Forward Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Feed-Forward Network (MLP) do Transformer")
print("=" * 70)

# ========== 1. CONFIGURAÇÃO ==========
print("\n1. CONFIGURAÇÃO")
print("-" * 70)

batch_size = 2
seq_len = 5
d_model = 16
d_ff = 4 * d_model  # convenção padrão

print(f"batch_size: {batch_size}")
print(f"seq_len: {seq_len}")
print(f"d_model: {d_model}")
print(f"d_ff: {d_ff} (= 4 x d_model)")

# ========== 2. DEFININDO O MÓDULO FEEDFORWARD ==========
print("\n2. DEFININDO O MÓDULO FeedForward")
print("-" * 70)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        h = self.linear1(x)        # [batch, seq_len, d_ff]
        h = self.activation(h)     # [batch, seq_len, d_ff] (shape preservado)
        out = self.linear2(h)      # [batch, seq_len, d_model]
        return out

ffn = FeedForward(d_model, d_ff)
print("Módulo FeedForward criado com sucesso")
print(ffn)

# ========== 3. CONTAGEM DE PARÂMETROS ==========
print("\n3. CONTAGEM DE PARÂMETROS")
print("-" * 70)

n_params_linear1 = sum(p.numel() for p in ffn.linear1.parameters())
n_params_linear2 = sum(p.numel() for p in ffn.linear2.parameters())
n_params_total = sum(p.numel() for p in ffn.parameters())

print(f"linear1 (d_model -> d_ff): {n_params_linear1} parâmetros")
print(f"  esperado (W + b): {d_model * d_ff} + {d_ff} = {d_model * d_ff + d_ff}")
print(f"linear2 (d_ff -> d_model): {n_params_linear2} parâmetros")
print(f"  esperado (W + b): {d_ff * d_model} + {d_model} = {d_ff * d_model + d_model}")
print(f"Total FFN: {n_params_total} parâmetros")

# ========== 4. FORWARD PASS ==========
print("\n4. FORWARD PASS")
print("-" * 70)

X = torch.randn(batch_size, seq_len, d_model)
print(f"X shape (entrada): {X.shape}")

output = ffn(X)
print(f"output shape (saída): {output.shape}")
print(f"Shape preservado (entrada == saída)? {X.shape == output.shape}")

# ========== 5. VERIFICANDO APLICAÇÃO POSIÇÃO-A-POSIÇÃO ==========
print("\n5. VERIFICANDO APLICAÇÃO POSIÇÃO-A-POSIÇÃO")
print("-" * 70)

# Processar a sequência inteira de uma vez
out_full = ffn(X)

# Processar cada posição isoladamente
out_isolated = torch.zeros_like(out_full)
for b in range(batch_size):
    for t in range(seq_len):
        token_vec = X[b, t, :].unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        out_isolated[b, t, :] = ffn(token_vec).squeeze()

diff = (out_full - out_isolated).abs().max().item()
print(f"Diferença máxima entre processamento em lote vs. isolado: {diff:.2e}")
print("(deve ser ~0, confirmando que o FFN é posição-a-posição)")

# ========== 6. COMPARANDO GELU E RELU ==========
print("\n6. COMPARANDO GELU E RELU")
print("-" * 70)

x_test = torch.linspace(-3, 3, 7)
gelu_out = F.gelu(x_test)
relu_out = F.relu(x_test)

print(f"{'x':>8} {'ReLU(x)':>10} {'GELU(x)':>10}")
for xi, r, g in zip(x_test.tolist(), relu_out.tolist(), gelu_out.tolist()):
    print(f"{xi:8.2f} {r:10.4f} {g:10.4f}")

print("\nObservação: para x negativo, ReLU zera tudo (gradiente morto).")
print("GELU permite pequenos valores negativos passarem, com transição suave.")

# ========== 7. EFEITO NA DISTRIBUIÇÃO DE ATIVAÇÕES ==========
print("\n7. EFEITO NA DISTRIBUIÇÃO DE ATIVAÇÕES INTERNAS (d_ff)")
print("-" * 70)

with torch.no_grad():
    h_pre_activation = ffn.linear1(X)
    h_post_activation = ffn.activation(h_pre_activation)

frac_zero_pre = (h_pre_activation < 0).float().mean().item()
frac_zero_post_gelu = (h_post_activation.abs() < 1e-3).float().mean().item()
frac_zero_relu = (F.relu(h_pre_activation).abs() < 1e-3).float().mean().item()

print(f"Fração de pré-ativações negativas: {frac_zero_pre:.2%}")
print(f"Fração de saídas ~0 após GELU: {frac_zero_post_gelu:.2%}")
print(f"Fração de saídas ~0 após ReLU (comparação): {frac_zero_relu:.2%}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Saída esperada (resumida): o experimento confirma que o shape de entrada e saída do FFN são idênticos (`[batch, seq_len, d_model]`), que a expansão interna passa por `d_ff = 4 * d_model`, que o processamento é matematicamente idêntico se feito em lote ou token a token, e que GELU produz uma quantidade menor de ativações exatamente zero em comparação com ReLU.

---

## Erros Comuns

### Erro 1: Esquecer a não-linearidade entre as duas camadas lineares

```python
# Errado: duas camadas lineares empilhadas sem ativação
# entre elas colapsam matematicamente em UMA única transformação linear!
class FeedForwardRuim(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.linear1(x))  # sem ativação!

# Certo: sempre inclua a ativação
class FeedForwardCerto(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))
```

Sem a ativação não-linear, $W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$ — duas transformações lineares compostas são apenas *outra* transformação linear. Você perderia toda a capacidade de expansão de $d_ff$: o modelo se comportaria como se tivesse uma única camada linear de $d_{model}$ para $d_{model}$, desperdiçando os parâmetros extras.

### Erro 2: Confundir a dimensão de expansão com d_model

```python
# Errado: usar d_model tanto na entrada quanto na "expansão"
ffn = FeedForward(d_model=768, d_ff=768)  # sem expansão real!

# Certo: seguir a convenção de 4x (ou o valor especificado pela arquitetura)
ffn = FeedForward(d_model=768, d_ff=768 * 4)  # 3072
```

Usar `d_ff == d_model` não é tecnicamente um erro de shape (o código roda), mas anula o propósito da expansão: o modelo perde a maior parte da capacidade não-linear extra que o FFN deveria fornecer, e os benchmarks de desempenho caem visivelmente na prática.

### Erro 3: Aplicar a ativação depois da segunda camada linear

```python
# Errado: ativação no lugar errado
def forward(self, x):
    h = self.linear1(x)
    out = self.linear2(h)
    return self.activation(out)  # ativação após contração, não deveria estar aqui

# Certo: ativação entre as duas camadas, no espaço expandido
def forward(self, x):
    h = self.activation(self.linear1(x))
    out = self.linear2(h)
    return out
```

A não-linearidade precisa acontecer no espaço de alta dimensão ($d_{ff}$), onde há mais "espaço" para capturar interações complexas. Aplicá-la depois da segunda camada linear (já de volta em $d_{model}$) muda completamente a semântica da operação e não corresponde à arquitetura padrão do Transformer — além de potencialmente distorcer a saída antes da conexão residual (Capítulo 18).

---

## Exercícios

### Exercício 20.1: Contagem de Parâmetros
Para `d_model = 512` e a convenção `d_ff = 4 * d_model`, calcule manualmente (sem rodar código) o número total de parâmetros do FFN, incluindo bias. Depois confirme com PyTorch.

### Exercício 20.2: Verificando Colapso Linear
Implemente um FFN sem ativação (apenas duas `nn.Linear` empilhadas) e mostre numericamente que ele é equivalente a uma única transformação linear (compare a saída com `Linear(d_model, d_model)` cujo peso é o produto das duas matrizes).

### Exercício 20.3: Comparando Ativações
Implemente três versões do FFN — uma com ReLU, uma com GELU e uma com Tanh. Passe o mesmo tensor de entrada pelas três e compare a fração de neurônios que ficam com saída exatamente zero em cada uma.

### Exercício 20.4: FFN com Razão Diferente
Implemente um FFN com `d_ff = 2 * d_model` e outro com `d_ff = 8 * d_model`. Compare a contagem de parâmetros total e o tempo de forward pass (usando `time.time()`) para uma entrada grande.

### Exercício 20.5: Aplicação Posição-a-Posição
Dado um tensor `X` de shape `[1, 4, 8]`, embaralhe a ordem dos tokens na dimensão `seq_len` (por exemplo, inverta a ordem). Mostre que a saída do FFN para cada token individual não muda — apenas a ordem das saídas muda junto com a ordem de entrada.

---

## Gabarito

### Exercício 20.1: Contagem de Parâmetros
```python
d_model = 512
d_ff = 4 * d_model  # 2048

# linear1: W [512, 2048] + b [2048]
params_linear1 = d_model * d_ff + d_ff

# linear2: W [2048, 512] + b [512]
params_linear2 = d_ff * d_model + d_model

total = params_linear1 + params_linear2
print(f"linear1: {params_linear1}")   # 512*2048 + 2048 = 1.050.624
print(f"linear2: {params_linear2}")   # 2048*512 + 512  = 1.049.088
print(f"total: {total}")              # 2.099.712

import torch.nn as nn
ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(),
    nn.Linear(d_ff, d_model)
)
n = sum(p.numel() for p in ffn.parameters())
print(f"PyTorch confirma: {n}")  # deve bater com 2.099.712
```

### Exercício 20.2: Verificando Colapso Linear
```python
import torch
import torch.nn as nn

torch.manual_seed(0)
d_model, d_ff = 4, 16

linear1 = nn.Linear(d_model, d_ff, bias=False)
linear2 = nn.Linear(d_ff, d_model, bias=False)

x = torch.randn(1, 3, d_model)

# FFN sem ativação
out_ffn = linear2(linear1(x))

# Equivalente: uma única transformação com W = W2 @ W1
W_combined = linear2.weight @ linear1.weight  # [d_model, d_model]
out_combined = x @ W_combined.T

diff = (out_ffn - out_combined).abs().max().item()
print(f"Diferença máxima: {diff:.2e}")  # ~0, confirma colapso linear
```

### Exercício 20.3: Comparando Ativações
```python
import torch
import torch.nn as nn

torch.manual_seed(1)
d_model, d_ff = 8, 32
x = torch.randn(2, 5, d_model)

for name, act in [("ReLU", nn.ReLU()), ("GELU", nn.GELU()), ("Tanh", nn.Tanh())]:
    linear1 = nn.Linear(d_model, d_ff)
    h = act(linear1(x))
    frac_zero = (h.abs() < 1e-3).float().mean().item()
    print(f"{name}: fração de saídas ~0 = {frac_zero:.2%}")
```

### Exercício 20.4: FFN com Razão Diferente
```python
import torch
import torch.nn as nn
import time

d_model = 512
for ratio in [2, 8]:
    d_ff = ratio * d_model
    ffn = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Linear(d_ff, d_model)
    )
    n_params = sum(p.numel() for p in ffn.parameters())
    x = torch.randn(8, 512, d_model)

    start = time.time()
    for _ in range(10):
        _ = ffn(x)
    elapsed = time.time() - start

    print(f"ratio={ratio}: {n_params} parâmetros, {elapsed:.4f}s para 10 forward passes")
```

### Exercício 20.5: Aplicação Posição-a-Posição
```python
import torch
import torch.nn as nn

torch.manual_seed(2)
d_model = 8
ffn = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, d_model))

X = torch.randn(1, 4, d_model)
X_invertido = X.flip(dims=[1])  # inverte a ordem dos tokens

out_original = ffn(X)
out_invertido = ffn(X_invertido)

# A saída invertida deve ser igual à saída original, também invertida
diff = (out_original.flip(dims=[1]) - out_invertido).abs().max().item()
print(f"Diferença máxima: {diff:.2e}")  # ~0, confirma processamento independente por posição
```

---

## Desafios Avançados (Opcionais)

### Fixação 20.1: Implementando GLU (Gated Linear Unit)
Modelos como Llama usam uma variante do FFN chamada SwiGLU, que introduz uma "porta" multiplicativa: `FFN(x) = W2 * (SiLU(W1 x) * W3 x)`. Implemente essa variante e compare sua contagem de parâmetros com o FFN padrão para o mesmo `d_model`.

### Fixação 20.2: Sparsity Induzida por ReLU
Meça, para uma rede treinada por algumas iterações de gradiente com ReLU, a fração de neurônios que ficam permanentemente em zero ("neurônios mortos") ao longo do treinamento. Compare com a mesma rede usando GELU.

### Fixação 20.3: Custo Computacional em FLOPs
Calcule analiticamente o número de FLOPs (multiplicação + soma) do FFN para uma sequência de comprimento `seq_len`, e compare com o FLOPs da camada de atenção multi-head para a mesma sequência. Em que ponto (em função de `seq_len`) a atenção passa a dominar o custo computacional?

### Fixação 20.4: Dropout no FFN
Adicione uma camada `nn.Dropout` depois da ativação e depois da segunda camada linear. Explique, em termos de regularização, por que o dropout é tipicamente aplicado nesses pontos específicos do FFN.

### Fixação 20.5: Visualizando Ativações Esparsas
Gere um heatmap (usando matplotlib) das ativações no espaço `d_ff` para um batch de tokens, comparando GELU e ReLU visualmente. Observe os padrões de esparsidade.

---

## Resumo

- **FFN (Feed-Forward Network)**: sub-camada do bloco Transformer que processa cada token individualmente, após a atenção ter "comunicado" informação entre tokens
- **Estrutura**: `Linear(d_model -> d_ff) -> ativação -> Linear(d_ff -> d_model)`, sempre preservando shape de entrada/saída
- **d_ff = 4 × d_model**: convenção padrão que equilibra capacidade extra e custo computacional
- **Posição-a-posição**: o mesmo FFN, com os mesmos pesos, é aplicado independentemente a cada token da sequência
- **GELU > ReLU**: GELU é suave, evita neurônios mortos, e é a escolha padrão em LLMs modernos
- **Parâmetros**: o FFN concentra aproximadamente o dobro dos parâmetros da camada de atenção em cada bloco

Próximo capítulo: **Bloco Transformer Completo** — vamos juntar atenção, FFN, conexões residuais e layer normalization em um único módulo reutilizável.

---

**Próximo**: [Capítulo 21: Bloco Transformer Completo](21_bloco_transformer.md)
