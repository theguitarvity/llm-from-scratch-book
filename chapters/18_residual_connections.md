# Capítulo 18: Residual Connections

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o problema de vanishing gradients em redes profundas
2. Explicar por que `output = x + F(x)` treina melhor que `output = F(x)`
3. Derivar matematicamente por que residual connections preservam o fluxo de gradiente
4. Identificar onde residual connections aparecem em um bloco Transformer
5. Medir empiricamente a diferença de magnitude de gradiente com e sem conexões residuais

---

## Por Que Isso Importa

Suponha que você empilhe 20 blocos Transformer, cada um contendo Multi-Head Attention (Capítulo 17) e, em breve, uma MLP (Capítulo 20). Cada bloco transforma sua entrada através de várias multiplicações de matriz, não-linearidades e normalizações. Ingenuamente, você poderia pensar: "mais camadas, mais capacidade, melhor modelo." Na prática, se você simplesmente empilhar `output = Bloco20(Bloco19(...Bloco1(x)...))` sem nenhum truque adicional, o treinamento frequentemente **piora** conforme você adiciona mais camadas — não por overfitting, mas porque o gradiente que precisa viajar da loss lá no final até os parâmetros do Bloco1 tem que atravessar 20 transformações não-lineares em sequência. A cada camada, o gradiente é multiplicado por uma matriz Jacobiana; se essas matrizes tendem a ter autovalores menores que 1 (o que é comum), o produto de 20 delas fica exponencialmente pequeno. É o fenômeno de **vanishing gradients**.

O sintoma prático é sutil e frustrante de debugar: você treina um modelo de 6 camadas e ele converge bem. Você aumenta para 24 camadas esperando um modelo melhor, e a loss mal se move — ou pior, fica pior que o modelo raso. Se você inspecionar a norma dos gradientes por camada (o que faremos no experimento deste capítulo), você vai ver algo revelador: camadas próximas da saída têm gradientes de magnitude razoável, mas camadas próximas da entrada têm gradientes praticamente zero. Elas simplesmente não estão aprendendo, porque o sinal de erro nunca chega até elas com força suficiente.

A solução, introduzida por He et al. em "Deep Residual Learning" (ResNets, 2015) e adotada centralmente pela arquitetura Transformer, é enganosamente simples: em vez de forçar cada camada a aprender a transformação completa `H(x)`, você deixa a camada aprender apenas o **resíduo** — a diferença entre a saída desejada e a entrada — e soma isso de volta à entrada original:

$$\text{output} = x + F(x)$$

em vez de

$$\text{output} = F(x)$$

Essa mudança de uma linha resolve o problema de vanishing gradients de forma quase mágica, porque cria um "caminho direto" (identity path) por onde o gradiente pode fluir sem ser atenuado por nenhuma multiplicação de matriz. É por isso que hoje é possível treinar Transformers com centenas de camadas — algo impensável sem residual connections.

---

## A Matemática do Gradient Highway

### Definindo o Bloco Residual

Um bloco residual tem a forma:

$$y = x + F(x)$$

onde $F$ é uma sub-rede qualquer (pode ser atenção, MLP, ou qualquer outra transformação) e $x$ é a entrada do bloco, com a mesma dimensão da saída de $F(x)$ — essa é uma restrição importante: para somar $x$ e $F(x)$ elemento a elemento, ambos precisam ter exatamente o mesmo shape.

$$x \in \mathbb{R}^{[\text{batch}, \text{seq\_len}, d_{model}]}, \quad F(x) \in \mathbb{R}^{[\text{batch}, \text{seq\_len}, d_{model}]}$$

### Derivando o Gradiente

Considere a derivada de $y$ em relação a $x$ (usando a regra da cadeia, tratando $F$ como uma função qualquer):

$$\frac{\partial y}{\partial x} = \frac{\partial x}{\partial x} + \frac{\partial F(x)}{\partial x} = I + \frac{\partial F(x)}{\partial x}$$

Aqui está o ponto crucial: o termo $I$ (a matriz identidade, representando a derivada de $x$ em relação a si mesmo) **está sempre lá, independente do que $F$ faz**. Mesmo que $\frac{\partial F(x)}{\partial x}$ seja próximo de zero (gradiente de $F$ desaparecendo), o gradiente total ainda contém o termo identidade, garantindo que o gradiente não desapareça completamente.

Agora empilhe $L$ blocos residuais: $y_L = y_{L-1} + F_L(y_{L-1})$, $y_{L-1} = y_{L-2} + F_{L-1}(y_{L-2})$, e assim por diante, até $y_1 = x_0 + F_1(x_0)$.

O gradiente da loss em relação à entrada $x_0$, pela regra da cadeia através de todas as camadas, é:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial y_L} \cdot \prod_{l=1}^{L} \left( I + \frac{\partial F_l}{\partial y_{l-1}} \right)$$

Expandindo o produto de somas, você obtém uma soma de muitos termos — um deles é exatamente $\frac{\partial \mathcal{L}}{\partial y_L} \cdot I \cdot I \cdots I = \frac{\partial \mathcal{L}}{\partial y_L}$, o gradiente da loss **propagado diretamente e sem atenuação** até a primeira camada, sem passar por nenhuma das transformações $F_l$. Esse é o "gradient highway" (autoestrada do gradiente): não importa quão "ruins" sejam as matrizes Jacobianas de cada $F_l$, sempre existe um caminho aditivo puro do início ao fim da rede por onde o gradiente flui sem ser multiplicado por nada menor que 1.

Compare isso com uma rede sem residuals, onde:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial y_L} \cdot \prod_{l=1}^{L} \frac{\partial F_l}{\partial y_{l-1}}$$

Aqui, cada termo do produto pode ser menor que 1 (ou maior, causando explosão), e o produto de $L$ termos pequenos tende a zero exponencialmente rápido conforme $L$ cresce. Não existe caminho alternativo — o gradiente é forçado a passar por todas as transformações não-lineares em sequência.

### Analogia: O Atalho que Preserva Informação

Pense em um prédio de 20 andares onde cada andar processa e possivelmente distorce uma mensagem antes de passá-la adiante — como o telefone sem fio. Depois de 20 andares, a mensagem original pode estar irreconhecível. Uma conexão residual é como instalar um elevador expresso paralelo às escadas: a mensagem original sempre chega ao topo intacta pelo elevador, e cada andar apenas *adiciona* uma nota ou correção à mensagem que já está lá — não a substitui inteiramente. Se um andar específico não tem nada útil a adicionar (aprende a produzir $F(x) \approx 0$), a informação simplesmente passa reto por ele, sem dano. Isso também explica por que redes residuais são mais fáceis de otimizar: a rede tem a opção de "não fazer nada" em qualquer camada (aprender $F(x) = 0$), que é um ponto de partida muito mais fácil de refinar do que ter que aprender a função identidade explicitamente com camadas não-lineares arbitrárias.

---

## Onde Residuals Aparecem no Bloco Transformer

Em um bloco Transformer padrão, há **duas** conexões residuais, uma ao redor de cada sub-camada principal:

```
x_entrada:                        [batch, seq_len, d_model]

# Sub-camada 1: Multi-Head Attention
attn_out = MultiHeadAttention(LayerNorm(x_entrada))
x_meio = x_entrada + attn_out     # <- primeira conexão residual

# Sub-camada 2: MLP / Feed-Forward
mlp_out = MLP(LayerNorm(x_meio))
x_saida = x_meio + mlp_out        # <- segunda conexão residual
```

(A posição exata do `LayerNorm` — antes ou depois da soma residual — é o tema do Capítulo 19, "Pre-LN vs Post-LN". Por ora, o que importa é que existe uma soma `x + F(x)` ao redor de cada sub-camada.)

Note que **ambas** as sub-camadas (atenção e MLP) recebem sua própria conexão residual independente. Isso significa que, em uma rede com $N$ blocos Transformer, existem $2N$ pontos onde o gradiente tem uma rota de escape direta — um "highway" que atravessa toda a profundidade da rede, permitindo empilhar dezenas ou centenas de blocos sem que o sinal de treinamento se perca.

Também vale notar a restrição de shape que isso impõe: como `x_entrada` e o output de cada sub-camada precisam ter o mesmo shape para serem somados, **toda sub-camada em um Transformer preserva `d_model`** na entrada e na saída (mesmo a MLP, que internamente expande para `4 * d_model` e depois contrai de volta — veremos isso no Capítulo 20). Essa restrição de "preservar dimensão" é uma consequência direta de precisarmos somar residuals.

---

## Experimento: Gradientes Com e Sem Residual em Rede Profunda

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Norma de Gradientes com e sem Residual Connections")
print("=" * 70)

torch.manual_seed(42)

# ========== CONFIGURAÇÃO ==========
d_model = 32
num_layers = 20
batch_size = 4

print(f"\nConfiguração:")
print(f"  d_model: {d_model}")
print(f"  num_layers: {num_layers}")
print(f"  batch_size: {batch_size}")


class CamadaSimples(nn.Module):
    """Uma 'sub-rede' F(x) simples: Linear -> Tanh -> Linear."""
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.ativacao = nn.Tanh()

    def forward(self, x):
        return self.linear2(self.ativacao(self.linear1(x)))


class RedeProfundaSemResidual(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.camadas = nn.ModuleList([CamadaSimples(d_model) for _ in range(num_layers)])

    def forward(self, x):
        for camada in self.camadas:
            x = camada(x)   # SEM soma residual: substitui x completamente
        return x


class RedeProfundaComResidual(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.camadas = nn.ModuleList([CamadaSimples(d_model) for _ in range(num_layers)])

    def forward(self, x):
        for camada in self.camadas:
            x = x + camada(x)   # COM soma residual: x + F(x)
        return x


# ========== ENTRADA ==========
print("\n1. ENTRADA")
print("-" * 70)

X = torch.randn(batch_size, d_model, requires_grad=True)
print(f"X shape: {X.shape}")

# ========== REDE SEM RESIDUAL ==========
print("\n2. REDE SEM RESIDUAL — FORWARD E BACKWARD")
print("-" * 70)

torch.manual_seed(0)
rede_sem = RedeProfundaSemResidual(d_model, num_layers)

X_sem = X.clone().detach().requires_grad_(True)
out_sem = rede_sem(X_sem)
loss_sem = out_sem.pow(2).mean()
loss_sem.backward()

print(f"loss: {loss_sem.item():.6f}")
print(f"norma do gradiente na ENTRADA (X_sem.grad): {X_sem.grad.norm().item():.8f}")

print("\nNorma do gradiente por camada (primeira Linear de cada CamadaSimples):")
normas_sem = []
for i, camada in enumerate(rede_sem.camadas):
    norma = camada.linear1.weight.grad.norm().item()
    normas_sem.append(norma)
    if i % 4 == 0 or i == num_layers - 1:
        print(f"  camada {i:2d}: {norma:.8f}")

# ========== REDE COM RESIDUAL ==========
print("\n3. REDE COM RESIDUAL — FORWARD E BACKWARD")
print("-" * 70)

torch.manual_seed(0)
rede_com = RedeProfundaComResidual(d_model, num_layers)

X_com = X.clone().detach().requires_grad_(True)
out_com = rede_com(X_com)
loss_com = out_com.pow(2).mean()
loss_com.backward()

print(f"loss: {loss_com.item():.6f}")
print(f"norma do gradiente na ENTRADA (X_com.grad): {X_com.grad.norm().item():.8f}")

print("\nNorma do gradiente por camada (primeira Linear de cada CamadaSimples):")
normas_com = []
for i, camada in enumerate(rede_com.camadas):
    norma = camada.linear1.weight.grad.norm().item()
    normas_com.append(norma)
    if i % 4 == 0 or i == num_layers - 1:
        print(f"  camada {i:2d}: {norma:.8f}")

# ========== COMPARAÇÃO ==========
print("\n4. COMPARAÇÃO DIRETA")
print("-" * 70)

print(f"Gradiente na entrada, SEM residual: {X_sem.grad.norm().item():.8f}")
print(f"Gradiente na entrada, COM residual: {X_com.grad.norm().item():.8f}")
razao = X_com.grad.norm().item() / max(X_sem.grad.norm().item(), 1e-12)
print(f"Razão (com / sem): {razao:.2f}x")

print("\nNorma média de gradiente por camada:")
print(f"  SEM residual: primeira camada = {normas_sem[0]:.8f}, última camada = {normas_sem[-1]:.8f}")
print(f"  COM residual: primeira camada = {normas_com[0]:.8f}, última camada = {normas_com[-1]:.8f}")

print("\nInterpretação:")
print("Sem residual, o gradiente na camada mais próxima da entrada é muito")
print("menor que na camada de saída (vanishing gradients). Com residual, o")
print("gradiente permanece em magnitude comparável em todas as camadas,")
print("porque o termo identidade (I) do gradiente não é atenuado.")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Experimento 2: Efeito de "Desligar" uma Camada (Identity Mapping)

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Uma Camada Residual 'Desligada' Não Quebra a Rede")
print("=" * 70)

torch.manual_seed(42)

d_model = 8
seq_len = 4
batch_size = 1

class BlocoResidual(nn.Module):
    def __init__(self, d_model, zerar_pesos=False):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        if zerar_pesos:
            # Simula uma camada que "não aprendeu nada útil" (F(x) ~ 0)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(x)   # x + F(x)

print("\n1. BLOCO NORMAL (pesos aleatórios)")
print("-" * 70)

X = torch.randn(batch_size, seq_len, d_model)
bloco_normal = BlocoResidual(d_model, zerar_pesos=False)
saida_normal = bloco_normal(X)
print(f"Entrada X (primeira linha): {X[0,0]}")
print(f"Saída (primeira linha):     {saida_normal[0,0]}")
print("A saída é diferente da entrada, como esperado.")

print("\n2. BLOCO 'DESLIGADO' (pesos zerados, F(x) = 0)")
print("-" * 70)

bloco_desligado = BlocoResidual(d_model, zerar_pesos=True)
saida_desligada = bloco_desligado(X)
print(f"Entrada X (primeira linha):  {X[0,0]}")
print(f"Saída (primeira linha):      {saida_desligada[0,0]}")
print(f"São idênticas? {torch.allclose(X, saida_desligada)}")

print("\nInterpretação:")
print("Quando F(x) aprende a produzir zero, o bloco residual vira uma função")
print("IDENTIDADE perfeita -- a informação passa direto, sem distorção.")
print("Isso significa que a rede pode 'optar por não usar' uma camada")
print("sem quebrar o fluxo de informação, algo impossível em output=F(x) puro")
print("(nesse caso, F(x)=0 destruiria toda a informação, não a preservaria).")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Somar tensores com shapes incompatíveis

```python
# ❌ Errado — F(x) muda a dimensão, impossibilitando a soma residual
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model * 4)  # expande sem contrair de volta

    def forward(self, x):
        return x + self.linear(x)  # RuntimeError: shapes incompatíveis

# ✓ Certo — a sub-rede sempre retorna à dimensão original antes da soma
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)  # contrai de volta

    def forward(self, x):
        return x + self.linear2(torch.relu(self.linear1(x)))
```

### Erro 2: Aplicar a soma residual antes da sub-camada, não depois

```python
# ❌ Errado — inverte a ordem: soma primeiro, aplica F depois
# (isso é uma arquitetura diferente e perde a propriedade de gradient highway)
def forward(self, x):
    x_somado = x + x
    return self.sublayer(x_somado)

# ✓ Certo — F(x) é computado a partir de x, depois somado de volta a x
def forward(self, x):
    return x + self.sublayer(x)
```

### Erro 3: Reescalar acidentalmente o caminho residual

```python
# ❌ Errado — multiplicar o caminho identidade por um fator (mesmo pequeno)
# quebra a propriedade de "gradiente 1 garantido", reintroduzindo atenuação
def forward(self, x):
    return 0.5 * x + self.sublayer(x)  # o termo identidade agora vale 0.5, não 1

# ✓ Certo — o caminho identidade deve ser preservado sem escala
# (se quiser controlar a magnitude da contribuição de F, escale F, não x)
def forward(self, x):
    return x + self.alpha * self.sublayer(x)  # aqui, x permanece intacto
```

### Erro 4: Esquecer que d_model precisa ser preservado em toda sub-camada

```python
# ❌ Errado — bloco de atenção que muda d_model no meio do caminho
# (funcionaria dentro do MultiHeadAttention, mas quebraria a soma residual
# se a saída final não voltasse para d_model)
attn_out = attention(x)  # se attn_out tiver shape diferente de x, a soma falha

# ✓ Certo — sempre garanta que W_O projeta de volta para d_model
# (ver Capítulo 17: concat de cabeças seguido de W_O: [d_model, d_model])
attn_out = attention(x)
assert attn_out.shape == x.shape
x = x + attn_out
```

---

## Exercícios

### Exercício 18.1: Derivar o Gradiente
Para `y = x + F(x)` com `F(x) = W @ x` (uma transformação linear simples), escreva `dy/dx` explicitamente em termos de `W` e da matriz identidade `I`.

### Exercício 18.2: Empilhando Blocos
Se você empilha 3 blocos residuais `y1 = x0 + F1(x0)`, `y2 = y1 + F2(y1)`, `y3 = y2 + F3(y2)`, expanda `y3` em termos de `x0` e mostre que o termo `x0` (sem nenhuma transformação `F`) aparece explicitamente na expressão final.

### Exercício 18.3: Implementar um Bloco Residual Genérico
Escreva uma classe `ResidualWrapper(nn.Module)` que recebe qualquer submódulo `sublayer` no construtor e aplica `x + sublayer(x)` no forward, verificando os shapes com um `assert`.

### Exercício 18.4: Medir Vanishing Gradients
Usando o código do Experimento 1, mude `num_layers` para 5, 20 e 50. Para cada valor, registre a norma do gradiente na primeira camada da rede sem residual. O que acontece conforme `num_layers` cresce?

### Exercício 18.5: Residual com Coeficiente
Implemente uma variante `y = x + alpha * F(x)`, onde `alpha` é um escalar aprendível inicializado em um valor pequeno (ex: 0.1). Explique por que essa técnica (usada em algumas arquiteturas modernas, como "LayerScale") pode ajudar a estabilizar o início do treinamento.

---

## Gabarito

### Exercício 18.1: Derivar o Gradiente
```
y = x + F(x), onde F(x) = W @ x
y = x + W @ x = (I + W) @ x

dy/dx = I + W
```
O gradiente é a soma da matriz identidade com `W`. Mesmo que `W` seja uma matriz "ruim" (por exemplo, com autovalores próximos de zero, o que atenuaria o sinal), o termo `I` garante que o gradiente nunca colapsa completamente para zero.

### Exercício 18.2: Empilhando Blocos
```
y1 = x0 + F1(x0)
y2 = y1 + F2(y1) = x0 + F1(x0) + F2(x0 + F1(x0))
y3 = y2 + F3(y2) = x0 + F1(x0) + F2(y1) + F3(y2)
```
Reorganizando:
```
y3 = x0 + [F1(x0) + F2(y1) + F3(y2)]
```
O termo `x0` aparece isolado, somado a todas as contribuições das sub-redes `F1, F2, F3`. Isso mostra explicitamente que a informação original de `x0` está sempre presente na saída final, não importa quantos blocos você empilhe — ela nunca é "substituída", apenas "complementada".

### Exercício 18.3: Implementar um Bloco Residual Genérico
```python
import torch.nn as nn

class ResidualWrapper(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        out = self.sublayer(x)
        assert out.shape == x.shape, (
            f"Shape incompatível para soma residual: "
            f"entrada {x.shape}, saída da sub-camada {out.shape}"
        )
        return x + out

# Exemplo de uso
import torch
d_model = 16
sub = nn.Linear(d_model, d_model)
bloco = ResidualWrapper(sub)
x = torch.randn(2, 5, d_model)
y = bloco(x)
print(y.shape)  # [2, 5, 16]
```

### Exercício 18.4: Medir Vanishing Gradients
```python
import torch
import torch.nn as nn

class CamadaSimples(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.ativacao = nn.Tanh()

    def forward(self, x):
        return self.linear2(self.ativacao(self.linear1(x)))

d_model = 32
for num_layers in [5, 20, 50]:
    torch.manual_seed(0)
    camadas = nn.ModuleList([CamadaSimples(d_model) for _ in range(num_layers)])
    x = torch.randn(4, d_model, requires_grad=True)
    h = x
    for c in camadas:
        h = c(h)  # SEM residual
    loss = h.pow(2).mean()
    loss.backward()
    print(f"num_layers={num_layers:3d}  norma do grad na entrada: {x.grad.norm().item():.10f}")

# Resultado esperado: a norma do gradiente diminui drasticamente
# (frequentemente por várias ordens de magnitude) conforme num_layers cresce,
# ilustrando o vanishing gradient problem sem conexões residuais.
```

### Exercício 18.5: Residual com Coeficiente
```python
import torch
import torch.nn as nn

class ResidualComAlpha(nn.Module):
    def __init__(self, sublayer, alpha_inicial=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.alpha = nn.Parameter(torch.tensor(alpha_inicial))

    def forward(self, x):
        return x + self.alpha * self.sublayer(x)

d_model = 16
bloco = ResidualComAlpha(nn.Linear(d_model, d_model))
x = torch.randn(2, 5, d_model)
y = bloco(x)
print(y.shape)
print(f"alpha inicial: {bloco.alpha.item()}")
```
Inicializar `alpha` pequeno faz com que, no início do treinamento, cada bloco contribua muito pouco além da identidade — a rede começa "quase transparente" (próxima de uma função identidade empilhada), o que é uma inicialização muito mais estável para redes profundas: os gradientes começam bem-comportados e a contribuição de cada bloco cresce gradualmente conforme `alpha` é ajustado pelo treinamento, em vez de a rede começar com transformações aleatórias fortes em todas as 20+ camadas simultaneamente.

---

## Desafios Avançados (Opcionais)

### Fixação 18.1: ResNets Extremamente Profundas
Repita o Experimento 1 com `num_layers=200`. A rede sem residual ainda propaga algum gradiente mensurável até a entrada? Em que ponto a norma do gradiente se torna numericamente zero (underflow em float32)?

### Fixação 18.2: Highway Networks
Pesquise "Highway Networks" (Srivastava et al., 2015), um predecessor das ResNets que usa um "gate" aprendido em vez de uma soma pura: `y = T(x) * F(x) + (1 - T(x)) * x`. Implemente essa variante e compare o comportamento do gradiente com a soma residual pura.

### Fixação 18.3: Remover Residuals de um Transformer Treinado
Pegue um bloco Transformer com residual connections já treinado (mesmo que superficialmente) e, em modo de avaliação, remova as somas residuais (`x = F(x)` em vez de `x = x + F(x)`) sem retreinar. Meça o impacto na qualidade da saída. O que isso revela sobre quanto os pesos de `F` dependem de operar sobre um "delta" pequeno em relação a `x`?

### Fixação 18.4: Norma de Ativação por Profundidade
Em vez de medir a norma do *gradiente*, meça a norma da *ativação* `x` conforme ela passa por cada bloco residual em uma rede de 50 camadas. A norma cresce, diminui, ou se estabiliza? Isso motiva por que LayerNorm (Capítulo 19) é frequentemente combinado com residual connections.

### Fixação 18.5: Gradient Checkpointing e Residuals
Pesquise como "gradient checkpointing" (usado para economizar memória em redes muito profundas) interage com conexões residuais durante o backward pass. Por que residual connections tornam o recomputation necessário no checkpointing mais previsível?

---

## Resumo

- **Vanishing gradients**: em redes profundas sem atalhos, o gradiente é multiplicado por uma Jacobiana em cada camada; produtos de muitos termos pequenos tendem a zero exponencialmente.
- **Conexão residual**: `output = x + F(x)` em vez de `output = F(x)` — a rede aprende apenas o "resíduo" a ser adicionado à entrada.
- **Gradient highway**: a derivada de `x + F(x)` em relação a `x` é `I + dF/dx`, garantindo que o termo identidade sempre propague o gradiente sem atenuação, não importa o que `F` faça.
- **Identidade como caso degenerado**: se `F(x)` aprende a produzir zero, o bloco vira uma função identidade perfeita — a rede pode "desligar" uma camada sem quebrar o fluxo de informação.
- **No Transformer**: existem duas conexões residuais por bloco (ao redor da atenção e ao redor da MLP), o que exige que toda sub-camada preserve `d_model` na entrada e na saída.
- **Consequência prática**: residual connections são o que torna viável treinar Transformers com dezenas ou centenas de blocos empilhados.

Próximo capítulo: **Layer Normalization** — como normalizar ativações entre camadas para estabilizar ainda mais o treinamento, e por que Transformers preferem LayerNorm a BatchNorm.

---

**Próximo**: [Capítulo 19: Layer Normalization](19_layer_normalization.md)
