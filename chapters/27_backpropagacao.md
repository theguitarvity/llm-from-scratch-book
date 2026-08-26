# Capítulo 27: Backpropagação e Gradientes

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Revisar a regra da cadeia (chain rule) e entender como ela fundamenta o backpropagation
2. Entender como o autograd do PyTorch constrói e percorre um grafo computacional automaticamente
3. Calcular gradientes manualmente em uma rede pequena de 2 camadas e comparar com `loss.backward()`
4. Inspecionar gradientes de parâmetros específicos para debugging
5. Diagnosticar e corrigir problemas comuns: gradientes `None`, `NaN`/`Inf`, acúmulo indevido e exploding gradients

---

## Por Que Isso Importa

Você já sabe calcular a loss — um único número que diz "o quão errado" o modelo está. Mas um número sozinho não ensina nada a ninguém. O que realmente treina o modelo é saber, para *cada um* dos milhões de parâmetros, **em qual direção e o quanto** movê-lo para reduzir essa loss. Essa informação é o gradiente, e o algoritmo que a calcula de forma eficiente para redes profundas é a backpropagação.

Se você já treinou um modelo em PyTorch, provavelmente escreveu `loss.backward()` sem pensar muito sobre o que acontece por baixo. Isso funciona bem — até não funcionar. Você vai, mais cedo ou mais tarde, se deparar com `param.grad` sendo `None` quando não deveria, com a loss virando `NaN` do nada após algumas iterações, ou com o modelo simplesmente não aprendendo nada apesar do código "parecer certo". Em todos esses casos, a única forma de debugar de verdade é entender o que backprop está fazendo mecanicamente — porque os sintomas aparecem nos gradientes, não na loss.

Pense em um caso real: você implementa uma normalização customizada e, sem perceber, usa uma operação in-place (`x += 1` em vez de `x = x + 1`) em um tensor que faz parte do grafo computacional. O forward pass roda perfeitamente, a loss é calculada, mas quando você chama `.backward()`, ou você recebe um erro cripticamente relacionado a "a leaf Variable that requires grad is being used in an in-place operation", ou pior — nenhum erro, mas o gradiente correspondente fica zerado ou desconectado, e o parâmetro simplesmente nunca se move. Sem entender como o grafo computacional é construído e como ele se conecta, esse tipo de bug é praticamente impossível de rastrear por tentativa e erro.

Este capítulo constrói essa intuição do zero: regra da cadeia, grafo computacional, gradientes calculados à mão comparados com autograd, e o catálogo de problemas que você vai efetivamente encontrar ao treinar um Transformer.

---

## Revisão: A Regra da Cadeia

Se $y = f(u)$ e $u = g(x)$, então a derivada de $y$ em relação a $x$ é:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Essa é a regra da cadeia — a derivada de uma composição de funções é o produto das derivadas de cada etapa. Redes neurais são, literalmente, composições longas de funções: embedding, projeção linear, atenção, MLP, normalização, camadas empilhadas repetidamente. Uma rede com $L$ camadas é uma composição de $L$ funções:

$$y = f_L(f_{L-1}(\dots f_2(f_1(x))\dots))$$

Para calcular como a loss final muda em relação a um parâmetro na primeira camada, a regra da cadeia diz que precisamos multiplicar as derivadas de *cada camada intermediária*, uma por uma, da última até a primeira:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial \theta_1}$$

Backpropagation é simplesmente um algoritmo eficiente para calcular essa cadeia de produtos **de trás para frente** (da loss até os parâmetros mais distantes), reaproveitando resultados intermediários em vez de recalcular tudo do zero para cada parâmetro. Sem essa reutilização, treinar uma rede com milhões de parâmetros seria computacionalmente inviável.

---

## O Grafo Computacional

Toda operação que você faz com tensores que têm `requires_grad=True` é registrada pelo PyTorch em um **grafo computacional dinâmico**: um grafo direcionado onde os nós são operações (soma, multiplicação, matmul, softmax, etc.) e as arestas representam o fluxo de dados.

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

u = w * x       # nó: multiplicação
y = u + b       # nó: soma
loss = y ** 2   # nó: potência
```

Esse código constrói o grafo:

```
x, w ---(mul)---> u ---(add com b)---> y ---(pow 2)---> loss
```

Cada nó "lembra" qual operação o criou e quais foram suas entradas — isso é armazenado no atributo `.grad_fn` do tensor resultante. Quando você chama `loss.backward()`, o PyTorch percorre esse grafo **de trás para frente**, aplicando a regra da cadeia em cada nó, acumulando o gradiente em `.grad` de cada tensor folha (`leaf`) que tem `requires_grad=True`.

```python
print(loss.grad_fn)        # <PowBackward0 ...>
print(loss.grad_fn.next_functions)  # aponta para o nó anterior (AddBackward0)
```

**Importante**: apenas tensores "folha" (criados diretamente por você, não resultado de uma operação) acumulam gradiente em `.grad` por padrão. Tensores intermediários (como `u` e `y` acima) têm gradiente calculado durante o backward, mas não o armazenam, a menos que você chame `.retain_grad()` explicitamente neles.

---

## Backprop Manual em uma Rede de 2 Camadas

Vamos construir a rede mais simples possível — uma camada linear seguida de outra — e calcular os gradientes manualmente, depois comparar com autograd.

### Setup

```
Entrada:  x = [1.0, 2.0]         (vetor 1x2)
Camada 1: W1 = [[0.5, -0.3],     (2x2)
                [0.2,  0.8]]
Camada 2: W2 = [1.0, -1.0]       (vetor 1x2, projeta para escalar)
Target:   y = 3.0
Loss:     MSE = (pred - y)^2
```

### Forward Pass (manual)

```
h = x @ W1
h[0] = x[0]*W1[0,0] + x[1]*W1[1,0] = 1.0*0.5 + 2.0*0.2 = 0.9
h[1] = x[0]*W1[0,1] + x[1]*W1[1,1] = 1.0*(-0.3) + 2.0*0.8 = 1.3

h = [0.9, 1.3]

pred = h @ W2 = 0.9*1.0 + 1.3*(-1.0) = 0.9 - 1.3 = -0.4

loss = (pred - y)^2 = (-0.4 - 3.0)^2 = (-3.4)^2 = 11.56
```

### Backward Pass (manual, aplicando a regra da cadeia)

**Passo 1 — Gradiente da loss em relação a `pred`:**

$$\frac{\partial \mathcal{L}}{\partial \text{pred}} = 2 (\text{pred} - y) = 2 \times (-3.4) = -6.8$$

**Passo 2 — Gradiente em relação a `W2`** (já que `pred = h @ W2`, ou seja `pred = h[0]*W2[0] + h[1]*W2[1]`):

$$\frac{\partial \text{pred}}{\partial W2[i]} = h[i] \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial W2[i]} = \frac{\partial \mathcal{L}}{\partial \text{pred}} \cdot h[i]$$

```
dL/dW2[0] = -6.8 * h[0] = -6.8 * 0.9 = -6.12
dL/dW2[1] = -6.8 * h[1] = -6.8 * 1.3 = -8.84
```

**Passo 3 — Gradiente em relação a `h`** (necessário para continuar propagando até `W1`):

$$\frac{\partial \text{pred}}{\partial h[i]} = W2[i] \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial h[i]} = \frac{\partial \mathcal{L}}{\partial \text{pred}} \cdot W2[i]$$

```
dL/dh[0] = -6.8 * W2[0] = -6.8 * 1.0 = -6.8
dL/dh[1] = -6.8 * W2[1] = -6.8 * (-1.0) = 6.8
```

**Passo 4 — Gradiente em relação a `W1`** (já que `h[j] = sum_i x[i] * W1[i,j]`):

$$\frac{\partial \mathcal{L}}{\partial W1[i,j]} = \frac{\partial \mathcal{L}}{\partial h[j]} \cdot x[i]$$

```
dL/dW1[0,0] = dL/dh[0] * x[0] = -6.8 * 1.0 = -6.8
dL/dW1[1,0] = dL/dh[0] * x[1] = -6.8 * 2.0 = -13.6
dL/dW1[0,1] = dL/dh[1] * x[0] =  6.8 * 1.0 =  6.8
dL/dW1[1,1] = dL/dh[1] * x[1] =  6.8 * 2.0 = 13.6
```

Note como cada passo reutiliza o gradiente calculado no passo anterior (`dL/dpred` é usado tanto para `W2` quanto para `h`; `dL/dh` é usado para `W1`) — essa reutilização é exatamente o que torna backprop eficiente em vez de recalcular a cadeia inteira para cada parâmetro individualmente.

Vamos verificar esses números com autograd no experimento a seguir.

---

## Experimento: Backprop Manual vs Autograd

```python
import torch

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Backpropagação Manual vs Autograd")
print("=" * 70)

# ========== 1. SETUP ==========
print("\n1. SETUP DA REDE DE 2 CAMADAS")
print("-" * 70)

x = torch.tensor([1.0, 2.0])
W1 = torch.tensor([[0.5, -0.3], [0.2, 0.8]], requires_grad=True)
W2 = torch.tensor([1.0, -1.0], requires_grad=True)
y_target = torch.tensor(3.0)

print(f"x = {x}")
print(f"W1 =\n{W1}")
print(f"W2 = {W2}")
print(f"y_target = {y_target}")

# ========== 2. FORWARD PASS ==========
print("\n2. FORWARD PASS")
print("-" * 70)

h = x @ W1
print(f"h = x @ W1 = {h}")

pred = h @ W2
print(f"pred = h @ W2 = {pred.item():.4f}")

loss = (pred - y_target) ** 2
print(f"loss = (pred - y)^2 = {loss.item():.4f}")

# ========== 3. BACKWARD VIA AUTOGRAD ==========
print("\n3. BACKWARD PASS (autograd)")
print("-" * 70)

loss.backward()

print(f"W1.grad =\n{W1.grad}")
print(f"W2.grad = {W2.grad}")

# ========== 4. COMPARAÇÃO COM CÁLCULO MANUAL ==========
print("\n4. COMPARAÇÃO COM GRADIENTES CALCULADOS À MÃO")
print("-" * 70)

W1_grad_manual = torch.tensor([[-6.8, 6.8], [-13.6, 13.6]])
W2_grad_manual = torch.tensor([-6.12, -8.84])

print(f"W1.grad (autograd) =\n{W1.grad}")
print(f"W1.grad (manual)   =\n{W1_grad_manual}")
print(f"Diferença máxima: {(W1.grad - W1_grad_manual).abs().max().item():.6f}")

print(f"\nW2.grad (autograd) = {W2.grad}")
print(f"W2.grad (manual)   = {W2_grad_manual}")
print(f"Diferença máxima: {(W2.grad - W2_grad_manual).abs().max().item():.6f}")

# ========== 5. INSPECIONANDO O GRAFO COMPUTACIONAL ==========
print("\n5. INSPECIONANDO O GRAFO COMPUTACIONAL")
print("-" * 70)

x2 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)
b2 = torch.tensor(1.0, requires_grad=True)

u = w2 * x2
y2 = u + b2
loss2 = y2 ** 2

print(f"loss2.grad_fn: {loss2.grad_fn}")
print(f"loss2.grad_fn.next_functions: {loss2.grad_fn.next_functions}")
print("Cada next_functions aponta para o nó anterior no grafo,")
print("formando a cadeia que o backward vai percorrer de trás para frente.")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Saída esperada (resumida):

```
W1.grad (autograd) =
tensor([[ -6.8000,   6.8000],
        [-13.6000,  13.6000]])
W2.grad (autograd) = tensor([-6.1200, -8.8400])
Diferença máxima: 0.000000
```

Os gradientes calculados manualmente batem exatamente com os do autograd — o que confirma que backprop nada mais é do que a regra da cadeia aplicada sistematicamente, e o PyTorch apenas automatiza a contabilidade.

---

## Problemas Comuns com Gradientes

### Gradientes `None`

Se `param.grad` é `None` após `.backward()`, uma das causas mais comuns:

```python
# Causa 1: requires_grad não foi setado
w = torch.tensor([1.0, 2.0])  # requires_grad=False por padrão!
# ... operações ...
loss.backward()
print(w.grad)  # None

# Causa 2: o tensor não é folha e não pediu retain_grad
x = torch.tensor(1.0, requires_grad=True)
y = x * 2  # y não é folha
z = y ** 2
z.backward()
print(y.grad)  # None (aviso: "não é leaf tensor")

# Causa 3: operação desconectou o grafo (ex: .detach(), .item(), numpy())
w = torch.tensor([1.0], requires_grad=True)
w_detached = w.detach()  # corta o grafo aqui
loss = (w_detached * 2).sum()
loss.backward()  # ERRO: w_detached não requer grad
```

### Gradientes `NaN` ou `Inf`

Geralmente causados por:
- **Learning rate alto demais**, fazendo os pesos explodirem em poucos passos
- **Divisão por zero** ou `log(0)` em alguma operação customizada (ex: cross-entropy manual sem clamp de probabilidade)
- **Overflow numérico** em exponenciais (ex: softmax sem subtração do máximo)

```python
# Diagnóstico simples: checar NaN antes de dar step
if torch.isnan(loss) or torch.isinf(loss):
    print("Loss inválida! Abortando este step.")
```

### Esquecer `zero_grad()` (Acúmulo Indevido)

O PyTorch **acumula** gradientes por padrão a cada `.backward()` — não os substitui. Isso é uma feature (útil para simular batches maiores acumulando gradientes de múltiplos mini-batches), mas é uma armadilha se você esquecer de zerar entre steps de otimização distintos:

```python
# ERRADO — gradientes de steps anteriores se acumulam com os novos
for batch in dataloader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    # esqueceu optimizer.zero_grad()!

# CERTO
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
```

Sem `zero_grad()`, cada step usa uma mistura acumulada de gradientes de vários batches diferentes, geralmente muito maiores em magnitude do que deveriam ser — os pesos são atualizados de forma inconsistente e o treinamento diverge ou se torna extremamente instável, muitas vezes com a loss oscilando descontroladamente em vez de convergir.

### Exploding Gradients e Gradient Clipping

Redes profundas (Transformers com dezenas de camadas empilhadas) podem sofrer de **exploding gradients**: como o gradiente de cada camada é multiplicado pelo gradiente da camada seguinte (regra da cadeia), se cada multiplicação amplifica ligeiramente a magnitude, o produto final pode crescer exponencialmente com a profundidade da rede.

A solução padrão é **gradient clipping**: limitar a norma total dos gradientes antes do `optimizer.step()`.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`clip_grad_norm_` calcula a norma L2 de todos os gradientes combinados e, se ela exceder `max_norm`, reescala **todos** os gradientes proporcionalmente para que a norma total fique exatamente igual a `max_norm`. Isso preserva a *direção* do gradiente (importante — ainda aponta para onde reduzir a loss), apenas limita sua *magnitude*. É praticamente padrão em treinamento de Transformers, porque a combinação de atenção, muitas camadas e sequências longas cria condições propícias para picos ocasionais de gradiente.

---

## Experimento 2: Gradient Clipping em Ação

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Gradient Clipping")
print("=" * 70)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

# Simular um caso de gradiente explosivo com uma loss artificialmente grande
x = torch.randn(4, 4)
y = torch.tensor([[1000.0], [1000.0], [1000.0], [1000.0]])  # target absurdo

pred = model(x)
loss = ((pred - y) ** 2).mean()

print(f"\nLoss (artificialmente grande): {loss.item():.2f}")

loss.backward()

# Norma total dos gradientes ANTES do clipping
total_norm_before = torch.sqrt(
    sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
)
print(f"Norma total dos gradientes ANTES do clip: {total_norm_before.item():.2f}")

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

total_norm_after = torch.sqrt(
    sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
)
print(f"Norma total dos gradientes DEPOIS do clip: {total_norm_after.item():.4f}")
print("(Deve ser <= max_norm=1.0)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Operação In-Place que Quebra o Grafo

```python
# ERRADO
h = x @ W1
h += 1.0  # operação in-place em tensor que faz parte do grafo!
loss = (h @ W2 - y) ** 2
loss.backward()  # RuntimeError em muitos casos, ou grafo corrompido silenciosamente

# CERTO
h = x @ W1
h = h + 1.0  # cria novo tensor, preserva o grafo
loss = (h @ W2 - y) ** 2
loss.backward()
```

### Erro 2: Esquecer `zero_grad()`

```python
# ERRADO — gradientes acumulam indefinidamente entre steps
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()

# CERTO
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
```

### Erro 3: Fazer `.backward()` Sem Limpar o Grafo em Loops de Debug

```python
# ERRADO — tentar chamar backward() duas vezes no mesmo grafo sem retain_graph
loss.backward()
loss.backward()  # RuntimeError: buffers já foram liberados

# CERTO (se realmente precisar backward duas vezes no mesmo grafo)
loss.backward(retain_graph=True)
loss.backward()

# Mas o padrão normal é recomputar o forward a cada iteração:
for step in range(n_steps):
    optimizer.zero_grad()
    loss = compute_loss()  # novo forward, novo grafo
    loss.backward()
    optimizer.step()
```

---

## Exercícios

### Exercício 27.1: Regra da Cadeia Manual
Dado $y = (3x + 2)^2$, calcule $\frac{dy}{dx}$ em $x=1$ manualmente usando a regra da cadeia. Depois confirme com autograd.

### Exercício 27.2: Grafo Computacional
Construa o tensor `z = (a * b + c) ** 2` com `a, b, c` como leaves com `requires_grad=True`. Imprima `z.grad_fn` e `z.grad_fn.next_functions`, e explique cada elemento.

### Exercício 27.3: Diagnosticar Gradiente `None`
O código abaixo produz `w.grad = None`. Encontre o problema e corrija:
```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
w_scaled = w * 2
w_detached = w_scaled.detach()
loss = (w_detached ** 2).sum()
loss.backward()
print(w.grad)
```

### Exercício 27.4: Implementar Gradient Clipping Manual
Sem usar `clip_grad_norm_`, escreva uma função que calcule a norma total dos gradientes de um modelo e os reescale se excederem um `max_norm`.

### Exercício 27.5: Acúmulo de Gradiente Proposital
Simule "gradient accumulation" (técnica para simular batch size maior): faça `.backward()` em 4 mini-batches sem chamar `zero_grad()` entre eles, depois um único `optimizer.step()`. Compare o gradiente acumulado com a soma dos gradientes individuais.

---

## Gabarito

### Exercício 27.1: Regra da Cadeia Manual
```python
import torch

# Manual: y = (3x+2)^2, dy/dx = 2*(3x+2)*3 = 6*(3x+2)
# em x=1: 6*(3*1+2) = 6*5 = 30

x = torch.tensor(1.0, requires_grad=True)
y = (3 * x + 2) ** 2
y.backward()
print(x.grad)  # tensor(30.)
```

### Exercício 27.2: Grafo Computacional
```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(1.0, requires_grad=True)

z = (a * b + c) ** 2
print(z.grad_fn)                  # PowBackward0 (a última operação: **2)
print(z.grad_fn.next_functions)   # aponta para AddBackward0 (a soma a*b + c)
# Percorrendo mais fundo levaria a MulBackward0 (a*b) e às leaves a, b, c
```

### Exercício 27.3: Diagnosticar Gradiente `None`
```python
# Problema: .detach() corta o grafo — w_detached não tem ligação de volta a w.
w = torch.tensor([1.0, 2.0], requires_grad=True)
w_scaled = w * 2
loss = (w_scaled ** 2).sum()  # sem detach, grafo intacto
loss.backward()
print(w.grad)  # agora funciona: tensor([8., 16.])
```

### Exercício 27.4: Implementar Gradient Clipping Manual
```python
import torch

def clip_grad_norm_manual(parameters, max_norm):
    total_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in parameters if p.grad is not None)
    )
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
    return total_norm

# Uso:
# clip_grad_norm_manual(model.parameters(), max_norm=1.0)
```

### Exercício 27.5: Acúmulo de Gradiente Proposital
```python
import torch
import torch.nn as nn

torch.manual_seed(42)
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

batches = [torch.randn(2, 3) for _ in range(4)]

optimizer.zero_grad()
for batch in batches:
    loss = model(batch).sum()
    loss.backward()  # acumula, sem zero_grad entre iterações

grad_acumulado = model.weight.grad.clone()

# Comparar somando gradientes individuais
optimizer.zero_grad()
grad_soma = torch.zeros_like(model.weight)
for batch in batches:
    optimizer.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    grad_soma += model.weight.grad

print(f"Gradiente acumulado: {grad_acumulado}")
print(f"Soma dos gradientes individuais: {grad_soma}")
# Devem ser (aproximadamente) iguais
```

---

## Desafios Avançados (Opcionais)

### Fixação 27.1: Vanishing Gradients
Construa uma rede sequencial muito profunda (20+ camadas lineares com `tanh`) sem residual connections. Observe como a magnitude do gradiente na primeira camada se compara à da última. Isso ilustra por que residual connections (Capítulo 18) são importantes.

### Fixação 27.2: Hooks de Gradiente
Use `tensor.register_hook()` para imprimir o gradiente de um tensor intermediário durante o backward, sem precisar de `retain_grad()`.

### Fixação 27.3: Gradiente Numérico vs Analítico
Implemente verificação de gradiente por diferenças finitas (`(f(x+eps) - f(x-eps)) / (2*eps)`) para uma função simples, e compare com o gradiente do autograd. Essa é a técnica clássica de "gradient checking".

### Fixação 27.4: Grafo Computacional Compartilhado
Crie um tensor `y` usado em duas ramificações diferentes do grafo (ex: `loss = f(y) + g(y)`). Verifique que o gradiente final em `y` é a soma das contribuições de cada ramo — a regra da cadeia se generaliza para múltiplos caminhos.

### Fixação 27.5: Clipping por Valor vs por Norma
Compare `clip_grad_norm_` (que preserva a direção do gradiente) com `clip_grad_value_` (que trunca cada componente individualmente, podendo distorcer a direção). Em qual cenário cada um faz mais sentido?

---

## Resumo

- **Regra da cadeia**: fundamento matemático do backprop — derivada de uma composição é o produto das derivadas de cada etapa
- **Grafo computacional**: PyTorch registra automaticamente cada operação sobre tensores com `requires_grad=True`, formando um grafo que `.backward()` percorre de trás para frente
- **Gradientes manuais vs autograd**: batem exatamente — autograd apenas automatiza a contabilidade da regra da cadeia
- **`param.grad = None`**: geralmente indica `requires_grad=False`, tensor não-folha sem `retain_grad`, ou grafo cortado por `.detach()`
- **`zero_grad()`**: obrigatório antes de cada novo `.backward()` — gradientes se acumulam por padrão
- **Gradient clipping**: `clip_grad_norm_` limita a magnitude do gradiente preservando a direção, essencial para estabilizar treinamento de Transformers

Próximo capítulo: **Otimizadores** — como o gradiente é efetivamente usado para atualizar os parâmetros, e por que Adam venceu SGD como padrão em Transformers.

---

**Próximo**: [Capítulo 28: Otimizadores](28_otimizadores.md)
