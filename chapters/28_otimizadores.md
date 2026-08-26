# Capítulo 28: Otimizadores

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender a fórmula de update do SGD e suas limitações práticas
2. Explicar como Momentum suaviza oscilações acumulando velocidade
3. Derivar as fórmulas de Adam (momentos de primeira e segunda ordem, bias correction) e entender por que ele é o padrão em Transformers
4. Diferenciar weight decay decoupled (AdamW) de L2 regularization tradicional
5. Escolher hiperparâmetros de otimizador (lr, betas, eps, weight_decay) com valores de referência sensatos

---

## Por Que Isso Importa

Você já sabe calcular gradientes — eles apontam a direção que reduz a loss. Mas "seguir o gradiente" não é tão simples quanto parece. Se você já tentou treinar um modelo e viu a loss oscilar violentamente sem nunca convergir, ou cair extremamente devagar apesar de o gradiente parecer "correto", o problema quase sempre está em *como* você usa o gradiente para atualizar os parâmetros — não no gradiente em si.

Imagine descer uma montanha no escuro, sentindo apenas a inclinação do chão sob os pés (o gradiente). Se você sempre dá um passo de tamanho fixo na direção da maior descida (SGD puro), em um vale estreito e alongado você vai ziguezaguear de um lado para o outro da parede do vale, avançando muito pouco na direção que realmente importa. Um esquiador experiente, por outro lado, acumula velocidade na direção em que já vinha se movendo consistentemente — é exatamente essa a intuição por trás do Momentum, e é a base para entender por que Adam, que combina essa ideia com adaptação individual de learning rate por parâmetro, se tornou virtualmente universal para treinar Transformers.

A escolha do otimizador não é um detalhe de implementação — ela determina se seu modelo converge em horas ou nunca converge de jeito nenhum. Praticamente todo paper de LLM usa Adam ou AdamW, com hiperparâmetros bem específicos (`betas=(0.9, 0.95)` ou `(0.9, 0.999)`, `eps=1e-8`), e entender de onde vêm esses números — e por que mudar o `weight_decay` de Adam sem migrar para AdamW pode fazer a regularização parar de funcionar corretamente — é essencial para debugar treinamento em qualquer escala.

---

## SGD: O Ponto de Partida

Stochastic Gradient Descent (SGD) atualiza cada parâmetro subtraindo o gradiente escalado por uma taxa de aprendizado (learning rate):

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

Onde $\eta$ é o learning rate e $\nabla_\theta \mathcal{L}$ é o gradiente da loss em relação a $\theta$.

### Limitações do SGD Puro

**1. Mesmo learning rate para todos os parâmetros.** Alguns parâmetros podem precisar de passos maiores (gradientes consistentemente pequenos, área "achatada" da superfície de loss) e outros de passos menores (gradientes grandes, área "íngreme"). Um único $\eta$ global ignora essa diferença — ou você usa um $\eta$ pequeno o suficiente para não explodir nos parâmetros sensíveis (mas então treina devagar demais nos outros), ou um $\eta$ grande o suficiente para treinar rápido nos parâmetros "achatados" (mas então diverge nos sensíveis).

**2. Oscilação em vales estreitos.** Se a superfície de loss tem curvatura muito diferente em direções diferentes (um "vale" alongado), o gradiente aponta quase perpendicular à direção real de progresso útil, fazendo o otimizador ziguezaguear de uma parede à outra em vez de avançar suavemente pelo fundo do vale.

```python
# SGD puro em PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

---

## Momentum: Acumulando Velocidade

Momentum resolve a oscilação mantendo uma média móvel exponencial (EMA) da direção dos gradientes recentes — uma "velocidade" $v$ que se acumula ao longo dos steps:

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Onde $\beta$ (tipicamente 0.9) controla quanto do "histórico" é preservado. Intuição: se o gradiente aponta consistentemente na mesma direção geral ao longo de vários steps, $v$ cresce naquela direção (acelera). Se o gradiente oscila entre direções opostas (como nas paredes de um vale estreito), essas oscilações se cancelam parcialmente na média, e o movimento líquido segue mais suavemente pelo fundo do vale.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Momentum já é uma melhoria significativa sobre SGD puro, mas ainda usa o mesmo $\eta$ para todos os parâmetros — ele não resolve o problema de "alguns parâmetros precisam de passos diferentes de outros".

---

## Adam: Momentum + Adaptação por Parâmetro

Adam (**Ada**ptive **M**oment Estimation) combina duas ideias: momentum (média móvel do gradiente, chamada de primeiro momento) e adaptação individual de learning rate por parâmetro, baseada em uma média móvel do *quadrado* do gradiente (segundo momento).

### As Fórmulas

Para cada parâmetro, a cada step $t$:

**Primeiro momento (média móvel do gradiente — como Momentum):**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Segundo momento (média móvel do gradiente ao quadrado — mede a "energia"/variância recente do gradiente):**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Onde $g_t = \nabla_\theta \mathcal{L}(\theta_t)$, e $\beta_1, \beta_2$ são tipicamente 0.9 e 0.999.

**Bias correction.** Como $m_0 = v_0 = 0$, nos primeiros steps essas médias móveis são artificialmente puxadas para zero (viés de inicialização). Adam corrige isso dividindo pelo fator $(1 - \beta^t)$, que começa pequeno (correção forte) e se aproxima de 1 conforme $t$ cresce (correção desaparece):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update final:**

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Onde $\epsilon$ (tipicamente `1e-8`) evita divisão por zero.

### Por Que Isso Funciona

A parte crucial é a divisão por $\sqrt{\hat{v}_t}$: se um parâmetro historicamente teve gradientes grandes e voláteis (alto $v_t$), seu passo efetivo é *reduzido* (dividido por um número grande). Se um parâmetro teve gradientes consistentemente pequenos (baixo $v_t$), seu passo efetivo é *ampliado* (dividido por um número pequeno). Isso equaliza automaticamente a velocidade de aprendizado entre parâmetros com escalas de gradiente muito diferentes — exatamente o problema que SGD puro não resolve.

O numerador $\hat{m}_t$ (a versão com bias correction do momentum) garante que a direção do update ainda seja suavizada pelo histórico recente, como no Momentum tradicional — não pulando erraticamente a cada gradiente ruidoso individual.

### Por Que Adam Domina em Transformers

Transformers têm parâmetros com escalas de gradiente extremamente heterogêneas: embeddings, matrizes de atenção, camadas de normalização, MLPs — cada tipo de camada tem uma dinâmica de gradiente diferente, especialmente no início do treinamento quando a inicialização ainda não "assentou". A adaptação por parâmetro do Adam lida bem com essa heterogeneidade sem exigir tuning cuidadoso de learning rate por camada. Além disso, Adam converge de forma robusta mesmo com ruído considerável nos gradientes (como acontece com mini-batches pequenos ou dados muito variados) — condição comum no treinamento de LLMs.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

---

## AdamW: Weight Decay Decoupled

### L2 Regularization Tradicional

Weight decay tradicional (L2 regularization) adiciona um termo à *loss*, penalizando pesos grandes:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2$$

O gradiente desse termo extra é $\lambda \theta$, que é somado ao gradiente original antes de entrar no otimizador:

$$g_t = \nabla_\theta \mathcal{L}(\theta_t) + \lambda \theta_t$$

O problema: em Adam, esse termo extra $\lambda \theta_t$ passa pelo mesmo processo de normalização adaptativa que o resto do gradiente (é acumulado em $m_t$ e $v_t$, dividido por $\sqrt{\hat{v}_t}$). Isso significa que a "força" efetiva do weight decay varia de parâmetro para parâmetro, de forma não intencional e difícil de prever — parâmetros com gradientes historicamente grandes acabam recebendo *menos* penalização de weight decay do que deveriam, porque o termo de regularização é diluído pela mesma normalização adaptativa.

### O Fix do AdamW

AdamW **desacopla** o weight decay do gradiente: em vez de somá-lo ao gradiente antes da normalização adaptativa, ele é aplicado diretamente ao parâmetro, separadamente, depois do update do Adam:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

Agora o weight decay tem magnitude consistente e previsível, independente da escala do gradiente daquele parâmetro especificamente. Essa mudança, aparentemente pequena, produz diferenças mensuráveis de qualidade final do modelo, e por isso **AdamW é o otimizador padrão de fato para treinar Transformers** — não Adam com `weight_decay` passado diretamente.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

---

## Hiperparâmetros e Valores Típicos

| Hiperparâmetro | O que controla | Valor típico (Transformers) |
|---|---|---|
| `lr` (learning rate) | Tamanho do passo geral | `1e-4` a `6e-4` (com warmup + decay) |
| `betas[0]` ($\beta_1$) | Suavização do momentum (1º momento) | `0.9` |
| `betas[1]` ($\beta_2$) | Suavização da variância (2º momento) | `0.95` a `0.999` (LLMs grandes frequentemente usam 0.95) |
| `eps` | Estabilidade numérica na divisão | `1e-8` (ou `1e-5` em precisão reduzida) |
| `weight_decay` | Força da regularização L2 decoupled | `0.01` a `0.1` |

Um detalhe prático importante: `weight_decay` normalmente **não** deve ser aplicado a parâmetros de bias e de LayerNorm (`gamma`/`beta`) — apenas às matrizes de peso. Isso é feito criando grupos de parâmetros separados:

```python
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))
```

---

## Exemplo Numérico Manual: Um Step de Adam

Vamos calcular um único step de Adam à mão para um parâmetro escalar.

**Setup:**

```
theta_0 = 1.0
gradiente g_1 = 0.5  (no primeiro step, t=1)

beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lr = 0.1
m_0 = 0, v_0 = 0
```

**Passo 1 — Atualizar momentos:**

```
m_1 = 0.9 * 0 + 0.1 * 0.5 = 0.05
v_1 = 0.999 * 0 + 0.001 * (0.5)^2 = 0.001 * 0.25 = 0.00025
```

**Passo 2 — Bias correction:**

```
m_hat_1 = m_1 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
v_hat_1 = v_1 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
```

**Passo 3 — Update:**

```
theta_1 = theta_0 - lr * m_hat_1 / (sqrt(v_hat_1) + eps)
        = 1.0 - 0.1 * 0.5 / (sqrt(0.25) + 1e-8)
        = 1.0 - 0.1 * 0.5 / 0.5
        = 1.0 - 0.1
        = 0.9
```

Note que, após bias correction, no primeiro step $\hat{m}_1$ e $\hat{v}_1$ efetivamente recuperam o próprio gradiente (já que $m_1/(1-\beta_1) \approx g_1$ quando não há histórico anterior) — o update do primeiro step do Adam se comporta essencialmente como um passo de SGD normalizado por sua própria magnitude.

---

## Experimento: SGD vs Adam em uma Tarefa de Regressão

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: SGD vs Momentum vs Adam — Trajetórias de Convergência")
print("=" * 70)

# ========== 1. TAREFA SINTÉTICA: REGRESSÃO LINEAR COM VALE ESTREITO ==========
print("\n1. TAREFA: Regressão Linear com Superfície de Loss Mal Condicionada")
print("-" * 70)

# Criamos uma superfície de loss com curvaturas muito diferentes em cada
# eixo, simulando um "vale estreito" — cenário clássico onde SGD sofre.
torch.manual_seed(0)
X = torch.randn(200, 2)
X[:, 0] *= 10.0  # eixo 0 tem escala muito maior -> curvatura diferente
true_w = torch.tensor([0.5, -0.8])
y = X @ true_w + 0.1 * torch.randn(200)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"true_w: {true_w}")

def train(optimizer_name, n_steps=200, lr=0.01):
    torch.manual_seed(42)
    w = torch.zeros(2, requires_grad=True)

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD([w], lr=lr)
    elif optimizer_name == "momentum":
        optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam([w], lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = X @ w
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, w.detach()

# ========== 2. TREINAR COM CADA OTIMIZADOR ==========
print("\n2. TREINANDO COM SGD, MOMENTUM E ADAM (mesmo lr, mesmos steps)")
print("-" * 70)

losses_sgd, w_sgd = train("sgd", n_steps=200, lr=0.001)
losses_mom, w_mom = train("momentum", n_steps=200, lr=0.001)
losses_adam, w_adam = train("adam", n_steps=200, lr=0.001)

print(f"\nLoss final SGD:      {losses_sgd[-1]:.6f} | w final: {w_sgd}")
print(f"Loss final Momentum: {losses_mom[-1]:.6f} | w final: {w_mom}")
print(f"Loss final Adam:     {losses_adam[-1]:.6f} | w final: {w_adam}")
print(f"\nw verdadeiro: {true_w}")

# ========== 3. VELOCIDADE DE CONVERGÊNCIA ==========
print("\n3. LOSS AO LONGO DO TREINO (amostras a cada 40 steps)")
print("-" * 70)

print(f"{'Step':>6} | {'SGD':>12} | {'Momentum':>12} | {'Adam':>12}")
for i in range(0, 200, 40):
    print(f"{i:>6} | {losses_sgd[i]:>12.4f} | {losses_mom[i]:>12.4f} | {losses_adam[i]:>12.4f}")

# ========== 4. COMPARANDO ADAM VS ADAMW COM WEIGHT DECAY ==========
print("\n4. ADAM COM L2 MANUAL VS ADAMW (weight_decay decoupled)")
print("-" * 70)

torch.manual_seed(42)
w1 = torch.zeros(2, requires_grad=True)
opt_adam_l2 = torch.optim.Adam([w1], lr=0.01, weight_decay=0.1)  # L2 acoplado

torch.manual_seed(42)
w2 = torch.zeros(2, requires_grad=True)
opt_adamw = torch.optim.AdamW([w2], lr=0.01, weight_decay=0.1)  # decoupled

for step in range(100):
    opt_adam_l2.zero_grad()
    loss1 = ((X @ w1 - y) ** 2).mean()
    loss1.backward()
    opt_adam_l2.step()

    opt_adamw.zero_grad()
    loss2 = ((X @ w2 - y) ** 2).mean()
    loss2.backward()
    opt_adamw.step()

print(f"w final (Adam + weight_decay acoplado): {w1.detach()}")
print(f"w final (AdamW, weight_decay decoupled): {w2.detach()}")
print("Note como os resultados divergem mesmo com o mesmo weight_decay nominal —")
print("a forma como ele é aplicado muda o resultado final.")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

O padrão esperado: Adam converge de forma mais rápida e estável que SGD puro na tarefa de vale estreito, porque adapta a magnitude do passo por parâmetro (compensando a diferença de escala entre os dois eixos de `X`). Momentum melhora sobre SGD puro, mas ainda menos consistentemente que Adam nesse cenário mal condicionado.

---

## Erros Comuns

### Erro 1: Usar `weight_decay` no `Adam` Esperando o Comportamento de L2 Correto

```python
# CUIDADO — funciona, mas o weight decay é acoplado ao gradiente,
# então sua força efetiva varia por parâmetro de forma não intencional
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

# MELHOR — weight decay desacoplado, comportamento previsível
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

### Erro 2: Learning Rate Não Ajustado ao Otimizador

```python
# ERRADO — usar o mesmo lr "alto" típico de SGD com Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # quase sempre diverge

# CERTO — Adam tipicamente precisa de lr bem menor que SGD
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

### Erro 3: Aplicar Weight Decay em Bias e LayerNorm

```python
# ERRADO — penaliza igualmente todos os parâmetros, incluindo bias e
# parâmetros de LayerNorm, que geralmente não deveriam ser regularizados
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# CERTO — separar em grupos, sem weight decay para bias/LayerNorm
decay, no_decay = [], []
for n, p in model.named_parameters():
    if p.dim() < 2:
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

---

## Exercícios

### Exercício 28.1: Update Manual de SGD
Com `theta=2.0`, `gradiente=0.4`, `lr=0.1`, calcule `theta` após um step de SGD puro.

### Exercício 28.2: Update Manual de Momentum
Com `theta=2.0`, `v_0=0`, `gradiente=0.4` (constante por 3 steps), `beta=0.9`, `lr=0.1`, calcule `theta` após 3 steps de Momentum.

### Exercício 28.3: Um Step Completo de Adam
Repita o cálculo manual da seção "Exemplo Numérico Manual", mas com `gradiente g_1 = -0.3` em vez de `0.5`. Calcule `m_1`, `v_1`, `m_hat_1`, `v_hat_1` e `theta_1`.

### Exercício 28.4: Comparar Convergência
Usando o código do experimento, treine com `lr=0.01` (em vez de `0.001`) para os três otimizadores. O que acontece com SGD? E com Adam?

### Exercício 28.5: Implementar Adam do Zero
Escreva uma função `adam_step(theta, grad, m, v, t, lr, beta1, beta2, eps)` que retorna `(theta_novo, m_novo, v_novo)`, replicando as fórmulas do Adam sem usar `torch.optim`.

---

## Gabarito

### Exercício 28.1: Update Manual de SGD
```python
theta = 2.0
grad = 0.4
lr = 0.1
theta_novo = theta - lr * grad
print(theta_novo)  # 2.0 - 0.04 = 1.96
```

### Exercício 28.2: Update Manual de Momentum
```python
theta, v, beta, lr, grad = 2.0, 0.0, 0.9, 0.1, 0.4

for step in range(3):
    v = beta * v + (1 - beta) * grad
    theta = theta - lr * v
    print(f"step {step+1}: v={v:.4f}, theta={theta:.4f}")

# step 1: v=0.0400, theta=1.9960
# step 2: v=0.0760, theta=1.9884
# step 3: v=0.1084, theta=1.9776
# Note como v cresce a cada step (acumulando na mesma direção),
# fazendo os passos efetivos aumentarem de tamanho.
```

### Exercício 28.3: Um Step Completo de Adam
```python
theta_0 = 1.0
g1 = -0.3
beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.1

m1 = beta1 * 0 + (1 - beta1) * g1          # -0.03
v1 = beta2 * 0 + (1 - beta2) * g1**2       # 0.000090

m_hat1 = m1 / (1 - beta1**1)               # -0.3
v_hat1 = v1 / (1 - beta2**1)               # 0.09

theta1 = theta_0 - lr * m_hat1 / (v_hat1**0.5 + eps)
# = 1.0 - 0.1 * (-0.3) / (0.3 + 1e-8)
# = 1.0 - 0.1 * (-1.0)
# = 1.1
print(theta1)  # ~1.1
```

### Exercício 28.4: Comparar Convergência
```python
# Com lr=0.01, SGD puro tende a divergir ou oscilar fortemente na
# tarefa de vale estreito (loss pode crescer ou ficar instável),
# porque o passo grande demais no eixo de alta curvatura (X[:,0])
# faz overshoot repetidamente.
# Adam, por adaptar o passo por parâmetro, tende a continuar
# convergindo de forma razoavelmente estável mesmo com lr maior,
# embora não seja imune a lr excessivo.
losses_sgd_high, _ = train("sgd", n_steps=200, lr=0.01)
losses_adam_high, _ = train("adam", n_steps=200, lr=0.01)
print(f"SGD lr=0.01 loss final: {losses_sgd_high[-1]:.4f}")
print(f"Adam lr=0.01 loss final: {losses_adam_high[-1]:.4f}")
```

### Exercício 28.5: Implementar Adam do Zero
```python
def adam_step(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    theta_novo = theta - lr * m_hat / (v_hat ** 0.5 + eps)
    return theta_novo, m, v

# Teste
theta, m, v = 1.0, 0.0, 0.0
for t in range(1, 4):
    theta, m, v = adam_step(theta, grad=0.5, m=m, v=v, t=t, lr=0.1)
    print(f"t={t}: theta={theta:.4f}")
```

---

## Desafios Avançados (Opcionais)

### Fixação 28.1: Reproduzir `torch.optim.Adam` Exatamente
Implemente uma classe `MeuAdam` completa (com estado por parâmetro) e compare, passo a passo, com `torch.optim.Adam` em uma rede pequena real, verificando que os pesos ficam idênticos até tolerância numérica.

### Fixação 28.2: RMSprop como Caso Especial
RMSprop usa apenas o segundo momento (sem momentum de primeira ordem). Implemente e compare com Adam — em que cenários a ausência do primeiro momento faz diferença?

### Fixação 28.3: Efeito de $\beta_2$ em LLMs Grandes
Pesquise (ou raciocine) por que muitos LLMs grandes usam `beta2=0.95` em vez do padrão `0.999`. Qual o efeito de uma janela de média móvel mais curta para o segundo momento?

### Fixação 28.4: Learning Rate Muito Alto com Adam
Treine a tarefa de regressão do experimento com `lr=1.0` no Adam. Observe o comportamento da loss. Isso mostra que a adaptação de Adam não é uma proteção absoluta contra learning rate mal escolhido.

### Fixação 28.5: Visualizar a Trajetória no Espaço de Parâmetros
Para o problema 2D do experimento, plote a trajetória de `w` (os dois componentes) ao longo do treino para SGD, Momentum e Adam sobre curvas de nível da loss. Observe visualmente o ziguezague do SGD vs a trajetória mais direta do Adam.

---

## Resumo

- **SGD**: `theta -= lr * grad` — simples, mas usa o mesmo passo para todos os parâmetros e oscila em vales estreitos
- **Momentum**: acumula uma média móvel do gradiente (velocidade), suavizando oscilações e acelerando na direção consistente
- **Adam**: combina momentum (1º momento) com adaptação de passo por parâmetro baseada na variância recente do gradiente (2º momento), com bias correction nos primeiros steps
- **AdamW**: desacopla weight decay do gradiente, aplicando-o diretamente ao parâmetro — comportamento de regularização mais previsível que L2 acoplado
- **Hiperparâmetros típicos**: `lr` ~`1e-4` a `6e-4`, `betas=(0.9, 0.95 ou 0.999)`, `eps=1e-8`, `weight_decay` ~`0.01` a `0.1`, sem decay em bias/LayerNorm
- **Adam é o padrão em Transformers**: lida bem com gradientes heterogêneos entre diferentes tipos de camada, sem precisar de tuning fino de lr por camada

Próximo capítulo: **Treinamento Autoregressivo** — como montar o loop de treinamento completo, preparar pares (input, target) por shift-by-one, e usar teacher forcing.

---

**Próximo**: [Capítulo 29: Treinamento Autoregressivo](29_treinamento_autoregressivo.md)
