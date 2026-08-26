# Capítulo 19: Layer Normalization

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender por que ativações precisam ser normalizadas entre camadas de uma rede profunda
2. Diferenciar BatchNorm de LayerNorm e explicar por que Transformers usam LayerNorm
3. Aplicar a fórmula de normalização com os parâmetros aprendidos gamma e beta
4. Comparar Pre-LN e Post-LN e justificar por que modelos modernos preferem Pre-LN
5. Implementar LayerNorm do zero e validar contra `nn.LayerNorm`

---

## Por Que Isso Importa

No Capítulo 18 você viu que residual connections resolvem o problema de o gradiente desaparecer ao atravessar muitas camadas. Mas resolver o fluxo do *gradiente* não resolve outro problema irmão: o fluxo das *ativações*. Cada bloco residual faz `x = x + F(x)`, o que significa que a magnitude de `x` pode crescer a cada camada — afinal, você está somando algo a ele repetidamente, camada após camada. Em uma rede de 24 ou 48 blocos, sem nenhum controle, a norma das ativações pode explodir (crescendo indefinidamente) ou, dependendo da inicialização, colapsar para valores muito pequenos ou muito grandes de forma inconsistente entre exemplos do batch. Isso é exatamente o tipo de instabilidade numérica que causa loss = `nan` no meio do treinamento, gradientes que explodem, ou treinamento que simplesmente não converge por mais que você ajuste a learning rate.

Se você já treinou uma rede profunda e viu a loss virar `nan` depois de algumas centenas de steps sem motivo aparente, é bem provável que o problema seja exatamente esse: ativações crescendo sem controle até estourarem a precisão numérica do float32. A solução é normalizar as ativações em algum ponto do pipeline — forçá-las a ter média zero e variância um antes de seguirem para a próxima transformação, de forma que a escala das ativações permaneça previsível e estável independentemente de quantas camadas você empilhar.

A pergunta natural é: normalizar *como*? Existem várias formas de normalização, e a escolha específica que os Transformers fazem — Layer Normalization, em vez do BatchNorm popularizado nas redes convolucionais — não é arbitrária. Ela decorre diretamente de uma característica central dos dados que Transformers processam: sequências de texto, que têm comprimento variável e são frequentemente processadas em lotes pequenos (ou até um exemplo por vez, durante geração de texto). Entender por que essa escolha importa é entender uma das decisões de design mais silenciosamente importantes da arquitetura Transformer.

---

## BatchNorm vs. LayerNorm

### A Fórmula Geral de Normalização

Ambas as técnicas seguem o mesmo esqueleto matemático: subtrair a média, dividir pelo desvio padrão, depois reescalar e deslocar com parâmetros aprendidos:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma \odot \hat{x} + \beta$$

A diferença inteira entre BatchNorm e LayerNorm está em **sobre qual conjunto de valores $\mu$ e $\sigma^2$ são calculados**.

### BatchNorm: Normaliza Através do Batch

Para uma ativação com shape `[batch, features]`, o BatchNorm calcula média e variância **para cada feature, através de todos os exemplos do batch**:

$$\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{i,j}, \quad \sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2$$

onde $N$ é o tamanho do batch e $j$ indexa a feature. Ou seja: para normalizar o valor do neurônio 5 do exemplo 3, o BatchNorm olha o valor do neurônio 5 em **todos os exemplos do batch** (exemplo 1, exemplo 2, ..., exemplo N) para calcular a média e o desvio.

Isso tem duas consequências problemáticas para Transformers:

1. **Depende do tamanho do batch**: com batch pequeno (ou batch=1, comum na inferência autoregressiva, onde você gera um token por vez), a estimativa de média e variância feita sobre 1 ou poucos exemplos é extremamente ruidosa e instável.
2. **Estatísticas de treino e inferência divergem**: BatchNorm mantém médias móveis (`running_mean`, `running_var`) calculadas durante o treino para usar na inferência, o que introduz uma discrepância entre o comportamento em treino e em produção — uma fonte clássica de bugs sutis.

### LayerNorm: Normaliza Através das Features

Para uma ativação com shape `[batch, seq_len, d_model]`, o LayerNorm calcula média e variância **para cada posição de cada exemplo, através da dimensão de features (`d_model`)**:

$$\mu_i = \frac{1}{d} \sum_{k=1}^{d} x_{i,k}, \quad \sigma_i^2 = \frac{1}{d} \sum_{k=1}^{d} (x_{i,k} - \mu_i)^2$$

onde $d = d_{model}$ e $i$ indexa a posição (token) dentro de um exemplo. Ou seja: para normalizar o token na posição 3 do exemplo 1, o LayerNorm olha **todos os `d_model` valores daquele único token**, e apenas eles — nunca olha outros tokens, nem outros exemplos do batch.

Essa é a diferença chave: **LayerNorm normaliza cada token individualmente, de forma completamente independente do batch e da posição de outros tokens na sequência.**

### Por Que Isso Importa para Transformers

1. **Independência do tamanho do batch**: como o cálculo de $\mu_i$ e $\sigma_i^2$ usa apenas as `d_model` features do próprio token, funciona igualmente bem com `batch_size=1` ou `batch_size=256` — não há degradação estatística com batches pequenos.
2. **Independência do comprimento da sequência**: sequências de texto têm comprimentos variáveis. BatchNorm teria que lidar com padding de forma cuidadosa (tokens de padding contaminando as estatísticas), enquanto LayerNorm normaliza cada posição isoladamente, sem se importar com quantos tokens existem na sequência.
3. **Comportamento idêntico entre treino e inferência**: LayerNorm não mantém médias móveis — calcula $\mu$ e $\sigma^2$ diretamente a cada forward pass, seja em treino ou inferência. Não há discrepância de comportamento.

| Aspecto | BatchNorm | LayerNorm |
|---|---|---|
| Normaliza sobre | exemplos do batch (por feature) | features do próprio token (por exemplo) |
| Depende de batch_size | Sim (degrada com batch pequeno) | Não |
| Estatísticas treino vs. inferência | Diferentes (running stats) | Idênticas |
| Adequado para sequências variáveis | Requer cuidado com padding | Naturalmente robusto |
| Uso típico | CNNs (visão computacional) | Transformers (NLP, sequências) |

---

## A Fórmula Completa e os Parâmetros Aprendidos

Para uma ativação $x \in \mathbb{R}^{[\text{batch}, \text{seq\_len}, d_{model}]}$, o LayerNorm opera assim:

**Passo 1 — Média e variância ao longo de `d_model`** (última dimensão):

$$\mu = \text{mean}(x, \text{dim}=-1), \quad \sigma^2 = \text{var}(x, \text{dim}=-1)$$

Shapes: $\mu, \sigma^2 \in \mathbb{R}^{[\text{batch}, \text{seq\_len}, 1]}$ (mantendo a dimensão para broadcasting).

**Passo 2 — Normalizar:**

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

O $\epsilon$ (tipicamente `1e-5`) evita divisão por zero quando $\sigma^2 \approx 0$ (o que aconteceria, por exemplo, se todos os valores do token fossem idênticos).

**Passo 3 — Escalar e deslocar com parâmetros aprendidos:**

$$y = \gamma \odot \hat{x} + \beta$$

onde $\gamma, \beta \in \mathbb{R}^{[d_{model}]}$ são vetores de parâmetros **aprendidos por gradiente descendente**, um valor por dimensão de feature (compartilhados entre todas as posições e exemplos).

Por que ter $\gamma$ e $\beta$ em vez de simplesmente deixar a saída normalizada (média 0, variância 1)? Porque forçar toda ativação a ter exatamente média 0 e variância 1 pode ser uma restrição excessiva — talvez a rede "precise" de ativações com uma escala ou deslocamento diferente para uma camada específica funcionar bem. $\gamma$ e $\beta$ dão à rede a liberdade de desfazer a normalização se isso for útil (por exemplo, inicializando $\gamma=1, \beta=0$, o que reproduz exatamente a saída normalizada, e deixando o gradiente ajustar a partir daí).

### Exemplo Numérico Manual

Vamos normalizar um único token com `d_model = 4`:

```
x = [2.0, 4.0, 4.0, 6.0]
```

**Passo 1 — Média e variância:**

```
mu = (2.0 + 4.0 + 4.0 + 6.0) / 4 = 16.0 / 4 = 4.0

variância (populacional, divisor = d, não d-1):
var = [(2-4)^2 + (4-4)^2 + (4-4)^2 + (6-4)^2] / 4
    = [4 + 0 + 0 + 4] / 4
    = 8 / 4 = 2.0
```

**Passo 2 — Normalizar** (com $\epsilon = 1e-5$, desprezível aqui):

```
sqrt(var + eps) = sqrt(2.0 + 0.00001) ≈ 1.41425

x_hat[0] = (2.0 - 4.0) / 1.41425 = -2.0 / 1.41425 ≈ -1.4142
x_hat[1] = (4.0 - 4.0) / 1.41425 =  0.0 / 1.41425 =  0.0
x_hat[2] = (4.0 - 4.0) / 1.41425 =  0.0
x_hat[3] = (6.0 - 4.0) / 1.41425 =  2.0 / 1.41425 ≈  1.4142

x_hat = [-1.4142, 0.0, 0.0, 1.4142]
```

Verificação: média de `x_hat` = 0.0, variância de `x_hat` = 1.0 (por construção).

**Passo 3 — Escalar e deslocar**, supondo `gamma = [1.0, 1.0, 1.0, 1.0]` e `beta = [0.5, 0.5, 0.5, 0.5]` (valores hipotéticos aprendidos):

```
y[0] = 1.0 * (-1.4142) + 0.5 = -0.9142
y[1] = 1.0 * (0.0)     + 0.5 =  0.5
y[2] = 1.0 * (0.0)     + 0.5 =  0.5
y[3] = 1.0 * (1.4142)  + 0.5 =  1.9142

y = [-0.9142, 0.5, 0.5, 1.9142]
```

Note que, se `gamma=1` e `beta=0`, `y` é exatamente `x_hat` — a saída normalizada pura. Com `gamma` e `beta` aprendidos, a rede pode deslocar e reescalar essa distribuição para o que for mais útil para as camadas seguintes.

---

## Pre-LN vs. Post-LN

Onde exatamente o LayerNorm é aplicado em relação à conexão residual (Capítulo 18) é uma decisão de design com implicações reais para a estabilidade do treinamento.

### Post-LN (arquitetura original do paper "Attention Is All You Need")

```
x_meio = LayerNorm(x_entrada + AtencaoMultiHead(x_entrada))
x_saida = LayerNorm(x_meio + MLP(x_meio))
```

Aqui, a normalização é aplicada **depois** da soma residual — o LayerNorm está "dentro" do caminho residual, normalizando a soma `x + F(x)` como um todo.

### Pre-LN (usado pela maioria dos modelos modernos, como GPT-2 em diante)

```
x_meio = x_entrada + AtencaoMultiHead(LayerNorm(x_entrada))
x_saida = x_meio + MLP(LayerNorm(x_meio))
```

Aqui, a normalização é aplicada **antes** de entrar na sub-camada (atenção ou MLP), e a soma residual final usa o `x` **não normalizado** diretamente. O caminho identidade puro (`x_entrada -> x_meio -> x_saida`, sem passar por nenhum LayerNorm) fica preservado de ponta a ponta.

### Por Que Pre-LN é Mais Estável

Lembre do Capítulo 18: a força das conexões residuais vem de existir um caminho de gradiente que é literalmente a identidade, sem nenhuma transformação no meio. No Post-LN, esse caminho **não é mais puro** — o gradiente que flui pelo atalho residual precisa atravessar o LayerNorm a cada bloco, porque o LayerNorm está posicionado *depois* da soma. O LayerNorm em si tem gradiente bem comportado, mas ele já não é mais o termo identidade exato — é `d(LayerNorm(z))/dz`, que envolve a normalização, e isso reintroduz uma fonte de atenuação (ainda que menor que sem residual nenhum) conforme você empilha muitos blocos.

No Pre-LN, a soma residual final (`x_meio = x_entrada + F(...)`) usa `x_entrada` diretamente, sem qualquer transformação — o "gradient highway" fica intacto exatamente como descrito no capítulo anterior. O LayerNorm ainda acontece, mas apenas *dentro* do ramo `F`, nunca no caminho identidade. Isso torna Pre-LN significativamente mais fácil de treinar em redes muito profundas (dezenas de blocos), frequentemente dispensando o "warm-up" cuidadoso de learning rate que Post-LN exige para não divergir logo nos primeiros passos de treinamento.

A contrapartida é que Post-LN, quando treina com sucesso, às vezes converge para soluções ligeiramente melhores (a normalização "no fim" pode regularizar mais agressivamente a magnitude das ativações finais). Mas a robustez de treinamento do Pre-LN é o motivo pelo qual ele se tornou o padrão de fato em modelos modernos — é uma escolha de "mais fácil de treinar de forma confiável" sobre "potencialmente um pouco melhor se tudo correr bem".

| Aspecto | Post-LN | Pre-LN |
|---|---|---|
| Posição do LayerNorm | depois da soma residual | antes da sub-camada, dentro do ramo F |
| Caminho identidade | passa pelo LayerNorm | permanece puro (sem transformação) |
| Estabilidade de treino | requer warm-up cuidadoso | mais robusto, menos sensível a hiperparâmetros |
| Uso | Transformer original (2017) | GPT-2, GPT-3 e a maioria dos modelos modernos |

---

## Experimento: LayerNorm do Zero vs. nn.LayerNorm

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Implementando LayerNorm do Zero")
print("=" * 70)

torch.manual_seed(42)


class LayerNormManual(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: [..., d_model] -- normaliza sobre a última dimensão
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# ========== CONFIGURAÇÃO ==========
batch_size = 2
seq_len = 5
d_model = 8

print(f"\nConfiguração:")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")
print(f"  d_model: {d_model}")

# ========== ENTRADA ==========
print("\n1. ENTRADA")
print("-" * 70)

X = torch.randn(batch_size, seq_len, d_model) * 5 + 3  # média e variância propositalmente longe de 0/1
print(f"X shape: {X.shape}")
print(f"Média por token (deve variar): {X.mean(dim=-1)[0]}")
print(f"Var por token (deve variar): {X.var(dim=-1, unbiased=False)[0]}")

# ========== NOSSA IMPLEMENTAÇÃO ==========
print("\n2. NOSSA LAYERNORM MANUAL")
print("-" * 70)

ln_manual = LayerNormManual(d_model)
out_manual = ln_manual(X)

print(f"output shape: {out_manual.shape}")
print(f"Média por token após LN (deve ser ~0): {out_manual.mean(dim=-1)[0]}")
print(f"Var por token após LN (deve ser ~1, antes de gamma/beta): ")
# nota: como gamma=1, beta=0 na inicialização, a saída aqui é a normalização pura
print(out_manual.var(dim=-1, unbiased=False)[0])

# ========== nn.LayerNorm NATIVO ==========
print("\n3. nn.LayerNorm NATIVO DO PYTORCH")
print("-" * 70)

ln_pytorch = nn.LayerNorm(d_model)
# Copiar os mesmos parâmetros para comparação justa
with torch.no_grad():
    ln_pytorch.weight.copy_(ln_manual.gamma)
    ln_pytorch.bias.copy_(ln_manual.beta)

out_pytorch = ln_pytorch(X)
print(f"output shape: {out_pytorch.shape}")

# ========== COMPARAÇÃO ==========
print("\n4. COMPARAÇÃO: manual vs. nativo")
print("-" * 70)

diferenca_max = (out_manual - out_pytorch).abs().max().item()
print(f"Diferença máxima absoluta: {diferenca_max:.10f}")
print(f"São numericamente iguais? {torch.allclose(out_manual, out_pytorch, atol=1e-6)}")

# ========== VERIFICANDO QUE NORMALIZA POR TOKEN, NÃO POR BATCH ==========
print("\n5. VERIFICANDO QUE A NORMALIZAÇÃO É POR TOKEN (não pelo batch)")
print("-" * 70)

# Mudar completamente outro exemplo do batch NÃO deve afetar a normalização
# do primeiro exemplo -- diferente de BatchNorm!
X2 = X.clone()
X2[1] = torch.randn(seq_len, d_model) * 100 + 1000  # segundo exemplo, valores extremos

out1 = ln_manual(X)
out2 = ln_manual(X2)

print(f"Saída do primeiro exemplo (X), token 0: {out1[0, 0]}")
print(f"Saída do primeiro exemplo (X2, com segundo exemplo alterado), token 0: {out2[0, 0]}")
print(f"São idênticas? {torch.allclose(out1[0], out2[0])}")
print("-> Confirmado: alterar outro exemplo do batch não afeta a normalização deste token,")
print("   porque LayerNorm calcula estatísticas apenas dentro do próprio token.")

# ========== EFEITO DE GAMMA E BETA APRENDIDOS ==========
print("\n6. EFEITO DE GAMMA E BETA DIFERENTES DE 1 E 0")
print("-" * 70)

with torch.no_grad():
    ln_manual.gamma.copy_(torch.linspace(0.5, 2.0, d_model))
    ln_manual.beta.copy_(torch.linspace(-1.0, 1.0, d_model))

out_com_params = ln_manual(X)
print(f"gamma: {ln_manual.gamma}")
print(f"beta: {ln_manual.beta}")
print(f"Saída token 0 (agora NÃO tem mais média 0 / var 1, pois gamma/beta mudaram):")
print(out_com_params[0, 0])
print(f"Média: {out_com_params[0,0].mean().item():.4f}, Var: {out_com_params[0,0].var(unbiased=False).item():.4f}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Experimento 2: Pre-LN vs. Post-LN — Estabilidade em Rede Profunda

```python
import torch
import torch.nn as nn

print("=" * 70)
print("EXPERIMENTO: Comparando Pre-LN e Post-LN em Rede Profunda")
print("=" * 70)

torch.manual_seed(42)

d_model = 32
num_layers = 30
batch_size = 4


class SubRede(nn.Module):
    """Sub-camada F(x) simples, análoga a uma MLP pequena."""
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class BlocoPostLN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.sub = SubRede(d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x + self.sub(x))


class BlocoPreLN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.sub = SubRede(d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.sub(self.ln(x))


def rodar_rede(bloco_cls, num_layers, d_model, batch_size):
    torch.manual_seed(0)
    camadas = nn.ModuleList([bloco_cls(d_model) for _ in range(num_layers)])
    x = torch.randn(batch_size, d_model, requires_grad=True)
    h = x
    for c in camadas:
        h = c(h)
    loss = h.pow(2).mean()
    loss.backward()
    return x.grad.norm().item(), h.norm().item()


print("\n1. COMPARANDO NORMA DE GRADIENTE NA ENTRADA")
print("-" * 70)

grad_norm_post, out_norm_post = rodar_rede(BlocoPostLN, num_layers, d_model, batch_size)
grad_norm_pre, out_norm_pre = rodar_rede(BlocoPreLN, num_layers, d_model, batch_size)

print(f"Post-LN ({num_layers} camadas): norma do gradiente na entrada = {grad_norm_post:.8f}")
print(f"Pre-LN  ({num_layers} camadas): norma do gradiente na entrada = {grad_norm_pre:.8f}")

print(f"\nNorma da ativação de saída:")
print(f"Post-LN: {out_norm_post:.4f}")
print(f"Pre-LN:  {out_norm_pre:.4f}")

print("\nInterpretação:")
print("Pre-LN mantém o caminho residual puro (sem LayerNorm no atalho),")
print("por isso tende a propagar gradientes de forma mais estável em redes")
print("muito profundas, especialmente sem ajustes cuidadosos de warm-up.")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Normalizar na dimensão errada

```python
# ❌ Errado — normaliza ao longo de seq_len (dim=1) em vez de d_model (dim=-1)
mu = x.mean(dim=1, keepdim=True)   # mistura informação entre tokens diferentes!
var = x.var(dim=1, keepdim=True)

# ✓ Certo — normaliza dentro de cada token, ao longo de d_model
mu = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
```

### Erro 2: Usar variância "unbiased" (com correção de Bessel) por engano

```python
# ❌ Errado (ou ao menos inconsistente com a maioria das implementações) —
# var() por padrão usa divisor (n-1), a chamada "variância amostral"
var = x.var(dim=-1, keepdim=True)  # usa unbiased=True por padrão

# ✓ Certo — LayerNorm usa a variância populacional (divisor n), não amostral
var = x.var(dim=-1, keepdim=True, unbiased=False)
```

### Erro 3: Esquecer o epsilon (ou colocá-lo dentro da raiz de forma errada)

```python
# ❌ Errado — divisão pode explodir se var for muito próxima de 0
x_hat = (x - mu) / torch.sqrt(var)

# ✓ Certo — eps garante estabilidade numérica
x_hat = (x - mu) / torch.sqrt(var + eps)
```

### Erro 4: Confundir a posição do LayerNorm (Pre-LN vs Post-LN) ao portar código

```python
# ❌ Errado — mistura os dois estilos sem perceber, quebrando a
# propriedade de caminho residual puro do Pre-LN
def forward(self, x):
    x = self.ln(x)                    # normaliza x
    x = x + self.sublayer(x)          # soma residual usando x JÁ normalizado
    return x
# Isso não é nem Pre-LN nem Post-LN corretamente -- o "x" residual
# não é mais a entrada original do bloco.

# ✓ Certo (Pre-LN) — LayerNorm só afeta a entrada da sub-camada,
# a soma residual usa a entrada original intacta
def forward(self, x):
    return x + self.sublayer(self.ln(x))
```

### Erro 5: Aplicar LayerNorm com gamma/beta compartilhados entre camadas por engano

```python
# ❌ Errado — reutilizar a mesma instância de LayerNorm em blocos diferentes,
# fazendo com que todos os blocos compartilhem os mesmos gamma/beta
ln_compartilhado = nn.LayerNorm(d_model)
blocos = [Bloco(ln_compartilhado) for _ in range(num_layers)]  # bug sutil!

# ✓ Certo — cada bloco tem sua própria instância de LayerNorm com
# parâmetros gamma/beta independentes
blocos = [Bloco(nn.LayerNorm(d_model)) for _ in range(num_layers)]
```

---

## Exercícios

### Exercício 19.1: Calcular LayerNorm Manualmente
Dado `x = [1.0, 3.0, 5.0, 7.0]` (um único token, `d_model=4`), calcule `mu`, `var` (populacional) e `x_hat` manualmente, com `eps=1e-5`.

### Exercício 19.2: Efeito de Gamma e Beta
Usando o `x_hat` do exercício anterior, aplique `gamma = [2.0, 2.0, 2.0, 2.0]` e `beta = [1.0, 1.0, 1.0, 1.0]`. Qual é a saída final `y`? Qual a média e variância de `y`?

### Exercício 19.3: BatchNorm vs. LayerNorm na Prática
Escreva um pequeno experimento comparando `nn.BatchNorm1d` e `nn.LayerNorm` aplicados ao mesmo tensor `[batch=1, features=8]`. O que acontece com BatchNorm quando `batch_size=1` durante `.eval()` versus `.train()`?

### Exercício 19.4: Implementar um Bloco Pre-LN Completo
Combine LayerNorm (deste capítulo), Multi-Head Attention (Capítulo 17) e Residual Connections (Capítulo 18) em um único bloco Pre-LN: `x = x + Atencao(LayerNorm(x))`. Verifique que os shapes se preservam.

### Exercício 19.5: Gradiente Através do LayerNorm
Crie um tensor `x` com `requires_grad=True`, aplique `nn.LayerNorm`, some o resultado e chame `.backward()`. Inspecione `x.grad`. Ele é uniforme ou varia por posição? Por quê?

---

## Gabarito

### Exercício 19.1: Calcular LayerNorm Manualmente
```python
x = [1.0, 3.0, 5.0, 7.0]

mu = (1.0 + 3.0 + 5.0 + 7.0) / 4 = 16.0 / 4 = 4.0

var = [(1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2] / 4
    = [9 + 1 + 1 + 9] / 4
    = 20 / 4 = 5.0

sqrt(var + eps) = sqrt(5.00001) ≈ 2.23608

x_hat[0] = (1.0 - 4.0) / 2.23608 ≈ -1.3416
x_hat[1] = (3.0 - 4.0) / 2.23608 ≈ -0.4472
x_hat[2] = (5.0 - 4.0) / 2.23608 ≈  0.4472
x_hat[3] = (7.0 - 4.0) / 2.23608 ≈  1.3416

x_hat ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
```

### Exercício 19.2: Efeito de Gamma e Beta
```python
x_hat = [-1.3416, -0.4472, 0.4472, 1.3416]
gamma = [2.0, 2.0, 2.0, 2.0]
beta = [1.0, 1.0, 1.0, 1.0]

y[0] = 2.0 * (-1.3416) + 1.0 = -1.6832
y[1] = 2.0 * (-0.4472) + 1.0 =  0.1056
y[2] = 2.0 * (0.4472)  + 1.0 =  1.8944
y[3] = 2.0 * (1.3416)  + 1.0 =  3.6832

y ≈ [-1.6832, 0.1056, 1.8944, 3.6832]

# média de y = 2 * media(x_hat) + 1.0 = 2*0 + 1.0 = 1.0 (= beta, já que gamma escala em torno de 0)
# var de y = gamma^2 * var(x_hat) = 4 * 1.0 = 4.0
```
Verificação em código:
```python
import torch
x_hat = torch.tensor([-1.3416, -0.4472, 0.4472, 1.3416])
gamma = torch.tensor(2.0)
beta = torch.tensor(1.0)
y = gamma * x_hat + beta
print(y)
print(f"média: {y.mean().item():.4f}, var: {y.var(unbiased=False).item():.4f}")
# média: ~1.0, var: ~4.0
```

### Exercício 19.3: BatchNorm vs. LayerNorm na Prática
```python
import torch
import torch.nn as nn

x = torch.randn(1, 8)  # batch_size=1

bn = nn.BatchNorm1d(8)
ln = nn.LayerNorm(8)

bn.train()
try:
    out_train = bn(x)
    print("BatchNorm em .train() com batch_size=1:", out_train)
except Exception as e:
    print("BatchNorm falhou em treino com batch_size=1:", e)

bn.eval()
out_eval = bn(x)  # usa running_mean/running_var acumulados, não trava
print("BatchNorm em .eval():", out_eval)

out_ln = ln(x)
print("LayerNorm (funciona igual em train ou eval):", out_ln)
```
Em `.train()`, `BatchNorm1d` com `batch_size=1` frequentemente lança erro ou produz variância zero (não há variação entre exemplos quando existe apenas um exemplo), tornando a normalização instável ou indefinida. `LayerNorm` não tem esse problema, pois nunca depende de outros exemplos do batch.

### Exercício 19.4: Implementar um Bloco Pre-LN Completo
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)


class BlocoPreLN(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        return x + self.attn(self.ln(x))


d_model, num_heads = 32, 4
bloco = BlocoPreLN(d_model, num_heads)
x = torch.randn(2, 6, d_model)
y = bloco(x)
print(y.shape)  # [2, 6, 32] -- shape preservado, como exigido pela soma residual
```

### Exercício 19.5: Gradiente Através do LayerNorm
```python
import torch
import torch.nn as nn

x = torch.randn(1, 6, requires_grad=True)
ln = nn.LayerNorm(6)
y = ln(x)
loss = y.sum()
loss.backward()

print(x.grad)
```
O gradiente não é uniforme entre as posições — porque `mu` e `var` dependem de todos os elementos de `x` simultaneamente, o gradiente de cada `x_i` depende não só de si mesmo, mas de como ele contribui para a média e a variância compartilhadas de todo o vetor. Elementos mais distantes da média tendem a ter magnitude de gradiente diferente dos elementos próximos à média, refletindo a natureza não-linear da normalização.

---

## Desafios Avançados (Opcionais)

### Fixação 19.1: RMSNorm
Pesquise "RMSNorm" (usado em modelos como LLaMA), uma variante simplificada que remove a subtração da média, normalizando apenas pela raiz quadrada média (RMS): $\hat{x} = x / \text{RMS}(x)$. Implemente-a e compare o custo computacional e a estabilidade de treinamento com LayerNorm padrão.

### Fixação 19.2: LayerNorm sem Gamma/Beta Aprendidos
Treine (ou simule) um bloco Pre-LN com `elementwise_affine=False` em `nn.LayerNorm` (removendo gamma e beta). Qual é o impacto na capacidade expressiva do modelo?

### Fixação 19.3: Sandwich-LN
Algumas arquiteturas aplicam LayerNorm tanto antes quanto depois da sub-camada, dentro do próprio ramo F (não no caminho residual): `x + LN(F(LN(x)))`. Implemente essa variante e compare a estabilidade numérica com Pre-LN puro.

### Fixação 19.4: Visualizando a Distribuição de Ativações por Camada
Em uma rede de 30 blocos Pre-LN, plote (ou imprima estatísticas de) a norma da ativação `x` na saída de cada bloco. Ela cresce de forma controlada, mesmo sem LayerNorm no caminho residual? Compare com uma rede sem LayerNorm nenhum.

### Fixação 19.5: LayerNorm e Precisão Mista (Mixed Precision)
Pesquise por que implementações de LayerNorm frequentemente forçam o cálculo em float32 mesmo quando o resto do modelo roda em float16/bfloat16 (mixed precision training). Que problema numérico isso evita?

---

## Resumo

- **Por que normalizar**: sem controle, ativações em redes profundas podem crescer ou encolher descontroladamente, causando instabilidade numérica e loss = nan.
- **BatchNorm vs. LayerNorm**: BatchNorm normaliza através dos exemplos do batch (por feature); LayerNorm normaliza através das features de cada token individual (por exemplo) — por isso é robusto a batch pequeno e sequências de comprimento variável.
- **Fórmula**: $\hat{x} = (x - \mu) / \sqrt{\sigma^2 + \epsilon}$, seguido de $y = \gamma \hat{x} + \beta$, com $\gamma$ e $\beta$ aprendidos por dimensão de feature.
- **Pre-LN vs. Post-LN**: Pre-LN normaliza a entrada da sub-camada, mantendo o caminho residual puro (identidade intacta); Post-LN normaliza depois da soma residual, o que reintroduz atenuação no caminho de gradiente.
- **Escolha moderna**: modelos como GPT-2 em diante usam Pre-LN por sua maior estabilidade em redes profundas, mesmo que Post-LN às vezes converja para soluções ligeiramente melhores quando o treino não diverge.
- **Independência estatística**: LayerNorm calcula estatísticas por token, tornando seu comportamento idêntico entre treino e inferência — sem médias móveis nem discrepâncias de modo.

Próximo capítulo: **MLP e Feed-Forward Network** — a segunda sub-camada de todo bloco Transformer, que expande e contrai a dimensão para processar cada token individualmente.

---

**Próximo**: [Capítulo 20: MLP e Feed-Forward Network](20_mlp_feedforward.md)
