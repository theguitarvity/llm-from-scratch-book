# Capítulo 13: Scaling por sqrt(d_k) e Softmax

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Provar por que a variância dos scores brutos cresce linearmente com a dimensão $d_k$
2. Explicar matematicamente por que dividir por $\sqrt{d_k}$ estabiliza essa variância
3. Descrever o que acontece com o softmax quando seus inputs são muito grandes (saturação e gradientes próximos de zero)
4. Revisar a fórmula do softmax e sua implementação numericamente estável (max-subtraction)
5. Demonstrar experimentalmente a diferença de comportamento com e sem scaling

---

## Por Que Isso Importa

No Capítulo 12 você viu que os scores brutos $QK^T$ podem ser qualquer número real — sem limite superior nem inferior. Isso pode parecer um detalhe técnico menor, mas é uma das decisões de design mais citadas do paper original do Transformer ("Attention Is All You Need"), e ignorá-la é uma das formas mais comuns de um Transformer implementado do zero simplesmente **não treinar**.

Aqui está o problema em termos concretos: imagine que você está debugando um modelo que, depois de algumas iterações de treinamento, para de aprender — o loss fica estagnado, os gradientes ficam praticamente zero, e a matriz de atenção colapsa para algo quase determinístico (um peso de 0.999 em um token, quase zero nos demais), mesmo em camadas que deveriam distribuir atenção mais suavemente. Se você inspecionar os scores brutos antes do softmax e perceber que eles estão na casa de centenas ou milhares, você encontrou a causa raiz: o softmax está **saturado**.

A intuição por trás disso é semelhante a girar um botão de volume que só reage a mudanças em uma faixa estreita. Se o volume já está no talo (softmax saturado, produzindo distribuição quase one-hot), girar um pouco mais o botão praticamente não muda nada — e é exatamente isso que acontece com os gradientes: eles ficam próximos de zero, e o otimizador não consegue mais ajustar os pesos de forma significativa. O scaling por $\sqrt{d_k}$ existe precisamente para manter os scores numa faixa em que o softmax continua "sensível" — nem saturado, nem uniforme demais.

---

## Por Que a Variância dos Scores Cresce com d

### A hipótese: componentes independentes e de variância unitária

Suponha que cada componente de $Q$ e $K$ é uma variável aleatória independente, com média 0 e variância 1 — uma suposição razoável logo após inicialização (pesos inicializados com Xavier/He, e embeddings normalizados). Ou seja, para cada dimensão $i$:

$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1, \quad \mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

O score é o dot product:

$$s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

### Calculando a variância de s

Cada termo $q_i k_i$ é o produto de duas variáveis independentes de média 0. A variância desse produto é:

$$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2$$

Como $q_i$ e $k_i$ são independentes e ambos têm média 0, $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$. E:

$$\mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = \text{Var}(q_i) \cdot \text{Var}(k_i) = 1 \cdot 1 = 1$$

Logo, $\text{Var}(q_i k_i) = 1$ para cada termo. Como a soma de $d_k$ variáveis **independentes** tem variância igual à soma das variâncias individuais:

$$\text{Var}(s) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \cdot 1 = d_k$$

**Este é o resultado central**: a variância do score bruto cresce **linearmente** com $d_k$, e portanto o desvio padrão cresce com $\sqrt{d_k}$.

$$\text{Var}(s) = d_k \qquad \Rightarrow \qquad \text{std}(s) = \sqrt{d_k}$$

### O que isso significa em números concretos

- Com $d_k = 64$ (comum em uma cabeça de atenção individual): $\text{std}(s) = \sqrt{64} = 8$. Scores tipicamente na faixa de $-24$ a $+24$ (3 desvios padrão).
- Com $d_k = 512$ (um modelo sem multi-head, ou d_model inteiro): $\text{std}(s) = \sqrt{512} \approx 22.6$. Scores podem facilmente passar de $\pm 60$.

Esses números, alimentados diretamente num softmax, produzem exatamente o problema de saturação que descrevemos. A solução é simples: dividir por $\sqrt{d_k}$.

### Por que dividir por sqrt(d_k) resolve

Definindo o score escalado $s' = s / \sqrt{d_k}$:

$$\text{Var}(s') = \text{Var}\left(\frac{s}{\sqrt{d_k}}\right) = \frac{1}{d_k}\text{Var}(s) = \frac{d_k}{d_k} = 1$$

Depois do scaling, a variância volta a ser **1, independente de $d_k$**. Isso significa que, não importa quão grande seja a dimensão das cabeças de atenção, os scores escalados ficam numa faixa estatisticamente estável — tipicamente entre $-3$ e $+3$. É exatamente essa faixa que mantém o softmax "vivo" e sensível a pequenas diferenças relativas entre scores.

$$\text{scores\_escalados} = \frac{QK^T}{\sqrt{d_k}}$$

---

## Softmax: Revisão e Estabilidade Numérica

### A fórmula

Dado um vetor de scores $z = [z_1, z_2, \ldots, z_n]$, o softmax converte em uma distribuição de probabilidade:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

Propriedades:

1. Cada saída está em $(0, 1)$.
2. A soma de todas as saídas é exatamente 1: $\sum_i \text{softmax}(z)_i = 1$.
3. É monotônica: se $z_i > z_j$, então $\text{softmax}(z)_i > \text{softmax}(z)_j$.
4. É invariante a deslocamento: $\text{softmax}(z + c) = \text{softmax}(z)$ para qualquer constante $c$.

### Por que softmax satura com inputs grandes

Considere dois scores, $z_1 = 20$ e $z_2 = 1$ (diferença de 19, típica de scores não escalados com $d_k$ grande):

$$e^{20} \approx 485.165.195, \qquad e^{1} \approx 2.72$$

$$\text{softmax}(z)_1 = \frac{485.165.195}{485.165.195 + 2.72} \approx 0.9999999944$$

O resultado é essencialmente $[1.0, 0.0]$ — uma distribuição one-hot, mesmo que $z_2$ não fosse tão pequeno assim em termos absolutos. Compare com scores escalados, digamos $z_1 = 2.0$ e $z_2 = 0.1$ (diferença de 1.9, típica após dividir por $\sqrt{d_k}$):

$$e^{2.0} \approx 7.389, \qquad e^{0.1} \approx 1.105$$

$$\text{softmax}(z)_1 = \frac{7.389}{7.389 + 1.105} \approx 0.870$$

Uma distribuição de 87%/13% — ainda decidida, mas não colapsada, e com espaço para o gradiente distinguir entre alternativas.

### Por que a saturação mata o gradiente

A derivada do softmax em relação ao score de entrada, para a própria posição $i$, é:

$$\frac{\partial \text{softmax}(z)_i}{\partial z_i} = \text{softmax}(z)_i \cdot (1 - \text{softmax}(z)_i)$$

Se $\text{softmax}(z)_i \approx 1$ (saturado), então $(1 - \text{softmax}(z)_i) \approx 0$, e o produto inteiro vai a zero. Ou seja, **quando a probabilidade já está perto de 1, o gradiente local do softmax desaparece** — o clássico problema de "vanishing gradient" localizado, aqui causado não pela profundidade da rede, mas pela escala dos inputs.

### Estabilidade numérica: max-subtraction

Há ainda um problema puramente numérico: $e^{z}$ pode facilmente **overflow** em float32 quando $z$ é grande. Por exemplo, $e^{100} \approx 2.7 \times 10^{43}$, que ainda cabe em float32, mas $e^{1000}$ já estoura (`inf`). Usamos a propriedade de invariância a deslocamento para subtrair o máximo antes de exponenciar:

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

Isso não muda o resultado matemático (porque softmax é invariante a deslocamento), mas garante que o maior expoente calculado seja sempre $e^0 = 1$, eliminando o risco de overflow. O `torch.softmax` (e `F.softmax`) já implementa essa técnica internamente — mas é essencial entender por que ela existe, porque você a reimplementará manualmente em experimentos de baixo nível.

```python
# Implementação didática de softmax numericamente estável
def softmax_manual(z, dim=-1):
    z_max = z.max(dim=dim, keepdim=True).values
    z_shifted = z - z_max          # evita overflow, resultado idêntico
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)
```

---

## Exemplo Numérico Manual

Vamos comparar softmax com e sem scaling usando os mesmos scores brutos.

### Scores brutos (d_k = 64, valores típicos e exagerados para ilustrar)

```
scores_brutos = [12.0, 8.0, -4.0]
```

### Sem scaling

```
exp(12.0) = 162754.79
exp(8.0)  = 2980.96
exp(-4.0) = 0.0183

soma = 165735.77

softmax = [162754.79/165735.77, 2980.96/165735.77, 0.0183/165735.77]
        = [0.9820, 0.0180, 0.0000001]
```

Distribuição quase totalmente concentrada em um único token.

### Com scaling (dividindo por sqrt(64) = 8)

```
scores_escalados = [12.0/8, 8.0/8, -4.0/8] = [1.5, 1.0, -0.5]

exp(1.5)  = 4.4817
exp(1.0)  = 2.7183
exp(-0.5) = 0.6065

soma = 7.8065

softmax = [4.4817/7.8065, 2.7183/7.8065, 0.6065/7.8065]
        = [0.5741, 0.3482, 0.0777]
```

Distribuição muito mais equilibrada — 57%/35%/8% em vez de 98%/2%/0.00001%. O token dominante ainda recebe mais peso (a ordem relativa não muda, pois softmax é monotônico), mas agora há espaço real para os outros tokens contribuírem e para os gradientes fluírem por todos os três caminhos.

---

## Experimento: Saturação do Softmax Sem Scaling

```python
import torch

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Scaling por sqrt(d_k) e Saturação de Softmax")
print("=" * 70)

# ========== VERIFICANDO A VARIÂNCIA CRESCE COM d ==========
print("\n1. VARIÂNCIA DOS SCORES CRESCE COM d_k")
print("-" * 70)

n_amostras = 20000
for d_k in [8, 64, 256, 1024]:
    q = torch.randn(n_amostras, d_k)
    k = torch.randn(n_amostras, d_k)
    scores = (q * k).sum(dim=-1)  # dot product por linha
    var_empirica = scores.var().item()
    print(f"d_k={d_k:>5}  ->  Var(scores) empírica = {var_empirica:7.2f}  "
          f"(esperado: {d_k})")

# ========== SATURAÇÃO SEM SCALING ==========
print("\n2. SOFTMAX SEM SCALING (d_k grande)")
print("-" * 70)

d_k = 256
seq_len = 4
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)

scores_brutos = Q @ K.T
print(f"scores brutos (linha 0): {scores_brutos[0].detach().numpy().round(2)}")
print(f"std dos scores brutos: {scores_brutos.std().item():.2f}")

attn_sem_scaling = torch.softmax(scores_brutos, dim=-1)
print(f"softmax SEM scaling (linha 0): {attn_sem_scaling[0].detach().numpy().round(6)}")
print(f"Entropia (bits) linha 0: {(-attn_sem_scaling[0] * torch.log2(attn_sem_scaling[0] + 1e-12)).sum().item():.4f}")

# ========== COM SCALING ==========
print("\n3. SOFTMAX COM SCALING POR sqrt(d_k)")
print("-" * 70)

scores_escalados = scores_brutos / (d_k ** 0.5)
print(f"scores escalados (linha 0): {scores_escalados[0].detach().numpy().round(4)}")
print(f"std dos scores escalados: {scores_escalados.std().item():.4f}")

attn_com_scaling = torch.softmax(scores_escalados, dim=-1)
print(f"softmax COM scaling (linha 0): {attn_com_scaling[0].detach().numpy().round(4)}")
print(f"Entropia (bits) linha 0: {(-attn_com_scaling[0] * torch.log2(attn_com_scaling[0] + 1e-12)).sum().item():.4f}")

# ========== GRADIENTES: SATURADO VS SAUDÁVEL ==========
print("\n4. COMPARANDO GRADIENTES (saturado vs escalado)")
print("-" * 70)

scores_saturado = torch.tensor([12.0, 8.0, -4.0], requires_grad=True)
probs_saturado = torch.softmax(scores_saturado, dim=-1)
loss_saturado = probs_saturado[0]  # queremos maximizar prob do índice 0
loss_saturado.backward()
print(f"scores saturados: {scores_saturado.detach().numpy()}")
print(f"probs saturados:  {probs_saturado.detach().numpy().round(6)}")
print(f"gradiente (d_probs[0]/d_scores): {scores_saturado.grad.numpy().round(8)}")

scores_saudavel = torch.tensor([1.5, 1.0, -0.5], requires_grad=True)
probs_saudavel = torch.softmax(scores_saudavel, dim=-1)
loss_saudavel = probs_saudavel[0]
loss_saudavel.backward()
print(f"\nscores escalados: {scores_saudavel.detach().numpy()}")
print(f"probs escalados:  {probs_saudavel.detach().numpy().round(4)}")
print(f"gradiente (d_probs[0]/d_scores): {scores_saudavel.grad.numpy().round(6)}")

print("\nRepare: o gradiente do caso saturado é ORDENS DE MAGNITUDE menor.")

# ========== SOFTMAX MANUAL NUMERICAMENTE ESTÁVEL ==========
print("\n5. SOFTMAX MANUAL (MAX-SUBTRACTION)")
print("-" * 70)

def softmax_manual(z, dim=-1):
    z_max = z.max(dim=dim, keepdim=True).values
    z_shifted = z - z_max
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

z_grande = torch.tensor([1000.0, 999.0, 998.0])
try:
    resultado_ingenuo = torch.exp(z_grande) / torch.exp(z_grande).sum()
    print(f"softmax ingênuo (sem max-subtraction): {resultado_ingenuo}")
except Exception as e:
    print(f"softmax ingênuo falhou: {e}")

resultado_estavel = softmax_manual(z_grande)
resultado_pytorch = torch.softmax(z_grande, dim=-1)
print(f"softmax manual estável: {resultado_estavel}")
print(f"torch.softmax (referência): {resultado_pytorch}")
print(f"Iguais? {torch.allclose(resultado_estavel, resultado_pytorch)}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Dividir por d_k em vez de sqrt(d_k)

```python
# Errado — dividir pela dimensão inteira, não pela raiz
scores_escalados = scores / d_k  # Super-corrige, achata demais a distribuição

# Certo
scores_escalados = scores / (d_k ** 0.5)
```

### Erro 2: Escalar depois do softmax

```python
# Errado — escalar as probabilidades já normalizadas não faz sentido
attn = torch.softmax(scores, dim=-1)
attn_escalado = attn / (d_k ** 0.5)  # Quebra a propriedade de somar 1!

# Certo — escalar ANTES do softmax
attn = torch.softmax(scores / (d_k ** 0.5), dim=-1)
```

### Erro 3: Usar d_model em vez de d_k quando há multi-head

```python
# Errado — usar a dimensão total do modelo em vez da dimensão por cabeça
d_model = 512
num_heads = 8
scores_escalados = scores / (d_model ** 0.5)  # Escala errada!

# Certo — usar d_k, a dimensão de CADA cabeça (d_model / num_heads)
d_k = d_model // num_heads  # 64
scores_escalados = scores / (d_k ** 0.5)
```

---

## Exercícios

### Exercício 13.1: Verificando a Fórmula da Variância
Gere 10.000 pares de vetores aleatórios com `d_k = 128` (`torch.randn`). Calcule o dot product de cada par e confirme empiricamente que a variância é aproximadamente 128.

### Exercício 13.2: Softmax Manual
Implemente `softmax_manual(z)` do zero (sem usar `torch.softmax`), incluindo max-subtraction, e valide contra `torch.softmax` em pelo menos 3 vetores de teste diferentes.

### Exercício 13.3: Entropia como Medida de Saturação
Escreva uma função que calcula a entropia (em bits) de uma distribuição de softmax. Compare a entropia de `[0.98, 0.01, 0.01]` com a de `[0.4, 0.35, 0.25]`. Qual é maior? O que isso significa em termos de "quão espalhada" está a atenção?

### Exercício 13.4: Encontrando o d_k Correto
Dado um `d_model = 512` e `num_heads = 8`, calcule qual deve ser o fator de scaling correto para os scores de atenção de cada cabeça.

### Exercício 13.5: Simulando o Efeito do Scaling no Gradiente
Para os scores `[20.0, 15.0, 5.0]`, calcule o gradiente do softmax em relação ao primeiro score, com e sem dividir por `sqrt(64)` antes. Compare as magnitudes.

---

## Gabarito

### Exercício 13.1: Verificando a Fórmula da Variância
```python
import torch
torch.manual_seed(0)

d_k = 128
n = 10000
q = torch.randn(n, d_k)
k = torch.randn(n, d_k)
scores = (q * k).sum(dim=-1)
print(f"Variância empírica: {scores.var().item():.2f}  (esperado ~{d_k})")
```

### Exercício 13.2: Softmax Manual
```python
import torch

def softmax_manual(z, dim=-1):
    z_max = z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z - z_max)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

for teste in [torch.tensor([1.0, 2.0, 3.0]),
              torch.tensor([100.0, 100.0, 100.0]),
              torch.tensor([-5.0, 0.0, 5.0])]:
    manual = softmax_manual(teste)
    referencia = torch.softmax(teste, dim=-1)
    print(f"manual={manual}, referencia={referencia}, iguais={torch.allclose(manual, referencia)}")
```

### Exercício 13.3: Entropia como Medida de Saturação
```python
import torch

def entropia_bits(probs):
    return (-probs * torch.log2(probs + 1e-12)).sum().item()

p1 = torch.tensor([0.98, 0.01, 0.01])
p2 = torch.tensor([0.4, 0.35, 0.25])

print(f"Entropia p1 (concentrada): {entropia_bits(p1):.4f} bits")
print(f"Entropia p2 (espalhada):   {entropia_bits(p2):.4f} bits")
# p2 tem entropia maior -> distribuição mais "espalhada"/incerta
```

### Exercício 13.4: Encontrando o d_k Correto
```python
d_model = 512
num_heads = 8
d_k = d_model // num_heads  # 64
fator_scaling = d_k ** 0.5  # 8.0
print(f"d_k = {d_k}, fator de scaling = {fator_scaling}")
```

### Exercício 13.5: Simulando o Efeito do Scaling no Gradiente
```python
import torch

scores_brutos = torch.tensor([20.0, 15.0, 5.0], requires_grad=True)
probs = torch.softmax(scores_brutos, dim=-1)
probs[0].backward()
print(f"Sem scaling - grad: {scores_brutos.grad}")

scores_escalados = (torch.tensor([20.0, 15.0, 5.0]) / (64 ** 0.5)).requires_grad_()
probs2 = torch.softmax(scores_escalados, dim=-1)
probs2[0].backward()
print(f"Com scaling - grad: {scores_escalados.grad}")
# O gradiente com scaling tende a ser maior em magnitude relativa
```

---

## Desafios Avançados (Opcionais)

### Fixação 13.1: Prova Formal com Covariância
Refaça a derivação da variância de $s = q \cdot k$ sem assumir que $q_i$ e $k_i$ são exatamente independentes de outras dimensões — mostre onde a independência é usada e o que aconteceria se houvesse correlação entre dimensões.

### Fixação 13.2: Scaling Alternativo
Teste empiricamente um scaling por $d_k$ (em vez de $\sqrt{d_k}$) e por $\sqrt[4]{d_k}$. Compare a variância resultante dos scores e a entropia média da distribuição de atenção em cada caso.

### Fixação 13.3: Temperature como Generalização
O scaling por $\sqrt{d_k}$ é matematicamente equivalente a usar uma "temperatura" $T = \sqrt{d_k}$ no softmax ($\text{softmax}(z/T)$). Implemente uma função `softmax_com_temperatura(z, T)` e explore como $T \to 0$ e $T \to \infty$ afetam a distribuição.

### Fixação 13.4: Impacto no Treinamento Real
Usando um mini-loop de treinamento (pode ser um problema de regressão simples com uma camada de atenção), compare a curva de loss ao longo de 100 passos com e sem scaling. Documente se o modelo sem scaling treina mais devagar ou trava.

### Fixação 13.5: Scaling Dependente dos Dados
Em vez de um fator fixo $\sqrt{d_k}$, implemente um scaling aprendido (um escalar `nn.Parameter` inicializado em $1/\sqrt{d_k}$). Isso é usado em algumas variantes de atenção — investigue se o modelo aprende a ajustar esse valor durante o treinamento.

---

## Resumo

- **Variância cresce com d_k**: $\text{Var}(QK^T) = d_k$ quando Q e K têm componentes de variância unitária e independentes
- **Scaling normaliza a variância**: dividir por $\sqrt{d_k}$ traz a variância de volta para 1, independente da dimensão
- **Softmax satura com inputs grandes**: diferenças grandes entre scores viram distribuições quase one-hot
- **Saturação mata gradientes**: $\text{softmax}_i(1-\text{softmax}_i) \to 0$ quando $\text{softmax}_i \to 1$
- **Max-subtraction garante estabilidade numérica**: subtrair o máximo antes de exponenciar evita overflow sem mudar o resultado
- **Use d_k, não d_model**: em multi-head attention, o scaling deve usar a dimensão de cada cabeça, não a dimensão total do modelo

Próximo capítulo: **Weighted Sum de V** — como os pesos de atenção normalizados combinam os Values em uma representação contextualizada.

---

**Próximo**: [Capítulo 14: Weighted Sum de V](14_weighted_sum.md)
