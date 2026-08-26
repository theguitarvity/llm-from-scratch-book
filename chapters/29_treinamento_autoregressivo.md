# Capítulo 29: Treinamento Autoregressivo

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o que significa "autoregressivo" e como isso molda a preparação dos dados de treino
2. Construir pares (input, target) a partir de texto contínuo via shift-by-one
3. Explicar teacher forcing e por que ele viabiliza treinamento paralelo e estável
4. Implementar o loop de treinamento completo: forward, loss, zero_grad, backward, clip, step
5. Usar overfitting em um exemplo pequeno como teste de sanidade da implementação

---

## Por Que Isso Importa

Todas as peças dos capítulos anteriores — atenção, MLP, logits, cross-entropy, backprop, otimizadores — existem para servir a um único loop, repetido milhões de vezes: mostrar um pedaço de texto ao modelo, pedir para ele prever a próxima palavra, medir o erro, e ajustar os pesos um pouquinho na direção que reduz esse erro. Esse loop, aplicado repetidamente sobre bilhões de tokens, é literalmente tudo o que existe por trás de "treinar um LLM". Não há mágica adicional — é este loop, com boa engenharia em volta.

Mas antes de rodar esse loop em escala, existe um teste de sanidade que separa quem realmente entende a implementação de quem só copiou código de algum lugar: pegar um único batch pequeno de dados e tentar fazer o modelo memorizá-lo perfeitamente, através de overfitting proposital. Se seu modelo, otimizador e loop de treinamento estão implementados corretamente, ele **deve** conseguir reduzir a loss daquele batch pequeno a praticamente zero em algumas centenas de steps. Se isso não acontece — se a loss estagna, oscila, ou não desce de forma alguma — existe um bug em algum lugar (shape errado, gradiente não fluindo, learning rate absurdo, dados montados incorretamente), e é muito mais barato descobrir isso em um experimento de 30 segundos do que depois de 6 horas de treino "de verdade" em um dataset gigante que simplesmente não converge.

Este capítulo também esclarece um ponto que confunde muita gente que aprende Transformers pela primeira vez: durante a geração de texto (que você vai ver no Capítulo 31), o modelo usa sua própria predição anterior como entrada para o próximo passo. Mas durante o *treinamento*, isso não acontece — usamos sempre o token real do texto, nunca o que o modelo teria predito. Essa técnica, chamada teacher forcing, é o que torna possível treinar um Transformer inteiro em paralelo, de uma vez só, em vez de token por token sequencialmente — e entender por que isso funciona (e quando não funciona perfeitamente) é fundamental para entender o comportamento do modelo durante geração.

---

## O Que Significa "Autoregressivo"

Um modelo de linguagem autoregressivo modela a probabilidade de uma sequência de tokens $x_1, x_2, \dots, x_T$ decompondo-a via regra da cadeia de probabilidade:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1})$$

Ou seja: a probabilidade da sequência inteira é o produto das probabilidades de cada token, condicionadas em **todos os tokens anteriores**. "Auto-regressivo" vem exatamente disso — o modelo regride (prediz) sobre sua própria saída anterior, token após token.

Isso é o que a máscara causal (Capítulo 15) impõe estruturalmente: na posição $t$, o modelo só pode "ver" (via atenção) os tokens em $1, \dots, t$, nunca os futuros. Isso garante que o modelo aprendido seja utilizável para geração — em inferência, os tokens futuros simplesmente não existem ainda.

---

## Preparando Pares (Input, Target): Shift-by-One

Dado um texto contínuo tokenizado, como transformamos isso em exemplos de treino? A resposta é surpreendentemente simples: **o input é a sequência, o target é a mesma sequência deslocada em uma posição**.

```
Texto tokenizado: [10, 25, 7, 91, 3, 44]

input  = [10, 25, 7, 91, 3]     (sequência[:-1])
target = [25, 7, 91, 3, 44]     (sequência[1:])
```

Ou seja, `target[i] = input[i+1]` para cada posição — o token que deveria vir a seguir. Isso significa que, na posição $i$ do input, o modelo vê os tokens $x_0, \dots, x_i$ (via atenção causal) e é treinado para prever $x_{i+1}$, que é exatamente `target[i]`.

**Ponto crucial**: com uma única sequência de comprimento $T$, obtemos $T-1$ exemplos de predição de próximo-token *simultaneamente*, um por posição. Não precisamos rodar o modelo $T-1$ vezes — o forward pass processa a sequência inteira de uma vez, e a máscara causal garante que cada posição só "trapaceie" olhando para o passado, nunca o futuro. Essa é uma das razões pelas quais Transformers treinam tão eficientemente comparados a RNNs.

```python
import torch

sequencia = torch.tensor([10, 25, 7, 91, 3, 44])

input_ids  = sequencia[:-1]  # [10, 25, 7, 91, 3]
target_ids = sequencia[1:]   # [25, 7, 91, 3, 44]

print(input_ids)
print(target_ids)
```

Em um batch com `batch_size` sequências de comprimento `seq_len`, os shapes ficam:

$$\text{input\_ids} \in \mathbb{R}^{[batch, seq\_len]} \qquad \text{target\_ids} \in \mathbb{R}^{[batch, seq\_len]}$$

E o forward do modelo produz `logits` de shape `[batch, seq_len, vocab_size]`, que comparamos contra `target_ids` via cross-entropy, exatamente como no Capítulo 26.

---

## Teacher Forcing

Durante o treinamento, na posição $i$, o modelo recebe como entrada o token **real** $x_i$ do texto — não o token que ele mesmo teria predito na posição anterior. Isso é chamado **teacher forcing**: o "professor" (o texto de treino real) sempre corrige o modelo, fornecendo a entrada correta a cada posição, independentemente do que o modelo teria produzido sozinho.

Por que isso importa:

**1. Paralelização.** Se o modelo tivesse que usar sua própria predição da posição $i$ como entrada da posição $i+1$ (como acontece durante geração — veja Capítulo 32), o treinamento seria inerentemente sequencial: seria preciso calcular a posição 0, depois a 1, depois a 2, e assim por diante, um token de cada vez. Com teacher forcing, como todas as entradas já são conhecidas de antemão (vêm do texto real), o forward pass inteiro da sequência acontece em uma única passada matricial paralela.

**2. Estabilidade.** Se o modelo ainda está no início do treinamento e comete um erro na posição 3, sem teacher forcing esse erro se propagaria para as posições 4, 5, 6... (o modelo erraria cada vez mais, condicionado em erros anteriores — um fenômeno chamado "exposure bias" ou acúmulo de erro). Com teacher forcing, cada posição recebe o contexto correto independentemente do desempenho do modelo nas posições anteriores, tornando os gradientes muito mais estáveis e o sinal de treino mais limpo.

**A ressalva importante**: teacher forcing cria uma discrepância entre treino e inferência — durante geração real, o modelo *precisa* usar suas próprias predições anteriores (porque o texto futuro genuinamente não existe ainda), uma condição que ele nunca viu durante o treino. Esse descompasso (exposure bias) é uma das razões pelas quais a qualidade de geração pode degradar ao longo de sequências muito longas, mesmo que a loss de treino esteja baixa.

---

## O Loop de Treinamento Completo

Juntando tudo dos capítulos anteriores, o loop de treinamento padrão segue esta sequência fixa de operações a cada batch:

```python
for batch in dataloader:
    optimizer.zero_grad()                    # 1. Zerar gradientes acumulados
    input_ids, target_ids = batch
    logits = model(input_ids)                 # 2. Forward pass
    loss = F.cross_entropy(                    # 3. Calcular loss
        logits.view(-1, vocab_size),
        target_ids.view(-1)
    )
    loss.backward()                            # 4. Backward pass (calcula gradientes)
    torch.nn.utils.clip_grad_norm_(             # 5. Gradient clipping
        model.parameters(), max_norm=1.0
    )
    optimizer.step()                           # 6. Atualizar parâmetros
```

Cada uma dessas seis etapas corresponde a um capítulo anterior deste livro:

1. `zero_grad()` — Capítulo 27 (gradientes se acumulam por padrão, é preciso zerar)
2. `model(input_ids)` — todo o Transformer, Capítulos 10-23
3. `cross_entropy(...)` — Capítulo 26
4. `loss.backward()` — Capítulo 27
5. `clip_grad_norm_(...)` — Capítulo 27
6. `optimizer.step()` — Capítulo 28

A ordem importa: gradient clipping precisa acontecer **depois** de `backward()` (que é quando os gradientes existem) e **antes** de `step()` (que é quando eles são usados para atualizar os pesos).

### Batching com DataLoader

Para treinar eficientemente, agrupamos múltiplas sequências em um batch, aproveitando paralelismo de GPU/CPU:

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids

dataset = TextDataset(token_ids=meus_tokens, seq_len=32)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

Cada item do dataset já vem no formato shift-by-one; o `DataLoader` cuida de empilhar múltiplos itens em um único tensor de batch e embaralhar a ordem a cada epoch.

### Monitorando a Loss

```python
for epoch in range(n_epochs):
    epoch_losses = []
    for step, (input_ids, target_ids) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())
        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    print(f"Epoch {epoch} | Loss média: {sum(epoch_losses)/len(epoch_losses):.4f}")
```

---

## Learning Rate Warmup e Scheduling (Introdução)

Iniciar o treinamento com o learning rate final desejado (ex: `6e-4`) frequentemente causa instabilidade nos primeiros steps: os pesos ainda estão em sua inicialização aleatória, os gradientes iniciais podem ser grandes e ruidosos, e um passo grande logo de cara pode empurrar o modelo para uma região ruim do espaço de parâmetros da qual é difícil se recuperar.

A solução padrão é **warmup**: começar com um learning rate muito baixo (próximo de zero) e aumentá-lo linearmente até o valor alvo ao longo dos primeiros milhares de steps, e só então aplicar um schedule de decaimento (cosine decay é o mais comum) pelo resto do treino.

```python
def get_lr(step, warmup_steps, max_lr, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # decaimento cosseno após o warmup
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
    return 0.5 * max_lr * (1 + math.cos(math.pi * progress))
```

Esse padrão de warmup + cosine decay é exatamente o que será usado no `CONFIG` do projeto completo no Capítulo 33 — por ora, o importante é entender a motivação: **warmup evita instabilidade inicial**, e **decay ao longo do treino permite refinamento mais preciso nos estágios finais**, quando passos grandes já não são mais desejáveis.

---

## Overfitting em um Exemplo Pequeno: O Teste de Sanidade Essencial

Antes de treinar em um dataset real, sempre vale a pena rodar este teste: pegar **um único batch pequeno** e treinar repetidamente só nele, sem nenhum outro dado, por várias centenas de steps. Se a implementação está correta, a loss deve cair para próximo de zero — o modelo tem capacidade mais que suficiente para simplesmente memorizar um punhado de exemplos.

Se isso **não** acontece — a loss estagna em um valor alto, oscila sem convergir, ou vira `NaN` — isso é evidência forte de um bug real na implementação (não um problema de "o modelo precisa de mais dados" ou "mais treino", já que estamos deliberadamente testando com um único batch). É o teste de sanidade mais barato e mais informativo que existe antes de investir tempo de treino de verdade.

---

## Experimento: Loop de Treinamento e Teste de Overfitting

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Treinamento Autoregressivo e Teste de Overfitting")
print("=" * 70)

# ========== 1. MODELO MINIMALISTA (simplificado, para o experimento) ==========
print("\n1. MODELO SIMPLIFICADO")
print("-" * 70)

class MiniLM(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        h = self.ln1(h + attn_out)
        h = self.ln2(h + self.mlp(h))

        logits = self.head(h)
        return logits

vocab_size = 20
d_model = 16
seq_len = 6

model = MiniLM(vocab_size, d_model, seq_len)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modelo criado com {n_params} parâmetros")

# ========== 2. PREPARANDO DADOS (SHIFT-BY-ONE) ==========
print("\n2. PREPARANDO INPUT/TARGET COM SHIFT-BY-ONE")
print("-" * 70)

sequencia_completa = torch.tensor([3, 7, 1, 9, 12, 5, 18])
input_ids = sequencia_completa[:-1].unsqueeze(0)   # [1, 6]
target_ids = sequencia_completa[1:].unsqueeze(0)   # [1, 6]

print(f"Sequência completa: {sequencia_completa.tolist()}")
print(f"input_ids:  {input_ids.tolist()}  shape={input_ids.shape}")
print(f"target_ids: {target_ids.tolist()}  shape={target_ids.shape}")

# ========== 3. UM STEP DE TREINO PASSO A PASSO ==========
print("\n3. UM STEP DE TREINO (forward -> loss -> backward -> clip -> step)")
print("-" * 70)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

optimizer.zero_grad()
logits = model(input_ids)
print(f"logits shape: {logits.shape}")  # [1, 6, vocab_size]

loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
print(f"loss antes do primeiro step: {loss.item():.4f}")

loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"norma do gradiente (antes do clip): {grad_norm.item():.4f}")

optimizer.step()
print("Step do otimizador aplicado.")

# ========== 4. TESTE DE OVERFITTING (SANITY CHECK) ==========
print("\n4. TESTE DE OVERFITTING EM UM ÚNICO BATCH")
print("-" * 70)
print("Se a implementação está correta, a loss deve cair para perto de zero.")

losses = []
n_steps = 300
for step in range(n_steps):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    losses.append(loss.item())

print(f"\nLoss no step 0:   {losses[0]:.4f}")
print(f"Loss no step 50:  {losses[50]:.4f}")
print(f"Loss no step 150: {losses[150]:.4f}")
print(f"Loss no step 299: {losses[-1]:.4f}")

# ========== 5. VERIFICANDO SE O MODELO MEMORIZOU ==========
print("\n5. VERIFICANDO PREVISÕES DEPOIS DO OVERFITTING")
print("-" * 70)

with torch.no_grad():
    logits_final = model(input_ids)
    preds = logits_final.argmax(dim=-1)

print(f"target_ids: {target_ids.tolist()}")
print(f"preds:      {preds.tolist()}")
acertos = (preds == target_ids).float().mean().item()
print(f"Acurácia de memorização: {acertos * 100:.1f}%")
print("(Deve ser próxima de 100% se o teste de sanidade passou)")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Saída esperada: a loss deve cair de um valor inicial em torno de `log(vocab_size) ≈ 3.0` (comportamento aproximadamente aleatório) para próximo de `0.0` ao final dos 300 steps, e as predições finais devem coincidir exatamente com os targets.

---

## Erros Comuns

### Erro 1: Esquecer o Shift-by-One (Input e Target Idênticos)

```python
# ERRADO — target igual ao input não ensina o modelo a prever o PRÓXIMO token
input_ids = sequencia
target_ids = sequencia  # deveria estar deslocado!

# CERTO
input_ids = sequencia[:-1]
target_ids = sequencia[1:]
```

Esse erro é particularmente traiçoeiro porque o treinamento "roda" e a loss até cai — o modelo aprende a tarefa trivial de "copiar a entrada", que é fácil de otimizar mas completamente inútil.

### Erro 2: Ordem Errada das Operações no Loop de Treino

```python
# ERRADO — step() antes de backward() não tem gradiente para usar;
# clip depois de step() não tem efeito nenhum
optimizer.step()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# CERTO
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Erro 3: Tirar Conclusões do Teste de Overfitting com Poucos Steps

```python
# ERRADO — concluir "a implementação está quebrada" depois de só 10 steps
for step in range(10):
    ...
# loss ainda está em 2.5, "não caiu" -> conclusão precipitada

# CERTO — dar tempo suficiente (centenas de steps) antes de diagnosticar,
# já que a convergência em overfitting de batch único ainda leva algumas
# dezenas a centenas de iterações, dependendo do lr e do tamanho do modelo
for step in range(300):
    ...
```

---

## Exercícios

### Exercício 29.1: Shift-by-One Manual
Dada a sequência `[4, 8, 15, 16, 23, 42]`, escreva manualmente `input_ids` e `target_ids`.

### Exercício 29.2: Shapes do Loop de Treino
Para `batch_size=8`, `seq_len=32`, `vocab_size=1000`, quais são os shapes de `input_ids`, `target_ids`, `logits`, e da `loss` escalar final?

### Exercício 29.3: Implementar um Dataset Simples
Implemente uma classe `Dataset` do PyTorch que recebe uma lista longa de token ids e um `seq_len`, e retorna pares `(input_ids, target_ids)` via `__getitem__`, como no exemplo do capítulo.

### Exercício 29.4: Rodar o Teste de Overfitting
Usando o `MiniLM` do experimento, rode o teste de overfitting com uma sequência diferente (de sua escolha, tamanho 8) e confirme que a loss cai para perto de zero.

### Exercício 29.5: Diagnosticar um Loop de Treino Quebrado
O código abaixo não converge. Encontre e corrija o(s) bug(s):
```python
def train_step(model, optimizer, input_ids, target_ids):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()
    return loss
```

---

## Gabarito

### Exercício 29.1: Shift-by-One Manual
```python
sequencia = [4, 8, 15, 16, 23, 42]
input_ids = sequencia[:-1]   # [4, 8, 15, 16, 23]
target_ids = sequencia[1:]   # [8, 15, 16, 23, 42]
```

### Exercício 29.2: Shapes do Loop de Treino
```
input_ids:  [8, 32]
target_ids: [8, 32]
logits:     [8, 32, 1000]
loss:       escalar (shape [])  (média sobre 8*32=256 previsões de token)
```

### Exercício 29.3: Implementar um Dataset Simples
```python
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1])
        target_ids = torch.tensor(chunk[1:])
        return input_ids, target_ids
```

### Exercício 29.4: Rodar o Teste de Overfitting
```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
sequencia = torch.tensor([2, 9, 4, 11, 7, 1, 6, 13])
input_ids = sequencia[:-1].unsqueeze(0)
target_ids = sequencia[1:].unsqueeze(0)

model = MiniLM(vocab_size=20, d_model=16, seq_len=7)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(300):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 20), target_ids.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

print(f"loss final: {loss.item():.4f}")  # deve estar perto de 0
```

### Exercício 29.5: Diagnosticar um Loop de Treino Quebrado
```python
# Bugs: falta optimizer.zero_grad() (gradientes acumulam entre steps).

def train_step(model, optimizer, input_ids, target_ids):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss
```

---

## Desafios Avançados (Opcionais)

### Fixação 29.1: Curva de Overfitting em Função do Tamanho do Modelo
Repita o teste de overfitting com modelos de `d_model` diferentes (8, 16, 64). Como o número de steps necessário para chegar perto de loss zero muda com a capacidade do modelo?

### Fixação 29.2: Implementar Warmup + Cosine Decay Real
Integre a função `get_lr` da seção de warmup ao loop de treino, atualizando o `lr` do otimizador a cada step via `optimizer.param_groups[0]["lr"] = novo_lr`. Plote o lr ao longo dos steps.

### Fixação 29.3: Quebrar Teacher Forcing Propositalmente
Modifique o loop de treino para, com alguma probabilidade, usar a predição do próprio modelo (em vez do token real) como entrada da próxima posição (uma forma simplificada de "scheduled sampling"). Observe o efeito na estabilidade do treino.

### Fixação 29.4: Overfitting em Múltiplas Sequências
Repita o teste de overfitting, mas com um batch de 4 sequências diferentes simultaneamente. A convergência é mais lenta, mais rápida, ou similar?

### Fixação 29.5: Efeito do Gradient Clipping no Teste de Sanidade
Rode o teste de overfitting com e sem `clip_grad_norm_`. Em um modelo tão pequeno, o clipping muda o resultado? Em que ponto (tamanho de modelo, `lr`) você esperaria que ele começasse a importar mais?

---

## Resumo

- **Autoregressivo**: o modelo decompõe a probabilidade da sequência via regra da cadeia, prevendo cada token condicionado em todos os anteriores
- **Shift-by-one**: `input = sequência[:-1]`, `target = sequência[1:]` — a forma padrão de gerar pares de treino a partir de texto contínuo
- **Teacher forcing**: durante o treino, sempre usar o token real (não o predito) como entrada — viabiliza paralelização e estabiliza gradientes, à custa de um descompasso com a inferência (exposure bias)
- **Loop de treino**: `zero_grad -> forward -> loss -> backward -> clip -> step`, nesta ordem exata
- **Overfitting em batch único**: teste de sanidade essencial — se a loss não cai a quase zero em um único batch pequeno, há um bug na implementação, não um problema de dados insuficientes
- **Warmup + decay**: começar com lr baixo evita instabilidade inicial; decaimento ao longo do treino permite refinamento nos estágios finais

Próximo capítulo: **Checkpoints e Avaliação** — como salvar o progresso do treinamento, medir overfitting de verdade com um conjunto de validação, e debugar quando a loss não se comporta como esperado.

---

**Próximo**: [Capítulo 30: Checkpoints e Avaliação](30_checkpoints_avaliacao.md)
