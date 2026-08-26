# Capítulo 30: Checkpoints e Avaliação

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Salvar e carregar checkpoints completos de treinamento (modelo, otimizador, epoch, config, loss)
2. Decidir entre salvar os melhores checkpoints vs salvar periodicamente
3. Implementar train/validation split e entender por que validação é indispensável
4. Reconhecer sinais de overfitting e aplicar contramedidas (mais dados, dropout, early stopping)
5. Seguir um checklist prático para debugar treinos que não convergem ou que explodem em NaN

---

## Por Que Isso Importa

Você deixou um treinamento rodando por 6 horas. No minuto 359, a energia cai, ou o processo trava por falta de memória, ou você simplesmente precisa desligar a máquina para outra coisa. Sem checkpoints, tudo isso — 6 horas de compute, todo o progresso do modelo — desaparece, e você recomeça do zero. Checkpoints não são um "nice to have" de produção; são infraestrutura básica que qualquer pessoa treinando um modelo de qualquer tamanho não-trivial precisa desde o primeiro experimento sério.

Mas checkpoints resolvem só metade do problema. A outra metade é saber *quando* o modelo está realmente aprendendo a generalizar, e não apenas decorando o conjunto de treino. Um modelo com loss de treino caindo suavemente até quase zero pode, ao mesmo tempo, estar piorando cada vez mais em dados que nunca viu — esse é o overfitting clássico, e sem um conjunto de validação separado, você simplesmente não teria como perceber isso: a única métrica disponível (loss de treino) mentiria, dizendo "está tudo ótimo" enquanto o modelo se torna progressivamente menos útil.

E depois existe a categoria de problema mais frustrante de todas: você roda o treino e a loss simplesmente não desce — fica estagnada em um platô, ou pior, explode para `NaN` depois de algumas centenas de steps. Isso não é incomum, e quase sempre tem uma causa identificável (learning rate, inicialização, um bug de shape sutil, dados corrompidos). Este capítulo fecha o ciclo de treinamento com o que é, na prática, a habilidade mais usada no dia a dia de quem treina modelos: um checklist sistemático de debugging, para não perder horas "tentando coisas aleatórias" quando a loss não se comporta.

---

## Salvando Checkpoints

Um checkpoint completo precisa conter tudo necessário para retomar o treinamento exatamente de onde parou — não apenas os pesos do modelo:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "step": global_step,
    "config": config,          # hiperparâmetros usados (d_model, n_layers, lr, etc.)
    "train_loss": train_loss,
    "val_loss": val_loss,
}
torch.save(checkpoint, "checkpoint_epoch5.pt")
```

**Por que salvar o `optimizer.state_dict()` também?** Otimizadores como Adam mantêm estado interno por parâmetro (os momentos $m_t$ e $v_t$ do Capítulo 28). Se você retomar o treino carregando apenas os pesos do modelo mas recriando o otimizador do zero, ele perde toda a "memória" acumulada sobre a dinâmica recente dos gradientes — o equivalente a reiniciar o momentum, o que pode causar um pico temporário de instabilidade logo após retomar.

**Por que salvar a `config`?** Para reconstruir a arquitetura do modelo (quantas camadas, quantas cabeças de atenção, `d_model`, `vocab_size`) antes de carregar os pesos — sem isso, você não sabe qual arquitetura instanciar para receber o `state_dict` salvo.

### Carregando um Checkpoint

```python
checkpoint = torch.load("checkpoint_epoch5.pt", map_location=device)

model = MeuTransformer(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=checkpoint["config"]["lr"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
print(f"Retomando treino a partir da epoch {start_epoch}")
```

### Estratégias de Salvamento: Periódico vs Melhor Checkpoint

Existem duas estratégias comuns, geralmente combinadas:

**Salvamento periódico**: salvar a cada N steps ou a cada epoch, independentemente da qualidade, como proteção contra falhas (energia, crash, timeout de cluster). Geralmente mantém-se apenas os últimos K checkpoints para não esgotar espaço em disco.

**Salvar apenas o melhor**: manter um checkpoint separado (`best_model.pt`) que só é sobrescrito quando a **loss de validação** (nunca a de treino) melhora:

```python
best_val_loss = float("inf")

for epoch in range(n_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Checkpoint periódico (proteção contra falhas)
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, f"checkpoint_last.pt")

    # Melhor checkpoint (baseado em validação, não em treino!)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, "best_model.pt")
        print(f"  Novo melhor modelo salvo (val_loss={val_loss:.4f})")
```

O motivo de usar validação (e não treino) para decidir qual checkpoint é o "melhor" é o cerne do próximo tópico.

---

## Train/Validation Split

Antes de treinar, o dataset é dividido em pelo menos duas partes:

- **Conjunto de treino** (tipicamente 80-95% dos dados): usado para calcular gradientes e atualizar pesos.
- **Conjunto de validação** (o restante): usado *apenas* para medir performance, **nunca** para atualizar pesos.

```python
from torch.utils.data import random_split

n_total = len(dataset)
n_val = int(0.1 * n_total)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Por Que Validação é Indispensável

A loss de treino mede apenas o quão bem o modelo se ajusta aos exemplos específicos que ele já viu e usou para atualizar seus pesos. Um modelo com capacidade suficiente (parâmetros o bastante) pode, em princípio, **memorizar** o conjunto de treino inteiro — decorando padrões específicos daqueles exemplos exatos, incluindo ruído e particularidades que não generalizam — e ainda assim ter loss de treino extremamente baixa.

O conjunto de validação nunca participa da atualização de pesos, então a loss medida nele reflete a capacidade real do modelo de **generalizar** para dados que ele nunca viu — que é, afinal, o objetivo real de treinar um modelo. A avaliação em validação deve ser feita em modo `eval()` e sem gradientes:

```python
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, target_ids in val_loader:
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / n_batches
```

`model.eval()` desativa comportamentos específicos de treino (como dropout, que introduz ruído aleatório propositalmente) para que a avaliação seja determinística e reflita o desempenho real do modelo. `torch.no_grad()` evita construir o grafo computacional desnecessariamente, economizando memória e tempo — já que não vamos fazer backward durante avaliação.

---

## Sinais de Overfitting

O padrão clássico de overfitting nas curvas de loss:

```
Epoch  |  Train Loss  |  Val Loss
-------|--------------|----------
1      |  3.20        |  3.15
2      |  2.10         |  2.05
3      |  1.40         |  1.50
4      |  0.90         |  1.55
5      |  0.50         |  1.70   <- train continua caindo, val piora
6      |  0.25         |  1.90   <- gap crescendo
```

**O sinal característico**: a loss de treino continua caindo (o modelo continua se ajustando aos dados de treino), mas a loss de validação para de melhorar e começa a **subir** — o modelo está memorizando particularidades do conjunto de treino que não generalizam, e piorando ativamente sua capacidade de lidar com dados novos.

### Contramedidas

**1. Mais dados.** A causa mais fundamental de overfitting é capacidade do modelo desproporcional à quantidade de dados disponível. Mais dados de treino é, na maioria dos casos, a solução mais eficaz — mas nem sempre viável.

**2. Regularização (dropout, weight decay).** Dropout (zerar aleatoriamente uma fração das ativações durante o treino) impede que o modelo dependa excessivamente de combinações específicas de neurônios, forçando redundância e generalização. Weight decay (Capítulo 28) penaliza pesos grandes, favorecendo soluções mais simples.

```python
self.dropout = nn.Dropout(p=0.1)
# aplicado após atenção e após MLP, tipicamente
```

**3. Early stopping.** Parar o treinamento quando a loss de validação para de melhorar por um número definido de epochs consecutivas (paciência), em vez de treinar pelo número fixo de epochs planejado originalmente:

```python
patience = 3
epochs_sem_melhora = 0
best_val_loss = float("inf")

for epoch in range(n_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_sem_melhora = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_sem_melhora += 1

    if epochs_sem_melhora >= patience:
        print(f"Early stopping na epoch {epoch} (sem melhora por {patience} epochs)")
        break
```

**4. Reduzir a capacidade do modelo.** Se o dataset é pequeno e fixo, reduzir `d_model`, número de camadas ou número de cabeças de atenção diminui a capacidade de memorização bruta, forçando o modelo a aprender padrões mais gerais em vez de decorar exemplos individuais.

---

## Checklist de Debugging

### Quando a Loss Não Cai

1. **Learning rate errado.** Muito baixo (`1e-6` quando deveria ser `3e-4`) faz o treino parecer "travado" — os passos são pequenos demais para sair do ponto inicial em tempo razoável. Teste com um lr maior antes de suspeitar de outra coisa.
2. **Inicialização de pesos ruim.** Pesos inicializados com variância muito alta ou muito baixa podem fazer ativações saturarem ou colapsarem logo no início, dificultando qualquer aprendizado. Verifique se `nn.init` está sendo aplicado corretamente.
3. **Bug de shape silencioso.** Um `view()` ou `reshape()` incorreto pode embaralhar dados entre exemplos do batch sem gerar erro (o shape final "bate", mas o conteúdo está errado). Sempre imprima shapes intermediários ao debugar.
4. **Dados errados.** Verifique se `input_ids` e `target_ids` estão de fato deslocados corretamente (Capítulo 29), se a tokenização não está corrompida, e se os dados não estão, por exemplo, todos zerados por um bug de carregamento.
5. **Gradientes não fluindo.** Verifique `param.grad` após `.backward()` — se estiverem `None` ou todos zero em alguma camada, há uma desconexão no grafo computacional (Capítulo 27).

```python
# Snippet de diagnóstico: inspecionar gradientes por camada
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"AVISO: {name} não tem gradiente!")
    else:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

### Quando a Loss Explode para NaN

1. **Learning rate muito alto.** A causa mais comum de longe — um passo grande demais empurra os pesos para uma região onde as ativações explodem, gerando `Inf`, e a próxima operação (como `log`) produz `NaN`.
2. **Falta de gradient clipping.** Sem `clip_grad_norm_`, um pico ocasional de gradiente (comum em Transformers, especialmente no início do treino) pode causar um update catastrófico de uma vez.
3. **Divisão por zero ou log de zero.** Comum em implementações manuais de softmax/cross-entropy sem estabilidade numérica (Capítulo 26), ou em normalizações customizadas onde o denominador pode chegar perto de zero.
4. **Precisão numérica reduzida (fp16) sem escala adequada.** Se estiver usando `float16` para treino misto, valores muito pequenos podem sofrer *underflow* e virar zero, ou muito grandes podem sofrer *overflow*, especialmente em softmax e nas normas de LayerNorm.

```python
# Snippet de diagnóstico: abortar o step se a loss for inválida
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Loss inválida detectada: {loss.item()}. Pulando este step.")
    optimizer.zero_grad()
    continue
```

---

## Experimento: Checkpointing, Validação e Detecção de Overfitting

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Checkpoints, Validação e Overfitting")
print("=" * 70)

# ========== 1. MODELO E DADOS SINTÉTICOS ==========
print("\n1. SETUP: MODELO PEQUENO + DADOS SINTÉTICOS (COM RUÍDO)")
print("-" * 70)

class MiniLM(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
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
        return self.head(h)

vocab_size = 30
seq_len = 8
n_examples_train = 40
n_examples_val = 15

# Dados de treino "estruturados" (padrão real) + dados de validação de mesma distribuição
def gerar_dados(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n, seq_len + 1), generator=g)

train_data = gerar_dados(n_examples_train, seed=1)
val_data = gerar_dados(n_examples_val, seed=2)

print(f"train_data shape: {train_data.shape}")
print(f"val_data shape: {val_data.shape}")

def batches(data, batch_size=8):
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        yield chunk[:, :-1], chunk[:, 1:]

# ========== 2. MODELO PROPOSITALMENTE GRANDE DEMAIS (PARA FORÇAR OVERFITTING) ==========
print("\n2. TREINANDO UM MODELO GRANDE DEMAIS PARA POUCOS DADOS (overfitting proposital)")
print("-" * 70)

model = MiniLM(vocab_size, d_model=64, seq_len=seq_len, dropout=0.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in batches(data):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / n

# ========== 3. LOOP DE TREINO COM VALIDAÇÃO E CHECKPOINTING ==========
print("\n3. LOOP DE TREINO: TRAIN LOSS VS VAL LOSS POR EPOCH")
print("-" * 70)

best_val_loss = float("inf")
history = []

for epoch in range(15):
    model.train()
    train_losses = []
    for x, y in batches(train_data):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = sum(train_losses) / len(train_losses)
    val_loss = evaluate(model, val_data)
    history.append((epoch, train_loss, val_loss))

    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": {"vocab_size": vocab_size, "d_model": 64, "seq_len": seq_len},
        }
        marker = " <- novo melhor checkpoint salvo"

    print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}{marker}")

# ========== 4. ANALISANDO O GAP TRAIN/VAL ==========
print("\n4. ANALISANDO OVERFITTING (gap entre train e val)")
print("-" * 70)

for epoch, tl, vl in history[::3]:
    gap = vl - tl
    status = "overfitting crescente" if gap > 0.5 else "gap saudável"
    print(f"Epoch {epoch:2d}: gap (val-train) = {gap:+.4f}  [{status}]")

# ========== 5. SALVANDO E RECARREGANDO O CHECKPOINT ==========
print("\n5. SALVANDO E RECARREGANDO O MELHOR CHECKPOINT")
print("-" * 70)

torch.save(checkpoint, "/tmp/best_checkpoint_demo.pt")
print(f"Checkpoint salvo (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

loaded = torch.load("/tmp/best_checkpoint_demo.pt")
model_reloaded = MiniLM(**loaded["config"])
model_reloaded.load_state_dict(loaded["model_state_dict"])
print("Checkpoint recarregado com sucesso.")

val_loss_reloaded = evaluate(model_reloaded, val_data)
print(f"val_loss após reload: {val_loss_reloaded:.4f} (deve bater com {loaded['val_loss']:.4f})")

# ========== 6. DIAGNÓSTICO: DETECTANDO NaN ==========
print("\n6. SIMULANDO E DETECTANDO LOSS NaN")
print("-" * 70)

loss_normal = torch.tensor(1.5)
loss_nan = torch.tensor(float("nan"))
loss_inf = torch.tensor(float("inf"))

for nome, l in [("normal", loss_normal), ("nan", loss_nan), ("inf", loss_inf)]:
    invalido = torch.isnan(l) or torch.isinf(l)
    print(f"loss {nome} = {l.item()} -> inválida? {invalido.item()}")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

Com dados sintéticos aleatórios (sem padrão real de linguagem), o modelo grande tende a mostrar sinais claros de overfitting: a loss de treino continua caindo enquanto a de validação estagna ou piora — exatamente o padrão descrito na seção anterior.

---

## Erros Comuns

### Erro 1: Decidir o Melhor Checkpoint com Base na Loss de Treino

```python
# ERRADO — usa train_loss para decidir o melhor checkpoint
if train_loss < best_train_loss:
    best_train_loss = train_loss
    torch.save(model.state_dict(), "best_model.pt")
# Isso favorece justamente o modelo mais "overfitado"

# CERTO — sempre usar val_loss
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
```

### Erro 2: Esquecer `model.eval()` Durante Avaliação

```python
# ERRADO — dropout continua ativo durante avaliação, tornando os
# resultados não-determinísticos e artificialmente piores
def evaluate(model, val_loader):
    total_loss = 0.0
    for x, y in val_loader:
        logits = model(x)  # dropout ainda ativo!
        ...

# CERTO
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    for x, y in val_loader:
        logits = model(x)
        ...
    model.train()  # não esquecer de voltar ao modo treino depois
```

### Erro 3: Salvar Apenas os Pesos, Sem Estado do Otimizador

```python
# ERRADO — perde o estado do Adam (momentos m_t, v_t) ao retomar
torch.save(model.state_dict(), "checkpoint.pt")

# CERTO — salva tudo necessário para retomar sem instabilidade
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")
```

---

## Exercícios

### Exercício 30.1: Salvar e Carregar um Checkpoint Completo
Escreva o código para salvar um checkpoint contendo modelo, otimizador, epoch e uma `config` arbitrária, e depois carregá-lo de volta em um novo modelo/otimizador.

### Exercício 30.2: Implementar Early Stopping
Escreva um loop de treino que interrompe automaticamente após 4 epochs consecutivas sem melhora na loss de validação.

### Exercício 30.3: Diagnosticar Overfitting a Partir de uma Tabela
Dada a tabela de losses abaixo, identifique em qual epoch o overfitting começa a se tornar visível:
```
Epoch 1: train=2.80 val=2.75
Epoch 2: train=1.90 val=1.95
Epoch 3: train=1.20 val=1.30
Epoch 4: train=0.70 val=1.45
Epoch 5: train=0.40 val=1.80
```

### Exercício 30.4: Checklist de Debugging Aplicado
Você roda um treino e a loss fica travada em `3.4` (aproximadamente `log(vocab_size)` para um vocabulário de 30 tokens) por 500 steps, sem se mover. Liste, em ordem de prioridade, as 3 primeiras coisas que você verificaria.

### Exercício 30.5: Detectar e Recuperar de Loss NaN
Escreva um loop de treino que detecta `NaN`/`Inf` na loss, pula o step problemático (sem atualizar pesos), e continua o treinamento normalmente nos steps seguintes.

---

## Gabarito

### Exercício 30.1: Salvar e Carregar um Checkpoint Completo
```python
import torch

# Salvar
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 7,
    "config": {"vocab_size": 30, "d_model": 64, "seq_len": 8},
}
torch.save(checkpoint, "meu_checkpoint.pt")

# Carregar
loaded = torch.load("meu_checkpoint.pt")
novo_model = MiniLM(**loaded["config"])
novo_model.load_state_dict(loaded["model_state_dict"])

novo_optimizer = torch.optim.AdamW(novo_model.parameters())
novo_optimizer.load_state_dict(loaded["optimizer_state_dict"])

print(f"Retomando da epoch {loaded['epoch'] + 1}")
```

### Exercício 30.2: Implementar Early Stopping
```python
patience = 4
epochs_sem_melhora = 0
best_val_loss = float("inf")

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_sem_melhora = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_sem_melhora += 1

    if epochs_sem_melhora >= patience:
        print(f"Early stopping na epoch {epoch}")
        break
```

### Exercício 30.3: Diagnosticar Overfitting a Partir de uma Tabela
```
O gap (val - train) começa a crescer visivelmente a partir da Epoch 4:
Epoch 3: gap = 1.30 - 1.20 = 0.10  (saudável)
Epoch 4: gap = 1.45 - 0.70 = 0.75  (overfitting começando)
Epoch 5: gap = 1.80 - 0.40 = 1.40  (overfitting claro)

O ponto de inflexão real está entre a Epoch 3 e a Epoch 4: até a Epoch 3,
val_loss ainda melhora junto com train_loss; a partir da Epoch 4, val_loss
piora enquanto train_loss continua caindo. O melhor checkpoint seria o
da Epoch 3.
```

### Exercício 30.4: Checklist de Debugging Aplicado
```
1. Verificar o learning rate — se estiver muito baixo (ex: 1e-6),
   o modelo mal se move do estado inicial em 500 steps.
2. Verificar se os gradientes estão de fato fluindo — inspecionar
   param.grad para todas as camadas, buscando None ou normas zero.
3. Verificar se input_ids/target_ids estão corretos — confirmar o
   shift-by-one e que os dados não estão, por exemplo, zerados por
   um bug de carregamento ou tokenização quebrada.

(loss travada em ~log(vocab_size) é o comportamento de um modelo que
não aprendeu NADA — está distribuindo probabilidade uniformemente,
como se estivesse na inicialização.)
```

### Exercício 30.5: Detectar e Recuperar de Loss NaN
```python
for step, (x, y) in enumerate(train_loader):
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Step {step}: loss inválida ({loss.item()}), pulando.")
        continue  # não faz backward nem step

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## Desafios Avançados (Opcionais)

### Fixação 30.1: Checkpoint Incremental com Rotação
Implemente um esquema que mantém apenas os últimos 3 checkpoints periódicos em disco, deletando automaticamente os mais antigos a cada novo salvamento.

### Fixação 30.2: Curva de Overfitting em Função da Regularização
Repita o experimento do capítulo com `dropout=0.0`, `0.1` e `0.3`. Compare o gap entre train e val loss em cada configuração.

### Fixação 30.3: Validação com K-Fold
Em vez de um único split fixo train/val, implemente uma validação k-fold simples (k=5) e calcule a média e o desvio padrão da val_loss entre os folds.

### Fixação 30.4: Detecção Automática de Overfitting
Escreva uma função que analisa o histórico de `(train_loss, val_loss)` por epoch e retorna automaticamente a epoch em que o overfitting começou (heurística: primeira epoch em que val_loss aumenta por 2 epochs consecutivas).

### Fixação 30.5: Gradient Norm Logging como Ferramenta de Diagnóstico
Registre a norma do gradiente (antes do clipping) a cada step ao longo de um treino inteiro. Plote essa série temporal. Picos anômalos de norma coincidem com quedas de qualidade do modelo, ou com origens específicas nos dados (ex: batches com sequências muito longas ou repetitivas)?

---

## Resumo

- **Checkpoint completo**: `model.state_dict()` + `optimizer.state_dict()` + epoch + config + métricas — não apenas os pesos
- **Salvar por validação, não por treino**: o "melhor" checkpoint deve ser decidido com base na loss de validação, nunca na de treino
- **Train/validation split**: essencial para distinguir memorização de generalização — a loss de treino sozinha é enganosa
- **Sinal de overfitting**: train loss continua caindo enquanto val loss estagna ou sobe — o gap entre elas cresce
- **Contramedidas**: mais dados, dropout, weight decay, early stopping, ou reduzir a capacidade do modelo
- **Checklist de debugging**: loss travada → verificar lr, gradientes, dados; loss NaN → verificar lr alto, falta de clipping, divisões/logs instáveis

Próximo capítulo: **Sampling, Temperature e Top-K/P** — depois de um modelo treinado e validado, como usá-lo de fato para gerar texto novo, controlando o equilíbrio entre coerência e criatividade.

---

**Próximo**: [Capítulo 31: Sampling, Temperature e Top-K/P](31_sampling_temperature.md)
