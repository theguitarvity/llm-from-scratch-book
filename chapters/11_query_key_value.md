# Capítulo 11: Q, K, V Explicados

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Explicar por que precisamos de três projeções lineares separadas (W_Q, W_K, W_V) em vez de reutilizar a mesma matriz
2. Implementar as projeções Q, K, V com `nn.Linear` e rastrear seus shapes em cada etapa
3. Descrever o papel semântico de cada uma das três quantidades (Query, Key, Value)
4. Diferenciar self-attention (Q, K, V da mesma origem) de cross-attention (Q de um lugar, K/V de outro)
5. Debugar erros comuns de shape ao implementar projeções Q/K/V

---

## Por Que Isso Importa

No Capítulo 10 você viu, de forma simplificada, que atenção usa Q, K e V, e por conveniência didática usamos $W_Q = W_K = W_V = I$ (matriz identidade). Isso deixou a matemática mais fácil de acompanhar, mas escondeu uma pergunta importante: **por que usar três projeções lineares diferentes, em vez de simplesmente reaproveitar o embedding X diretamente, ou usar a mesma matriz de projeção três vezes?**

Essa pergunta não é só acadêmica. Se você já debugou um Transformer que "não aprende nada" ou "converge para atenção uniforme e nunca sai dali", uma causa comum é justamente essa: alguém inicializou W_Q e W_K com os mesmos pesos (ou pior, compartilhou a mesma camada `nn.Linear` para as três), e o modelo perdeu a capacidade de aprender papéis distintos para cada token.

Pense em uma festa de networking. Cada pessoa carrega, ao mesmo tempo, três informações diferentes sobre si: (1) o que ela está procurando na festa agora ("estou buscando alguém que trabalhe com dados") — isso é a **Query**; (2) como ela se anuncia para os outros ("eu trabalho com dados, 5 anos de experiência") — isso é a **Key**; e (3) o que ela realmente tem a oferecer numa conversa, se alguém decidir conversar com ela ("posso falar sobre meu projeto de pipeline de dados") — isso é o **Value**. Repare que essas três coisas são sobre a mesma pessoa, mas não são a mesma informação. O que você procura não é necessariamente igual ao que você oferece, e como você se anuncia pode nem coincidir perfeitamente com o que você realmente entrega.

É exatamente essa distinção que W_Q, W_K e W_V aprendem a capturar. Cada uma delas é uma "lente" diferente aplicada ao mesmo embedding de entrada X, extraindo um aspecto diferente da informação contida naquele token. Se usássemos a mesma matriz para as três, estaríamos forçando "o que eu procuro", "como eu me anuncio" e "o que eu ofereço" a serem sempre idênticos — o que colapsaria boa parte da expressividade do mecanismo de atenção.

---

## As Três Projeções Lineares

### Ponto de partida: por que precisamos de X primeiro

Antes de Q, K, V existirem, precisamos de uma representação vetorial dos tokens: o embedding $X \in \mathbb{R}^{[n, d]}$, onde $n$ é o comprimento da sequência e $d$ é a dimensão do modelo (`d_model`). Esse X já foi construído nos Capítulos 6-9 (embedding lookup + eventualmente soma de posição). Q, K, V **não são entradas independentes** — todas as três derivam do mesmo X, através de transformações lineares diferentes.

### As matrizes de projeção

Definimos três matrizes de pesos aprendidas:

$$W_Q \in \mathbb{R}^{[d_{model}, d_k]}, \quad W_K \in \mathbb{R}^{[d_{model}, d_k]}, \quad W_V \in \mathbb{R}^{[d_{model}, d_v]}$$

Na prática, na maioria das implementações (incluindo a que vamos usar), $d_k = d_v = d_{model}$ quando não há multi-head (esse detalhe muda no Capítulo 17, quando dividimos $d_{model}$ entre várias cabeças). Por enquanto, vamos manter tudo na mesma dimensão $d$ para simplificar.

As projeções são:

$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V$$

Com shapes:

$$X \in \mathbb{R}^{[n, d]} \quad\to\quad Q, K, V \in \mathbb{R}^{[n, d]}$$

Ou, incluindo a dimensão de batch, como é comum em PyTorch:

$$X \in \mathbb{R}^{[batch, n, d]} \quad\to\quad Q, K, V \in \mathbb{R}^{[batch, n, d]}$$

Note que Q, K e V têm exatamente o mesmo shape — o que muda entre elas não é a forma, mas o **conteúdo**: cada uma é uma combinação linear diferente das mesmas features de entrada, porque $W_Q$, $W_K$, $W_V$ têm valores diferentes (e são atualizadas de forma independente durante o treinamento).

### Por que três matrizes e não uma só

Suponha, por absurdo, que $W_Q = W_K = W_V = W$. Nesse caso, $Q = K = V = XW$. O que acontece com os scores de atenção?

$$\text{scores} = QK^T = (XW)(XW)^T = XWW^TX^T$$

Isso ainda funciona matematicamente, mas force uma restrição severa: a matriz de similaridade entre tokens fica presa a uma forma quadrática simétrica em torno de $WW^T$ (que é sempre semidefinida positiva). Isso significa, entre outras coisas, que **a matriz de scores seria sempre simétrica** ($\text{scores}[i,j] = \text{scores}[j,i]$), o que quebra uma propriedade essencial da atenção: a relação "token A presta atenção em token B" não precisa ser recíproca. Um pronome como "ele" precisa prestar muita atenção no substantivo que ele referencia, mas o substantivo não precisa prestar atenção de volta no pronome com a mesma intensidade.

Além disso, quando fazemos $V$ igual a $Q$ ou $K$, o valor que é efetivamente "copiado" para a saída fica acoplado à forma como a relevância é calculada, reduzindo a capacidade do modelo de separar "o que é relevante" de "o que deve ser passado adiante". Três matrizes independentes dão ao otimizador (SGD/Adam) três graus de liberdade extras para encontrar soluções melhores.

### Papel semântico de cada projeção

- **Query ($Q$)**: representa "o que esse token está procurando" em outros tokens. É a pergunta que a posição atual faz ao resto da sequência.
- **Key ($K$)**: representa "como esse token se anuncia" — a etiqueta que ele expõe para que outras posições, ao formularem uma Query, possam decidir se ele é relevante.
- **Value ($V$)**: representa "o que esse token efetivamente oferece" quando é selecionado. É o conteúdo de informação real que flui para a saída, ponderado pela relevância calculada via Q e K.

Uma forma útil de lembrar: **Q e K decidem QUANTO atenção dar** (são usados só para calcular os pesos), e **V decide O QUE é passado adiante** (é combinado com os pesos calculados). Se você trocar Q e K por Value errado, a saída fica com conteúdo errado mesmo que os pesos de relevância estejam corretos — e vice-versa.

---

## Implementação com nn.Linear

Até agora usamos multiplicação de matriz explícita ($XW$). Na prática, usamos `nn.Linear`, que já embute a matriz de pesos (e, opcionalmente, um bias) como parâmetros aprendíveis.

```python
import torch
import torch.nn as nn

d_model = 8

# Uma camada linear PARA CADA papel — pesos independentes
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)
```

Repare: `bias=False` é comum (não sempre obrigatório) em implementações de atenção, porque o bias em Q/K/V costuma trazer pouco benefício e adiciona parâmetros. Muitas implementações modernas (GPT-2, LLaMA) removem o bias dessas projeções.

Aplicando a um batch de embeddings:

```python
batch_size, seq_len = 2, 5
X = torch.randn(batch_size, seq_len, d_model)

Q = W_q(X)  # [2, 5, 8]
K = W_k(X)  # [2, 5, 8]
V = W_v(X)  # [2, 5, 8]
```

Internamente, `nn.Linear(d_model, d_model)` guarda um peso de shape `[d_model, d_model]` (na verdade `[out_features, in_features]`, transposto internamente na multiplicação) e faz $Y = XW^T + b$. O resultado, para nossos propósitos, é equivalente à formulação $XW$ que usamos nas fórmulas — a diferença de convenção (transposta ou não) é só um detalhe de implementação do PyTorch.

### Verificando que os pesos são realmente diferentes

Um bug clássico é inicializar as três camadas de forma que acabem idênticas (por exemplo, copiando o `state_dict` de uma para as outras por engano). Vale sempre verificar:

```python
print(torch.equal(W_q.weight, W_k.weight))  # Deve ser False
print(torch.equal(W_k.weight, W_v.weight))  # Deve ser False
```

Se qualquer uma dessas comparações retornar `True` logo após a inicialização aleatória, algo está errado na forma como as camadas foram criadas (provavelmente você criou uma única `nn.Linear` e reusou o mesmo objeto três vezes).

---

## Self-Attention vs Cross-Attention

Até aqui, assumimos implicitamente que Q, K e V vêm todos do mesmo tensor de entrada X. Isso é chamado de **self-attention**: a sequência presta atenção em si mesma.

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \qquad \text{(mesma origem } X\text{)}$$

Mas o mecanismo de atenção não exige isso. Em arquiteturas encoder-decoder (como o Transformer original de tradução, ou modelos de captioning de imagem), existe **cross-attention**: a Query vem de uma sequência (por exemplo, o decoder, gerando a tradução em português), enquanto Key e Value vêm de uma sequência diferente (o encoder, que processou a frase original em inglês).

$$Q = X_{decoder} W_Q, \quad K = X_{encoder} W_K, \quad V = X_{encoder} W_V$$

Shapes explícitos, com sequências de comprimentos diferentes:

- $X_{decoder} \in \mathbb{R}^{[batch, n_{dec}, d]}$ → $Q \in \mathbb{R}^{[batch, n_{dec}, d]}$
- $X_{encoder} \in \mathbb{R}^{[batch, n_{enc}, d]}$ → $K, V \in \mathbb{R}^{[batch, n_{enc}, d]}$
- Scores: $QK^T \in \mathbb{R}^{[batch, n_{dec}, n_{enc}]}$ — note que a matriz de scores **não precisa ser quadrada** em cross-attention, porque $n_{dec}$ pode ser diferente de $n_{enc}$.
- Output final: $\in \mathbb{R}^{[batch, n_{dec}, d]}$ — sempre no shape da Query, porque é para cada posição do decoder que estamos computando uma saída.

Este livro, sendo focado em um modelo puramente decoder (estilo GPT), usa **apenas self-attention** — não implementaremos cross-attention explicitamente. Mas entender a distinção deixa claro por que a arquitetura é flexível: Q, K, V são só três projeções lineares independentes, e nada na matemática exige que elas venham da mesma fonte.

---

## Experimento: Q, K, V na Prática

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENTO: Projeções Q, K, V")
print("=" * 70)

# ========== CONFIGURAÇÃO ==========
batch_size = 2
seq_len = 4
d_model = 6

print(f"\nConfiguracao:")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")
print(f"  d_model: {d_model}")

# ========== ENTRADA ==========
print("\n1. ENTRADA (X)")
print("-" * 70)

X = torch.randn(batch_size, seq_len, d_model)
print(f"X shape: {X.shape}")

# ========== CRIANDO AS TRÊS PROJEÇÕES ==========
print("\n2. CRIANDO W_Q, W_K, W_V INDEPENDENTES")
print("-" * 70)

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

print(f"W_q.weight shape: {W_q.weight.shape}")
print(f"W_k.weight shape: {W_k.weight.shape}")
print(f"W_v.weight shape: {W_v.weight.shape}")

# Confirmar que são independentes
print(f"\nW_q == W_k? {torch.equal(W_q.weight, W_k.weight)}")
print(f"W_k == W_v? {torch.equal(W_k.weight, W_v.weight)}")

# ========== APLICANDO AS PROJEÇÕES ==========
print("\n3. APLICANDO PROJEÇÕES")
print("-" * 70)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# ========== VERIFICANDO QUE SÃO CONTEÚDOS DIFERENTES ==========
print("\n4. VERIFICANDO CONTEÚDO (mesma origem, valores diferentes)")
print("-" * 70)

print(f"Q[0,0,:] = {Q[0,0,:].detach().numpy().round(3)}")
print(f"K[0,0,:] = {K[0,0,:].detach().numpy().round(3)}")
print(f"V[0,0,:] = {V[0,0,:].detach().numpy().round(3)}")
print("\nRepare: os três vetores vêm do MESMO X[0,0,:], mas são DIFERENTES")
print("porque cada um passou por uma matriz de projeção distinta.")

# ========== SELF-ATTENTION: MESMA ORIGEM ==========
print("\n5. SELF-ATTENTION: Q, K, V da mesma sequência")
print("-" * 70)

print(f"X usado para Q: shape {X.shape}")
print(f"X usado para K: shape {X.shape}")
print(f"X usado para V: shape {X.shape}")
print("Em self-attention, a mesma sequência gera Q, K e V.")

# ========== CROSS-ATTENTION: ORIGENS DIFERENTES ==========
print("\n6. CROSS-ATTENTION: Q de um lugar, K/V de outro")
print("-" * 70)

seq_len_decoder = 3
seq_len_encoder = 7

X_decoder = torch.randn(batch_size, seq_len_decoder, d_model)
X_encoder = torch.randn(batch_size, seq_len_encoder, d_model)

Q_cross = W_q(X_decoder)   # Query vem do decoder
K_cross = W_k(X_encoder)   # Key vem do encoder
V_cross = W_v(X_encoder)   # Value vem do encoder

print(f"X_decoder shape: {X_decoder.shape}")
print(f"X_encoder shape: {X_encoder.shape}")
print(f"Q_cross shape: {Q_cross.shape}  (segue o comprimento do decoder)")
print(f"K_cross shape: {K_cross.shape}  (segue o comprimento do encoder)")
print(f"V_cross shape: {V_cross.shape}  (segue o comprimento do encoder)")

scores_cross = Q_cross @ K_cross.transpose(-2, -1)
print(f"\nscores_cross shape: {scores_cross.shape}  <- NÃO é quadrada!")
print(f"  [batch={batch_size}, n_dec={seq_len_decoder}, n_enc={seq_len_encoder}]")

print("\n" + "=" * 70)
print("Experimento Completo!")
print("=" * 70)
```

---

## Erros Comuns

### Erro 1: Compartilhar a mesma camada Linear para Q, K, V

```python
# Errado
shared_linear = nn.Linear(d_model, d_model)
Q = shared_linear(X)
K = shared_linear(X)
V = shared_linear(X)
# Q, K, V serão IDÊNTICOS (mesmos pesos, mesma entrada = mesma saída)

# Certo
W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)
Q = W_q(X)
K = W_k(X)
V = W_v(X)
```

### Erro 2: Confundir o shape de saída de Q com o de K em cross-attention

```python
# Errado: assumir que Q e K sempre têm o mesmo comprimento de sequência
scores = Q_cross @ K_cross.T  # Falha se os batches/dims não alinharem certo

# Certo: usar transpose nas duas últimas dimensões, respeitando shapes distintos
scores = Q_cross @ K_cross.transpose(-2, -1)
# [batch, n_dec, d] @ [batch, d, n_enc] = [batch, n_dec, n_enc]
```

### Erro 3: Esquecer que V não participa do cálculo de scores

```python
# Errado: usar V para calcular similaridade
scores = Q @ V.transpose(-2, -1)  # Semanticamente errado!

# Certo: scores usam Q e K; V só entra depois, na ponderação
scores = Q @ K.transpose(-2, -1)
attn_weights = torch.softmax(scores, dim=-1)
output = attn_weights @ V
```

---

## Exercícios

### Exercício 11.1: Shapes de Cross-Attention
Se `X_decoder` tem shape `[1, 4, 16]` e `X_encoder` tem shape `[1, 10, 16]`, quais são os shapes de Q, K, V e da matriz de scores?

### Exercício 11.2: Detectando Pesos Compartilhados
Escreva uma função `verificar_pesos_independentes(W_q, W_k, W_v)` que recebe três camadas `nn.Linear` e levanta um erro se quaisquer duas tiverem pesos idênticos.

### Exercício 11.3: Implementação sem nn.Linear
Reimplemente as projeções Q, K, V usando apenas `torch.randn` para criar as matrizes de peso e multiplicação de matriz explícita (`@`), sem usar `nn.Linear`.

### Exercício 11.4: Papel de Cada Projeção
Dado o texto "O gato perseguiu o rato porque estava com fome", explique em uma frase o que a Query do token "estava" deveria estar "procurando", e qual token deveria ter uma Key mais compatível.

### Exercício 11.5: Self vs Cross
Classifique cada cenário abaixo como self-attention ou cross-attention: (a) GPT gerando o próximo token olhando para o texto já gerado; (b) um modelo de tradução olhando, ao gerar cada palavra em português, para as palavras da frase original em inglês; (c) um classificador de sentimento processando uma única frase.

---

## Gabarito

### Exercício 11.1: Shapes de Cross-Attention
```python
import torch, torch.nn as nn

d_model = 16
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

X_decoder = torch.randn(1, 4, 16)
X_encoder = torch.randn(1, 10, 16)

Q = W_q(X_decoder)  # [1, 4, 16]
K = W_k(X_encoder)  # [1, 10, 16]
V = W_v(X_encoder)  # [1, 10, 16]

scores = Q @ K.transpose(-2, -1)  # [1, 4, 10]
print(Q.shape, K.shape, V.shape, scores.shape)
# torch.Size([1, 4, 16]) torch.Size([1, 10, 16]) torch.Size([1, 10, 16]) torch.Size([1, 4, 10])
```

### Exercício 11.2: Detectando Pesos Compartilhados
```python
import torch

def verificar_pesos_independentes(W_q, W_k, W_v):
    pares = [("W_q", "W_k", W_q, W_k), ("W_k", "W_v", W_k, W_v), ("W_q", "W_v", W_q, W_v)]
    for nome_a, nome_b, a, b in pares:
        if torch.equal(a.weight, b.weight):
            raise ValueError(f"{nome_a} e {nome_b} têm pesos idênticos! Verifique a inicialização.")
    print("Todas as projeções são independentes.")

verificar_pesos_independentes(W_q, W_k, W_v)
```

### Exercício 11.3: Implementação sem nn.Linear
```python
import torch

d_model = 8
seq_len = 5
batch_size = 1

torch.manual_seed(42)
X = torch.randn(batch_size, seq_len, d_model)

W_Q = torch.randn(d_model, d_model) * 0.1
W_K = torch.randn(d_model, d_model) * 0.1
W_V = torch.randn(d_model, d_model) * 0.1

Q = X @ W_Q  # [1, 5, 8]
K = X @ W_K  # [1, 5, 8]
V = X @ W_V  # [1, 5, 8]

print(Q.shape, K.shape, V.shape)
```

### Exercício 11.4: Papel de Cada Projeção
A Query de "estava" deveria procurar por informação sobre "quem estava com fome" — ou seja, ela busca o sujeito da oração relativa. O token com a Key mais compatível provavelmente seria "rato" (o sujeito mais próximo e sintaticamente plausível), não "gato". Esse é exatamente o tipo de ambiguidade de correferência que um modelo bem treinado aprende a resolver através dos pesos de W_Q e W_K.

### Exercício 11.5: Self vs Cross
```
(a) Self-attention — o texto gerado presta atenção nele mesmo (incluindo o próprio contexto anterior).
(b) Cross-attention — Query vem do decoder (português sendo gerado), Key/Value vêm do encoder (frase em inglês).
(c) Self-attention — uma única sequência sendo processada, sem uma segunda fonte de K/V.
```

---

## Desafios Avançados (Opcionais)

### Fixação 11.1: d_k Diferente de d_model
Modifique o experimento para que $W_Q$ e $W_K$ projetem para uma dimensão $d_k = 4$, menor que $d_{model} = 8$, enquanto $W_V$ mantém $d_v = 8$. Verifique que os scores ainda funcionam (shape `[n, n]`) e que o output final ainda tem shape `[n, d_v]`.

### Fixação 11.2: Inicialização e Colapso de Atenção
Inicialize $W_Q$ e $W_K$ com pesos muito pequenos (multiplicados por 0.001). Compute a matriz de atenção resultante. Ela fica próxima de uniforme? Por quê isso é um problema no início do treinamento?

### Fixação 11.3: Compartilhando K e V (mas não Q)
Algumas arquiteturas eficientes (como Multi-Query Attention) compartilham K e V entre múltiplas cabeças, mantendo Q independente. Implemente uma versão simplificada com uma única $W_K$ e $W_V$ compartilhada, mas duas $W_Q$ diferentes, e compare o número de parâmetros com a versão totalmente independente.

### Fixação 11.4: Cross-Attention com Máscara de Padding
Em cross-attention real, o encoder pode ter tokens de padding que não devem receber atenção. Implemente uma máscara que zera (via `-inf` antes do softmax) as colunas de `scores_cross` correspondentes a posições de padding do encoder.

### Fixação 11.5: Gradiente de Cada Projeção
Faça um `backward()` a partir da soma do output de atenção e inspecione `W_q.weight.grad`, `W_k.weight.grad`, `W_v.weight.grad`. Eles são diferentes entre si? O que isso confirma sobre a independência das três projeções?

---

## Resumo

- **Três projeções independentes**: W_Q, W_K, W_V são matrizes aprendidas separadamente, nunca a mesma matriz reaproveitada
- **Mesma origem X, papéis diferentes**: Q, K, V derivam do mesmo embedding, mas capturam aspectos diferentes da informação
- **Q e K decidem "quanto", V decide "o quê"**: Query e Key só determinam os pesos de atenção; Value carrega o conteúdo que efetivamente flui para a saída
- **Self-attention vs cross-attention**: em self-attention, Q, K, V vêm da mesma sequência; em cross-attention, Q vem de uma sequência e K/V de outra
- **nn.Linear implementa a projeção**: `nn.Linear(d_model, d_model, bias=False)` é a forma padrão de implementar cada projeção na prática
- **Compartilhar pesos por engano é um bug real**: sempre verifique que W_Q, W_K, W_V têm valores distintos após a inicialização

Próximo capítulo: **Attention Scores** — como Q e K se combinam via produto escalar para medir similaridade entre tokens.

---

**Próximo**: [Capítulo 12: Attention Scores](12_atencao_scores.md)
