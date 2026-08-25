# Capítulo 01: Introdução e Motivação

## 🎯 Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender o que é uma Language Model (LM) em termos simples
2. Saber por que estudar LLMs do zero é importante
3. Conhecer a história breve que levou aos Transformers modernos
4. Ver o "mapa mental" completo do que vamos construir

---

## 💡 Intuição

Uma **Language Model** é uma máquina que aprendeu padrões em linguagem. Dado uma sequência de palavras, ela prediz qual é a próxima palavra mais provável.

Por exemplo:
- Entrada: "O gato subiu no..."
- Saída: distribuição de probabilidades para próxima palavra
- Predição mais provável: "telhado", "árvore", "sofá" (em ordem de probabilidade)

Essa habilidade simples — prever o próximo token — é surpreendentemente poderosa. Quando você treina em bilhões de palavras, o modelo aprende não apenas padrões linguísticos, mas conceitos, raciocínio, e até conhecimento de fatos.

---

## 📖 Breve História

### A Pré-história: Modelos Antigos

**n-gramas** (1990s): Contavam a frequência de sequências de n palavras. "O gato" → probabilidade do que vem depois.
- Simples, mas sem real compreensão de contexto longínquo.

**Redes Neurais Recorrentes (RNNs)** (2000s-2010s):
- Processavam sequências passo a passo (token por token).
- Memória em um estado oculto que "passava adiante".
- **Problema**: memória curta efetiva, difícil de treinar com backpropagation através do tempo (BPTT).

**LSTMs e GRUs** (2010s):
- Mecanismos de "porta" para controlar fluxo de informação.
- Melhor memória de longo termo.
- Ainda linear na velocidade de processamento (tokens sequencialmente).

### O Nascimento do Transformer (2017)

Artigo: *"Attention is All You Need"* (Vaswani et al., 2017).

**Inovação crucial**: Mecanismo de **Atenção** (Attention).
- Ao invés de processar sequência token-por-token, processa *todo o contexto em paralelo*.
- Cada token pode "olhar para" qualquer outro token na sequência.
- Escalável com GPUs modernas.

**Resultado**: Modelos maiores, treinamento mais rápido, melhor qualidade.

### Explosão de Escala (2018-2024)

- **BERT** (2018): Pré-treinamento bidirecional em massa.
- **GPT** (2018): Pré-treinamento autoregressivo (esquerda para direita).
- **GPT-2** (2019): 1.5B parâmetros. Surpreendentemente bom em tarefas não treinadas.
- **GPT-3** (2020): 175B parâmetros. Few-shot learning. "Wow" da comunidade.
- **GPT-4** (2023): Multimodal, ainda melhor raciocínio.
- **Llama, Claude, Gemini** (2023-2024): Modelos open ou proprietary de alta qualidade.

**Padrão observado**: Quanto maior o modelo e mais dados, melhor a performance. (Lei de escala empírica).

---

## 🏗️ O que Vamos Construir

Neste livro, você construirá uma LLM completa do zero. Aqui está a progressão:

### Fase 1: Fundamentos (Capítulos 01-05)
- Tensores e PyTorch: A ferramenta básica.
- Álgebra linear: O idioma da IA.
- Operações: Broadcasting, shapes, gradientes.

### Fase 2: Blocos Básicos (Capítulos 06-15)
- Embeddings: Como representar palavras em números.
- Projeções lineares: Transformações.
- **Atenção**: O mecanismo-chave.

### Fase 3: Arquitetura Completa (Capítulos 16-25)
- Multi-head attention: Múltiplas perspectivas paralelas.
- Transformer block: A unidade de construção.
- Tokenização e posicionamento: Preparar entrada.

### Fase 4: Treinamento (Capítulos 26-30)
- Loss (cross-entropy) e otimizadores (Adam).
- Loop de treinamento autoregressivo.
- Debugging e avaliação.

### Fase 5: Geração (Capítulos 31-32)
- Sampling, temperature, top-k.
- Completação de texto.

### Fase 6: Projeto James Jr. (Capítulos 33-34)
- Modelo completo, treinado, salvável e carregável.
- Geração de texto fim a fim.

---

## 🧠 Filosofia: Entender Antes de Abstrair

Este livro segue uma regra de ouro:

> **Implemente explicitamente com tensores ANTES de usar abstrações.**

Exemplo:
- Capítulos 06-09: Implementamos embedding e projeções com `.matmul()` direto.
- Capítulo 16: Depois envolvemos em uma classe, entender o que ela faz.
- Capítulo 33: Usamos `nn.Linear` confiadamente porque *já sabemos o que ela faz*.

Por que? Porque **abstração sem compreensão é mágica, não entendimento**. 

Quando você vê `nn.MultiheadAttention` funcionar depois de implementar manualmente, você não apenas sabe *como* funciona — você sente na carne o que está acontecendo nos números.

---

## 📊 Exemplos Recorrentes

Ao longo do livro, usaremos o mesmo pequeno exemplo muitas vezes:

**Sequência de entrada**: "O gato"
- Tokenizado: [tok_O, tok_gato]
- Tensor de embeddings: Shape [1, 2, 4]
  - Batch size 1
  - Comprimento de sequência 2
  - Embedding dim 4

**Pesos de atenção**: Shape [4, 4]
- Porque queremos projetar embedding de dim 4 para dim 4

Esses pequenos números fazem tudo ficar legível. Você verá todos os valores.

---

## 🛠️ Implementação Mínima: Um Toy Model

Vamos ver (muito brevemente) como uma LLM *completa* se vê:

```python
import torch
import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        x = self.embed(input_ids)  # [batch, seq_len, d_model]
        
        for layer in self.transformer:
            x = layer(x)  # [batch, seq_len, d_model]
        
        logits = self.head(x)  # [batch, seq_len, vocab_size]
        return logits

# Uso
model = TinyLM(vocab_size=1000, d_model=64, num_layers=2, num_heads=4)
input_ids = torch.randint(0, 1000, (2, 10))  # batch de 2, seq len 10
logits = model(input_ids)  # [2, 10, 1000]
```

Isso é uma LLM válida, funcional. Mas **você não entende o que cada operação faz**.

Neste livro, você entenderá. Implementará cada linha você mesmo. E quando usar `nn.Linear`, saberá exatamente o que está acontecendo.

---

## 🔍 Erros Comuns ao Começar

1. **"Não preciso de álgebra linear, só quero usar PyTorch"**
   - ❌ Erro. Toda operação em deep learning é operação linear (ou ativação).
   - ✅ Invista 2-3 horas em álgebra linear agora, economize semanas depois.

2. **"Vou pular para usar `transformers` library"**
   - ❌ Você perderá a chance de entender o que os Transformers *realmente* fazem.
   - ✅ Faça do zero primeiro. Use a library depois, com confiança.

3. **"Shapes de tensores são fáceis"**
   - ❌ Novamente, erro. 80% dos bugs em deep learning são de shape errado.
   - ✅ Neste livro, você verá e debugará shapes constantemente.

4. **"LLMs são magic, não posso entender"**
   - ❌ Mentira. É matemática + código, ambos entendíveis.
   - ✅ Se você chegou aqui, você consegue. Paciência.

---

## ✍️ Exercícios

### Exercício 1.1: Histórico Pessoal
Escreva 3 frases sobre um modelo de linguagem que você usou (ChatGPT, Claude, etc.). O que você acha que está acontecendo "por trás" quando ele responde uma pergunta?

**Dica**: Não precisa estar certo. Trata-se de ancorar o que você já sabe.

### Exercício 1.2: Predição Next-Token
Dada a sequência: "Inteligência Artificial é...", liste 5 palavras prováveis como próximo token. 

Qual você acha que um modelo bem treinado teria como TOP-1 (mais provável)?

### Exercício 1.3: Tamanho de Modelo
Pesquise os tamanhos de 3 modelos (GPT-3, Llama 2, Claude): quantos bilhões de parâmetros? Escreva em seu caderno.

---

## 📚 Gabarito e Reflexão

### Exercício 1.1: Histórico Pessoal
Não há resposta única. Exemplos válidos:
- "Digitei uma pergunta e ele respondeu. Acho que procurou em um banco de dados."
- "Deve usar padrões de muitos textos para gerar novo texto."

O ponto é que você está fazendo uma hipótese. Neste livro, confirmaremos se ela está perto da realidade.

### Exercício 1.2: Predição Next-Token
Palavras prováveis:
- "uma", "o", "a", "fundamental", "importante"

GPT-3 provavelmente teria TOP-1 = "uma" ou "o" (artigos são frequentes após verbo-ser).

### Exercício 1.3: Tamanho de Modelo
- GPT-3: 175B parâmetros
- Llama 2 (70B): 70B parâmetros  
- Claude 3: ~200B+ (Anthropic não divulga exato)

Perspectiva: Um parâmetro ≈ um "número que o modelo aprendeu". 175 bilhões é muito. Seu cérebro tem ~86 bilhões neurônios, mas modelos de IA não são neurônios 1:1.

---

## 🎓 Resumo

- Uma **Language Model** aprende a prever o próximo token em uma sequência.
- **Transformers** (2017) revolucionaram por permitir paralelização via atenção.
- Nós vamos **construir do zero**, entendendo cada camada.
- **Filosofia**: Implementação explícita antes de abstrações.
- Prepare-se para matemática, código e debugging.

No próximo capítulo: configuramos nosso ambiente PyTorch e testamos instalação.

---

**Próximo**: [Capítulo 02: Setup do Ambiente](02_setup_ambiente.md)
