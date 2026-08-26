# Guia Extra: Treinando LLMScratch com Dados Reais e Públicos

Este é um guia complementar ao livro, focado 100% em uma coisa: tirar LLMScratch do dataset de brinquedo (4 frases repetidas do Capítulo 33) e treiná-lo com texto real, público e open-source, em volume suficiente para o modelo aprender padrões de linguagem de verdade — vocabulário amplo, gramática, associações semânticas.

**Pré-requisito**: ter completado o Capítulo 33 (o scaffold modular de LLMScratch). Este guia estende esse mesmo projeto — não recomeça do zero.

---

## Por Que Isso Importa

O dataset do Capítulo 33 foi escolhido de propósito para ser pequeno e determinístico: você queria ver o pipeline inteiro rodar em segundos e confirmar que cada peça (tokenizer, modelo, trainer, checkpoint, gerador) se encaixa. Ele cumpriu esse papel. Mas um modelo treinado em 4 frases repetidas 5 vezes não aprende linguagem — ele memoriza 4 frases.

"Consciência" de um modelo de linguagem, no sentido prático, não é nada místico: é a quantidade e diversidade de padrões estatísticos de texto que ele viu durante o treino. Um modelo que só viu "o gato dormia no sofá" nunca vai gerar nada sobre economia, código, ou clima — porque essas distribuições de tokens simplesmente não existem no que ele foi exposto. Mais dados reais = mais padrões = respostas menos repetitivas e mais coerentes.

A boa notícia: você não precisa dos trilhões de tokens que a OpenAI ou a Anthropic usam. Com um modelo pequeno (10-50M parâmetros, como os que este livro constrói), alguns milhões a poucas centenas de milhões de tokens de um dataset público bem escolhido já produzem uma diferença perceptível — de "gera lixo repetitivo" para "gera frases gramaticalmente plausíveis com algum sentido local".

Este guia cobre: onde encontrar esses dados, como baixá-los sem estourar seu disco ou sua RAM, como adaptar o tokenizer e o pipeline de dados do projeto modular para eles, e como ajustar a configuração de treino para um resultado que você realmente sinta diferença ao gerar texto.

---

## Parte 1: Escolhendo um Dataset Público

### Critérios de escolha para um projeto didático/pequeno

| Critério | Por quê importa |
|----------|------------------|
| Licença permissiva (MIT, CC0, CC-BY, ODC-BY) | Você pode usar, redistribuir, treinar sobre ele sem risco legal |
| Tamanho gerenciável (não precisa do dataset inteiro) | Datasets como C4/OSCAR têm centenas de GB — você vai usar uma fatia |
| Qualidade de texto (não HTML cru, não spam) | Texto ruidoso demais desperdiça capacidade do modelo pequeno |
| Disponível via HuggingFace `datasets` | Streaming, cache automático, sem scripts de scraping manuais |

### Datasets recomendados

**Para começar rápido (inglês, texto limpo, ótimo para modelos pequenos):**

- **TinyStories** (`roneneldan/TinyStories`) — histórias infantis curtas geradas para ensinar modelos pequenos a formar frases coerentes. Vocabulário simples, gramática correta. É literalmente desenhado para modelos do tamanho que este livro constrói. **Ponto de partida recomendado.**
- **WikiText-103** (`wikitext`, config `wikitext-103-raw-v1`) — artigos da Wikipedia em inglês, ~100M tokens, texto limpo e bem formatado.

**Para português (conecta diretamente com o idioma deste livro):**

- **Wikipedia em português** (`wikimedia/wikipedia`, config `20231101.pt`) — dump oficial da Wikipedia PT-BR, licença CC-BY-SA.
- **OSCAR (subset pt)** (`oscar-corpus/OSCAR-2301`, config `pt`) — corpus multilíngue extraído do Common Crawl, filtrado por idioma. Grande, então você vai usar streaming e pegar só uma fatia.
- **BrWaC** (Brazilian Web as Corpus) — corpus acadêmico de português brasileiro, ~2.7 bilhões de tokens, disponível via pedido de acesso ao NILC-USP (não é HuggingFace direto, mas vale mencionar como referência de alta qualidade).

**Para escala maior (quando seu modelo crescer):**

- **C4** (`allenai/c4`, config `en` ou `pt`) — Common Crawl limpo, usado para treinar o T5 original. Centenas de GB — sempre use `streaming=True`.
- **The Pile** (`EleutherAI/pile`, ou subsets específicos como `pile-uncopyrighted`) — mistura diversa (livros, código, papers, web). Licenças variam por subset — confira cada um.

**Regra prática**: comece com **TinyStories** (inglês) ou **Wikipedia PT** (português) — ambos pequenos o suficiente para caber em memória, limpos, e com resultado visível em poucas horas de treino em uma GPU/M-series modesta.

---

## Parte 2: Baixando os Dados com a Biblioteca `datasets`

Instale a dependência:

```bash
pip install datasets tokenizers
```

### Opção A: Dataset pequeno, cabe em memória (TinyStories, WikiText-103)

```python
from datasets import load_dataset

# Baixa e cacheia localmente (~/.cache/huggingface/datasets)
ds = load_dataset("roneneldan/TinyStories", split="train")
print(f"Exemplos: {len(ds):,}")
print(f"Primeiro exemplo:\n{ds[0]['text'][:300]}")
```

### Opção B: Dataset grande, usar streaming (OSCAR, C4)

Streaming evita baixar o dataset inteiro — ele baixa e processa em blocos, sob demanda:

```python
from datasets import load_dataset

ds = load_dataset(
    "oscar-corpus/OSCAR-2301",
    "pt",
    split="train",
    streaming=True,
)

# Pegue só os primeiros N exemplos para um dataset local pequeno
textos = []
for i, exemplo in enumerate(ds):
    if i >= 50_000:
        break
    textos.append(exemplo["text"])

print(f"Coletados {len(textos):,} documentos via streaming")
```

**Por que streaming importa aqui**: OSCAR-2301 em português tem dezenas de GB. Sem streaming, o `load_dataset` tentaria baixar tudo antes de você processar uma única linha. Com streaming, você lê exatamente os primeiros 50 mil documentos e para — o resto nunca toca seu disco.

### Salvando uma fatia local para reuso

Depois de coletar sua fatia, salve como texto puro — isso desacopla o resto do pipeline da biblioteca `datasets`:

```python
from pathlib import Path

output_path = Path("data/raw/corpus.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for texto in textos:
        texto_limpo = texto.strip().replace("\n\n", "\n")
        if len(texto_limpo) > 50:  # descarta documentos vazios/curtos demais
            f.write(texto_limpo + "\n")

print(f"Corpus salvo: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
```

---

## Parte 3: Treinando um Tokenizer BPE Real

O `CharTokenizer` do Capítulo 33 foi uma simplificação didática. Com dados reais, você quer um tokenizer BPE de verdade (Capítulo 24 mostrou o algoritmo do zero — agora vamos usar a biblioteca `tokenizers`, que é a mesma usada por HuggingFace Transformers em produção, mas o algoritmo por baixo é o que você já entende).

```python
# scripts/train_tokenizer.py
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_bpe_tokenizer(
    corpus_path: str,
    vocab_size: int = 8000,
    output_path: str = "data/tokenizer.json",
) -> None:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )

    tokenizer.train([corpus_path], trainer)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer treinado e salvo em {output_path}")
    print(f"Vocab size real: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    train_bpe_tokenizer("data/raw/corpus.txt", vocab_size=8000)
```

Rode:

```bash
python scripts/train_tokenizer.py
```

### Integrando ao projeto modular: `tokenizer.py`

No Capítulo 33, `tokenizer.py` continha `CharTokenizer`. Adicione uma segunda classe com a **mesma interface** (`encode`/`decode`) — essa é a vantagem prática de ter isolado o tokenizer em seu próprio arquivo desde o início: trocar a implementação não exige tocar em `model/`, `data/` ou `training/`.

```python
# tokenizer.py (adicionar ao arquivo existente)
from tokenizers import Tokenizer as HFTokenizer


class BPETokenizer:
    """Wrapper fino sobre a biblioteca `tokenizers`, mantendo a mesma
    interface de CharTokenizer (encode/decode) para ser um substituto
    direto em data/dataset.py e inference/generator.py."""

    def __init__(self, tokenizer_path: str = "data/tokenizer.json"):
        self._tok = HFTokenizer.from_file(tokenizer_path)
        self.vocab_size = self._tok.get_vocab_size()
        self.pad_id = self._tok.token_to_id("<pad>")

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = self._tok.encode(text).ids
        if max_len is not None:
            ids = ids[:max_len]
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i != self.pad_id]
        return self._tok.decode(ids)
```

---

## Parte 4: Adaptando `data/dataset.py` para Corpus Real

O `TextDataset` do Capítulo 33 recebia uma lista pequena de strings inteiras na memória. Com um corpus real (mesmo "pequeno" em termos de LLM, ainda são dezenas de milhares de documentos), você quer **chunking**: concatenar o corpus e cortar em blocos de `max_seq_len`, em vez de um exemplo por documento (documentos têm tamanhos muito variados, e isso desperdiça padding).

```python
# data/dataset.py (nova classe, ao lado de TextDataset)
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ChunkedCorpusDataset(Dataset):
    """Dataset para corpus real: concatena todo o texto tokenizado e
    corta em blocos fixos de max_seq_len + 1 (input + target via shift).

    Diferente de TextDataset (Capítulo 33), que trata cada frase como
    um exemplo, aqui o corpus inteiro vira um único stream de tokens
    — mais eficiente para texto de tamanho variável.
    """

    def __init__(self, corpus_path: str, tokenizer, max_len: int = 128):
        self.max_len = max_len

        text = Path(corpus_path).read_text(encoding="utf-8")
        all_ids = tokenizer.encode(text)  # sem max_len: queremos tudo

        # Descarta o resto que não forma um bloco completo
        n_blocks = len(all_ids) // (max_len + 1)
        usable = n_blocks * (max_len + 1)
        self.data = torch.tensor(all_ids[:usable], dtype=torch.long)
        self.data = self.data.view(n_blocks, max_len + 1)

        print(f"Corpus tokenizado: {len(all_ids):,} tokens -> {n_blocks:,} blocos de {max_len}")

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.data[idx]
        return block[:-1], block[1:]
```

**Nota de memória**: para corpus grandes (>1GB de texto), tokenizar tudo em memória de uma vez não escala. A próxima etapa natural é pré-tokenizar em disco em um arquivo binário (`.bin` com `numpy.memmap`) e usar `memmap` no `__getitem__` — mas para os datasets recomendados na Parte 1 usados em fatias de dezenas a centenas de MB, a abordagem acima já funciona bem em uma máquina com 16GB+ de RAM.

---

## Parte 5: Escalando a Configuração

O `LLMScratchConfig` do Capítulo 33 foi dimensionado para rodar em segundos com dados de brinquedo. Para dados reais, os números mudam — mas a classe é a mesma, só os valores:

```python
# config.py — perfil "dados reais, modelo pequeno-médio"
from config import LLMScratchConfig

config_real = LLMScratchConfig(
    vocab_size=8000,        # bate com o vocab_size do tokenizer BPE treinado
    d_model=256,             # de 64 -> 256
    num_layers=6,            # de 2 -> 6
    num_heads=8,              # de 2 -> 8 (d_model/num_heads = 32, ok)
    d_ff=1024,                # 4x d_model, mantendo a proporção
    max_seq_len=256,          # de 128 -> 256 (mais contexto)
    dropout=0.1,
    learning_rate=3e-4,       # menor: modelos maiores toleram menos LR agressivo
    batch_size=16,             # menor batch por causa do maior seq_len/d_model
    num_epochs=3,               # com corpus real, 1 "epoch" já é muito mais dado
    grad_clip_norm=1.0,
    checkpoint_path="checkpoints/llmscratch_real.pt",
)
```

Isso dá aproximadamente **~15-20M parâmetros** — ainda leve o suficiente para treinar em uma GPU de consumidor ou em Apple Silicon (MPS) em algumas horas, mas grande o suficiente para produzir texto com estrutura gramatical reconhecível.

### Tabela de referência: perfis de treino

| Perfil | d_model | num_layers | num_heads | Parâmetros aprox. | Hardware / tempo |
|--------|---------|------------|-----------|---------------------|-------------------|
| Brinquedo (Cap. 33) | 64 | 2 | 2 | ~150K | CPU, segundos |
| Este guia (recomendado) | 256 | 6 | 8 | ~15-20M | GPU/MPS, 2-6h |
| Ambicioso | 512 | 8 | 8 | ~60-80M | GPU dedicada, 1-2 dias |

---

## Parte 6: Ajustando o `Trainer` para Corpus Real

O `Trainer` do Capítulo 33 já está bem desenhado para isso — a mudança principal é **learning rate warmup**, que se torna importante com mais dados e um modelo maior (sem warmup, os primeiros passos com LR alto em pesos ainda quase aleatórios podem desestabilizar o treino).

```python
# training/trainer.py (adicionar ao Trainer existente)
import math

import torch


class Trainer:
    def __init__(self, model, config, device, warmup_steps: int = 500, total_steps: int = 10_000):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=0.01
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0

    def _current_lr(self) -> float:
        """Warmup linear seguido de decaimento cosseno (padrão em treino de LLMs)."""
        if self.step_count < self.warmup_steps:
            return self.config.learning_rate * self.step_count / self.warmup_steps

        progress = (self.step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        for group in self.optimizer.param_groups:
            group["lr"] = self._current_lr()

        # ... resto igual ao Capítulo 33 (forward, loss, backward, clip, step) ...
        self.step_count += 1
        return loss.item()  # placeholder — mantenha o corpo do Capítulo 33 aqui
```

**Por que cosseno e não um decaimento fixo**: o warmup evita choque inicial; o decaimento cosseno reduz o LR suavemente até perto de zero no fim do treino, o que na prática produz convergência mais estável que um LR constante — é o padrão usado desde o GPT-2/GPT-3 e a maioria dos LLMs modernos.

---

## Parte 7: Script de Treino Completo (`scripts/train_real.py`)

Este script é o novo composition root para este cenário — separado de `main.py` (que continua servindo o exemplo didático do Capítulo 33), mantendo a mesma filosofia: constrói objetos, não tem lógica de negócio.

```python
# scripts/train_real.py
import torch
from torch.utils.data import DataLoader, random_split

from config import LLMScratchConfig
from data.dataset import ChunkedCorpusDataset
from model.llmscratch import LLMScratch
from tokenizer import BPETokenizer
from training.trainer import Trainer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    tokenizer = BPETokenizer("data/tokenizer.json")

    config = LLMScratchConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        max_seq_len=256,
        learning_rate=3e-4,
        batch_size=16,
        num_epochs=3,
        checkpoint_path="checkpoints/llmscratch_real.pt",
    )

    dataset = ChunkedCorpusDataset("data/raw/corpus.txt", tokenizer, max_len=config.max_seq_len)

    # Split treino/validação — essencial com dados reais (Capítulo 30)
    val_size = max(1, int(0.02 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = LLMScratch(config)
    print(f"Modelo criado com {model.num_parameters():,} parâmetros")

    total_steps = len(train_loader) * config.num_epochs
    trainer = Trainer(model, config, device, warmup_steps=500, total_steps=total_steps)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
```

Rode:

```bash
python scripts/train_real.py
```

Saída esperada (números variam conforme o dataset escolhido):

```
Device: mps
Corpus tokenizado: 8,432,109 tokens -> 32,938 blocos de 256
Modelo criado com 16,847,360 parâmetros
Epoch 1/3: train=6.21 val=5.89
Epoch 2/3: train=4.73 val=4.58
Epoch 3/3: train=3.95 val=3.91
```

Uma loss caindo de ~6-7 (próxima de `ln(vocab_size)`, ou seja, "chutando aleatoriamente") para 3-4 já indica que o modelo aprendeu estrutura real de linguagem — não é um LLM fluente, mas está muito além do dataset de brinquedo do Capítulo 33.

---

## Parte 8: Avaliando o Resultado

### Perplexidade

Reaproveite o `Evaluator` do Exercício 33.5:

```python
from training.evaluator import Evaluator

evaluator = Evaluator(model, device)
ppl = evaluator.perplexity(val_loader)
print(f"Perplexidade em validação: {ppl:.2f}")
```

Como referência intuitiva: perplexidade próxima de `vocab_size` significa "aleatório"; abaixo de ~50-100 (para um modelo pequeno em um corpus modesto) já indica aprendizado real de padrões locais de linguagem.

### Geração qualitativa

```python
from inference.generator import Generator

generator = Generator(model, tokenizer, device)
for prompt in ["Era uma vez", "O governo anunciou", "A inteligência artificial"]:
    texto = generator.generate(prompt, max_new_tokens=60, temperature=0.8, top_k=40)
    print(f"\n>>> {prompt}\n{texto}")
```

Compare a coerência local (concordância de gênero/número, estrutura de frase) antes e depois de treinar com dados reais — essa comparação lado a lado é a evidência mais direta de que o "conhecimento" do modelo cresceu.

---

## Erros Comuns

### Erro 1: Treinar sem `streaming=True` em datasets grandes

```python
# Trava a máquina ou enche o disco tentando baixar o C4 inteiro (>800GB)
ds = load_dataset("allenai/c4", "pt", split="train")

# Correto: streaming + limite explícito
ds = load_dataset("allenai/c4", "pt", split="train", streaming=True)
```

### Erro 2: Vocab size do tokenizer diferente do `config.vocab_size`

```python
# Se o tokenizer BPE foi treinado com vocab_size=8000 mas o config diz 1000,
# o modelo trava com "index out of range" na embedding table.
config = LLMScratchConfig(vocab_size=1000)  # ERRADO se o tokenizer tem 8000

# Correto: sempre derive do tokenizer treinado
config = LLMScratchConfig(vocab_size=tokenizer.vocab_size)
```

### Erro 3: Ignorar licença do dataset

Nem todo corpus público é livre para qualquer uso. `The Pile` mistura subsets com licenças diferentes; alguns dumps de Common Crawl incluem conteúdo protegido por copyright que passou pelo filtro. Antes de treinar (e principalmente antes de distribuir um modelo treinado), confira a licença específica do dataset e do subset que você usou.

### Erro 4: Não fazer split de validação

```python
# Treina com o corpus inteiro, sem forma de detectar overfitting (Capítulo 30)
dataset = ChunkedCorpusDataset(...)
loader = DataLoader(dataset, ...)  # sem holdout

# Correto: sempre reserve uma fatia de validação
train_ds, val_ds = random_split(dataset, [...])
```

---

## Checklist Prático

1. Escolher um dataset da Parte 1 (recomendado: TinyStories ou Wikipedia PT)
2. Baixar/coletar uma fatia com `datasets` (streaming se necessário) e salvar como `.txt`
3. Treinar um tokenizer BPE (`scripts/train_tokenizer.py`)
4. Adicionar `BPETokenizer` a `tokenizer.py` e `ChunkedCorpusDataset` a `data/dataset.py`
5. Ajustar `config.py` com um perfil maior (Parte 5)
6. Adicionar warmup + cosine decay ao `Trainer` (Parte 6)
7. Rodar `scripts/train_real.py`, acompanhar loss de treino e validação
8. Avaliar com perplexidade e geração qualitativa (Parte 8)

---

## Próximos Passos

Depois deste guia, o caminho natural é o **Capítulo 34** do livro (distributed training, mixed precision, gradient accumulation) para escalar ainda mais — as técnicas de lá se aplicam diretamente sobre o pipeline construído aqui, já que ele segue exatamente a mesma arquitetura modular do Capítulo 33.

---

**Ver também**:
[Capítulo 24: Tokenização](chapters/24_tokenizacao.md) ·
[Capítulo 29: Treinamento Autoregressivo](chapters/29_treinamento_autoregressivo.md) ·
[Capítulo 30: Checkpoints e Avaliação](chapters/30_checkpoints_avaliacao.md) ·
[Capítulo 33: Projeto LLMScratch](chapters/33_projeto_james_jr.md) ·
[Capítulo 34: Além de LLMScratch](chapters/34_beyond.md)
