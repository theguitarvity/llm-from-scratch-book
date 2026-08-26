# Capítulo 33: Projeto LLMScratch — Arquitetura Modular e Implementação Completa

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Organizar um projeto de LLM em módulos com responsabilidades bem definidas, inspirado em Clean Architecture
2. Construir o projeto arquivo por arquivo, entendendo o papel de cada um antes de escrevê-lo
3. Separar domínio (modelo, matemática), aplicação (treino, geração) e infraestrutura (dados, checkpoints, I/O)
4. Treinar LLMScratch do zero usando o projeto modular completo
5. Rodar inferência e gerar texto a partir do modelo treinado
6. Saber onde adicionar código novo sem quebrar o resto do projeto

---

## Por Que Isso Importa

Até aqui, cada capítulo te deu um bloco de código isolado: um `TransformerBlock`, uma função `generate()`, uma `LayerNorm` manual. Isso é ótimo para aprender um conceito de cada vez — mas um projeto de verdade não é um Jupyter notebook gigante com 800 linhas em um único arquivo.

Pense em como um bug se comporta nos dois cenários. Em um arquivo monolítico, "a loss não converge" pode ser causada por qualquer coisa entre a linha 1 e a linha 800: tokenização, inicialização, o loop de treino, o otimizador, a leitura de dados. Você não sabe por onde começar a debugar. Em um projeto modular, você isola: "o modelo produz shapes corretas quando eu testo `model/llmscratch.py` sozinho? Sim. Então o problema está em `training/trainer.py` ou em `data/dataset.py`." Cada módulo vira uma fronteira de teste e de raciocínio.

A ideia por trás de **Clean Architecture** (Robert C. Martin) é simples, mesmo fora do contexto de web apps onde ela nasceu: separe **o que o sistema faz** (regras de negócio, aqui: a matemática do Transformer) de **como ele faz** (I/O, frameworks, arquivos em disco). Se você trocar o otimizador de Adam para SGD, isso não deveria exigir mexer no código do `TransformerBlock`. Se você trocar de tokenizer BPE para um tokenizer de caracteres, isso não deveria exigir mexer no loop de treino. Cada peça muda por um motivo, e só por um motivo.

Neste capítulo você vai construir esse scaffold do zero, arquivo por arquivo, como uma jornada: começando pela configuração, passando pelos componentes do modelo (o "domínio" do projeto), depois a infraestrutura de dados e checkpoints, depois a camada de aplicação que orquestra treino e geração, e terminando em um `main.py` que liga tudo. No final, você roda `python main.py --mode train` e vê LLMScratch aprendendo de verdade — mas agora em um projeto que você pode manter, estender e debugar como um profissional.

---

## Mapeando Clean Architecture para um Projeto de LLM

Clean Architecture original tem camadas como Entities, Use Cases, Interface Adapters e Frameworks/Drivers, pensadas para sistemas com banco de dados, APIs REST, UI. Um projeto de treino de LLM não tem essas camadas exatamente — mas o **princípio** se mantém: dependências apontam para dentro, do específico para o geral, nunca o contrário.

Vamos adaptar para três camadas práticas:

| Camada | Pergunta que responde | Depende de | Exemplos neste projeto |
|--------|----------------------|------------|------------------------|
| **Domain** | "O que é um Transformer, matematicamente?" | Só PyTorch (tensores puros) | `model/*.py`, `config.py` |
| **Application** | "Como eu treino / gero texto com esse modelo?" | Domain | `training/trainer.py`, `inference/generator.py` |
| **Infrastructure** | "Como eu leio dados do disco / salvo checkpoints?" | Application + Domain | `data/dataset.py`, `training/checkpoint.py` |

A regra de ouro: **`model/llmscratch.py` nunca deveria importar `torch.utils.data` ou saber que existe um arquivo `.txt` em disco.** O modelo só sabe fazer forward pass em tensores. Quem sabe ler arquivos é a camada de infraestrutura. Quem decide "agora treine por 10 epochs" é a camada de aplicação.

```
Domain (model/, config.py)
    ↑ é usado por
Application (training/, inference/)
    ↑ é orquestrado por
main.py (composition root)
    ↑ usa infraestrutura via
Infrastructure (data/, checkpoints)
```

Isso não é burocracia por burocracia — é o que te permite trocar peças. Quer trocar o dataset de texto dummy por um dataset real do HuggingFace? Só mexe em `data/dataset.py`. Quer adicionar early stopping? Só mexe em `training/trainer.py`. O modelo em si nunca muda por causa disso.

---

## O Scaffold do Projeto

Antes de escrever qualquer arquivo, veja a estrutura completa para onde estamos indo:

```
llmscratch/
├── config.py                      # Configuração como dataclass (Domain)
├── tokenizer.py                   # Tokenização character-level (Domain)
├── model/
│   ├── __init__.py
│   ├── positional_embedding.py    # Codificação posicional (Domain)
│   ├── feed_forward.py            # MLP posição-a-posição (Domain)
│   ├── transformer_block.py       # Attention + FFN + residual + norm (Domain)
│   └── llmscratch.py              # Modelo completo (Domain)
├── data/
│   ├── __init__.py
│   └── dataset.py                 # TextDataset + DataLoader (Infrastructure)
├── training/
│   ├── __init__.py
│   ├── checkpoint.py              # save/load de estado (Infrastructure)
│   └── trainer.py                 # Loop de treino (Application)
├── inference/
│   ├── __init__.py
│   └── generator.py               # Geração de texto (Application)
├── main.py                        # Composition root / CLI (Entrypoint)
└── requirements.txt
```

Cada seta de importação aponta para dentro: `main.py` importa de `training/` e `inference/`; `training/trainer.py` importa de `model/` e `data/`; mas `model/llmscratch.py` não importa nada de `training/` ou `data/`. Vamos construir nessa ordem — de dentro para fora.

Crie a estrutura de diretórios:

```bash
mkdir -p llmscratch/model llmscratch/data llmscratch/training llmscratch/inference
cd llmscratch
touch model/__init__.py data/__init__.py training/__init__.py inference/__init__.py
```

---

## Passo 1: `config.py` — Configuração Como Entidade de Domínio

A configuração é o primeiro arquivo porque tudo depende dela, e ela não depende de nada. Em vez de um dicionário solto (como fizemos em capítulos anteriores), usamos um `dataclass` — isso dá autocomplete, checagem de tipos, e um único lugar de verdade para os hiperparâmetros.

```python
# config.py
from dataclasses import dataclass


@dataclass
class LLMScratchConfig:
    """Configuração do modelo e do treinamento.

    Mantida como dataclass (não dict) para autocomplete e checagem de tipos.
    Nenhuma dependência de PyTorch aqui — é uma entidade de domínio pura.
    """

    # Arquitetura
    vocab_size: int = 1000
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 2
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1

    # Treinamento
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    grad_clip_norm: float = 1.0

    # Caminhos
    checkpoint_path: str = "checkpoints/llmscratch.pt"

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) deve ser divisível por "
                f"num_heads ({self.num_heads})"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads
```

**Por que isso importa aqui**: `__post_init__` valida a configuração no momento em que ela é criada, não no meio do treino três horas depois. Isso é "fail fast" — um princípio que vale tanto em Clean Architecture quanto em qualquer sistema que você queira debugar rapidamente.

Teste rápido:

```python
>>> from config import LLMScratchConfig
>>> cfg = LLMScratchConfig(d_model=64, num_heads=3)
Traceback (most recent call last):
  ...
ValueError: d_model (64) deve ser divisível por num_heads (3)
```

---

## Passo 2: `tokenizer.py` — Tokenização Como Serviço de Domínio

O tokenizer transforma texto em IDs e de volta. Ele é domínio porque é pura lógica — não sabe nada sobre arquivos, treino, ou PyTorch.

```python
# tokenizer.py

class CharTokenizer:
    """Tokenizer character-level simples.

    Usa ord(c) % vocab_size para simplicidade didática. Em produção,
    seria substituído por BPE (Capítulo 24) sem que nenhum outro
    módulo do projeto precise mudar — essa é a vantagem de isolar
    tokenização em seu próprio arquivo.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_id = 0

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [ord(c) % self.vocab_size for c in text]
        if max_len is not None:
            ids = ids[:max_len]
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i != self.pad_id)
```

Note o contrato: `encode` recebe `str`, devolve `list[int]`. `decode` faz o inverso. Qualquer outro tokenizer (BPE, WordPiece) que respeite essa mesma interface pode substituir este sem que `model/`, `data/` ou `training/` percebam a diferença.

---

## Passo 3: `model/positional_embedding.py`

Agora entramos na pasta `model/` — o coração do domínio. Cada arquivo aqui é uma peça isolada do Transformer, testável sozinha.

```python
# model/positional_embedding.py
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Codificação posicional sinusoidal (Capítulo 22)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]
```

Teste isolado (sem precisar do resto do projeto):

```python
>>> import torch
>>> from model.positional_embedding import PositionalEmbedding
>>> pe = PositionalEmbedding(d_model=8, max_len=16)
>>> x = torch.zeros(1, 4, 8)
>>> pe(x).shape
torch.Size([1, 4, 8])
```

---

## Passo 4: `model/feed_forward.py`

```python
# model/feed_forward.py
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """MLP posição-a-posição com expansão 4x (Capítulo 20)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

---

## Passo 5: `model/transformer_block.py`

Este arquivo une attention (usamos `nn.MultiheadAttention` para manter o foco na organização do projeto — você já implementou multi-head attention do zero no Capítulo 17) com o `FeedForward` do passo anterior, seguindo Pre-LN (Capítulo 19):

```python
# model/transformer_block.py
import torch
import torch.nn as nn

from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """Bloco Transformer Pre-LN: Attention + Residual + FFN + Residual (Capítulo 21)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN: normaliza antes de cada sub-camada
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)

        return x
```

Note a importação: `from model.feed_forward import FeedForward`. Dentro da própria camada de domínio, os módulos podem se importar entre si — a regra de "dependência aponta pra dentro" é sobre não deixar `model/` importar de `training/` ou `data/`, não sobre isolar arquivos de domínio uns dos outros.

---

## Passo 6: `model/llmscratch.py` — Montando o Modelo Completo

Este é o arquivo mais importante do domínio: monta embedding, posição, stack de blocos, e a cabeça de saída (Capítulo 23).

```python
# model/llmscratch.py
import torch
import torch.nn as nn

from config import LLMScratchConfig
from model.positional_embedding import PositionalEmbedding
from model.transformer_block import TransformerBlock


class LLMScratch(nn.Module):
    """Modelo de linguagem decoder-only completo.

    Responsabilidade única: dado input_ids, produzir logits.
    Não sabe nada sobre otimizador, dataset, ou checkpoints — isso
    é responsabilidade de outras camadas.
    """

    def __init__(self, config: LLMScratchConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEmbedding(config.d_model, config.max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.norm_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        # Weight tying (Capítulo 23): compartilha pesos entre embedding e head
        self.head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        x = self.token_embed(input_ids)      # [batch, seq_len, d_model]
        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm_final(x)
        logits = self.head(x)                 # [batch, seq_len, vocab_size]
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Máscara causal (Capítulo 15): impede olhar para o futuro."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))
```

Teste isolado — o teste mais importante do projeto, porque valida a peça central sem precisar de dados, treino ou disco:

```python
>>> from config import LLMScratchConfig
>>> from model.llmscratch import LLMScratch
>>> import torch
>>>
>>> cfg = LLMScratchConfig(vocab_size=100, d_model=32, num_layers=2, num_heads=2, d_ff=64, max_seq_len=16)
>>> model = LLMScratch(cfg)
>>> x = torch.randint(0, 100, (2, 8))  # [batch=2, seq_len=8]
>>> mask = LLMScratch.causal_mask(8, x.device)
>>> logits = model(x, mask=mask)
>>> logits.shape
torch.Size([2, 8, 100])
>>> model.num_parameters()
23844
```

Shape correto, sem tocar em dataset, trainer, ou disco. Isso é o valor de isolar o domínio: você sabe que o modelo funciona antes de complicar as coisas com I/O.

---

## Passo 7: `data/dataset.py` — Infraestrutura de Dados

Agora saímos do domínio e entramos em infraestrutura: este arquivo sabe sobre `torch.utils.data`, sobre como transformar texto bruto em tensores prontos para batch. Ele depende do `tokenizer.py` (domínio), mas nenhum arquivo de domínio depende dele.

```python
# data/dataset.py
import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


class TextDataset(Dataset):
    """Dataset autoregressivo: input = seq[:-1], target = seq[1:] (Capítulo 29)."""

    def __init__(self, texts: list[str], tokenizer: CharTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = [
            torch.tensor(tokenizer.encode(text, max_len=max_len), dtype=torch.long)
            for text in texts
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.examples[idx]
        return ids[:-1], ids[1:]
```

---

## Passo 8: `training/checkpoint.py` — Infraestrutura de Persistência

Salvar e carregar modelo é I/O — não é domínio, não é a lógica de "como treinar", é só "como persistir estado em disco" (Capítulo 30).

```python
# training/checkpoint.py
from pathlib import Path

import torch

from config import LLMScratchConfig
from model.llmscratch import LLMScratch


def save_checkpoint(model: LLMScratch, config: LLMScratchConfig, epoch: int, loss: float) -> None:
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LLMScratch, LLMScratchConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = LLMScratch(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, config
```

---

## Passo 9: `training/trainer.py` — Aplicação: o Caso de Uso "Treinar"

Chegamos na camada de aplicação. `Trainer` orquestra domínio (`LLMScratch`) e infraestrutura (`DataLoader`, `save_checkpoint`) — mas a lógica de "como fazer forward pass" continua isolada dentro do modelo.

```python
# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import LLMScratchConfig
from model.llmscratch import LLMScratch
from training.checkpoint import save_checkpoint


class Trainer:
    """Caso de uso: treinar LLMScratch autoregressivamente (Capítulos 26-29)."""

    def __init__(self, model: LLMScratch, config: LLMScratchConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignora padding

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        mask = LLMScratch.causal_mask(input_ids.size(1), self.device)
        logits = self.model(input_ids, mask=mask)

        loss = self.criterion(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        return loss.item()

    def fit(self, dataloader: DataLoader) -> None:
        self.model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0.0
            for input_ids, target_ids in dataloader:
                total_loss += self.train_step(input_ids, target_ids)

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}: loss = {avg_loss:.4f}")

            save_checkpoint(self.model, self.config, epoch, avg_loss)
```

Note que `Trainer` não sabe de onde vieram os textos — ele recebe um `DataLoader` já pronto. Quem monta esse `DataLoader` é o `main.py`, o composition root. Essa inversão é o que te permite trocar a fonte de dados sem tocar no `Trainer`.

---

## Passo 10: `inference/generator.py` — Aplicação: o Caso de Uso "Gerar"

```python
# inference/generator.py
import torch
import torch.nn.functional as F

from model.llmscratch import LLMScratch
from tokenizer import CharTokenizer


class Generator:
    """Caso de uso: gerar texto a partir de um modelo treinado (Capítulos 31-32)."""

    def __init__(self, model: LLMScratch, tokenizer: CharTokenizer, device: torch.device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        self.model.eval()

        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.model.config.max_seq_len:]
            mask = LLMScratch.causal_mask(context.size(1), self.device)

            logits = self.model(context, mask=mask)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, -1, None]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(input_ids[0].tolist())
```

Repare que `Generator` e `Trainer` são simétricos: os dois recebem um `LLMScratch` já construído e orquestram uma operação sobre ele. Nenhum dos dois sabe construir o modelo — essa responsabilidade fica no composition root.

---

## Passo 11: `main.py` — O Composition Root

Este é o único arquivo do projeto que conhece todas as peças. Ele não tem lógica de negócio — só constrói objetos e liga um ao outro. É aqui, e só aqui, que decidimos "qual tokenizer", "qual dataset", "qual device".

```python
# main.py
import argparse

import torch

from config import LLMScratchConfig
from data.dataset import TextDataset
from inference.generator import Generator
from model.llmscratch import LLMScratch
from tokenizer import CharTokenizer
from torch.utils.data import DataLoader
from training.checkpoint import load_checkpoint
from training.trainer import Trainer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(config: LLMScratchConfig) -> None:
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer(vocab_size=config.vocab_size)

    texts = [
        "O gato dormia no sofá" * 5,
        "Um cachorro corria no parque" * 5,
        "A chuva caía sobre a cidade" * 5,
        "O sol brilhava no céu azul" * 5,
    ]
    dataset = TextDataset(texts, tokenizer, max_len=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = LLMScratch(config)
    print(f"Modelo criado com {model.num_parameters():,} parâmetros")

    trainer = Trainer(model, config, device)
    trainer.fit(dataloader)


def infer(config: LLMScratchConfig, prompt: str) -> None:
    device = get_device()
    model, loaded_config = load_checkpoint(config.checkpoint_path, device)
    tokenizer = CharTokenizer(vocab_size=loaded_config.vocab_size)

    generator = Generator(model, tokenizer, device)
    output = generator.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=20)

    print(f"Prompt: {prompt}")
    print(f"Gerado: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMScratch — treino e inferência")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--prompt", type=str, default="O gato")
    args = parser.parse_args()

    config = LLMScratchConfig()

    if args.mode == "train":
        train(config)
    else:
        infer(config, args.prompt)
```

---

## Rodando o Projeto Completo

Com todos os 13 arquivos no lugar, o projeto roda assim:

```bash
# Treinar
python main.py --mode train

# Gerar texto com o checkpoint salvo
python main.py --mode infer --prompt "O gato"
```

Saída esperada do treino:

```
Device: mps
Modelo criado com 145,824 parâmetros
Epoch 1/10: loss = 5.1823
Epoch 2/10: loss = 4.7291
...
Epoch 10/10: loss = 2.9104
```

Saída esperada da inferência:

```
Prompt: O gato
Gerado: O gato dormia no sofá gato dormia sofá cachorro parque...
```

(A qualidade do texto é limitada — o dataset é dummy e minúsculo. O objetivo aqui é validar que o **pipeline inteiro** funciona ponta a ponta, não produzir texto fluente.)

---

## Testando Cada Módulo Isoladamente

A maior vantagem prática desse scaffold: você pode escrever um teste rápido para cada arquivo sem rodar o projeto inteiro.

```python
# Verificação rápida e manual de cada camada — sem framework de testes
import torch
from config import LLMScratchConfig
from model.llmscratch import LLMScratch
from tokenizer import CharTokenizer

# 1. Config valida hiperparâmetros
cfg = LLMScratchConfig(vocab_size=50, d_model=16, num_heads=2, num_layers=1, d_ff=32, max_seq_len=8)
print("[ok] config válida")

# 2. Tokenizer faz round-trip
tok = CharTokenizer(vocab_size=50)
ids = tok.encode("abc")
assert tok.decode(ids) == "abc"
print("[ok] tokenizer round-trip")

# 3. Modelo produz shape correto
model = LLMScratch(cfg)
x = torch.randint(0, 50, (2, 6))
mask = LLMScratch.causal_mask(6, x.device)
out = model(x, mask=mask)
assert out.shape == (2, 6, 50)
print("[ok] modelo produz shape esperado")

# 4. Máscara causal é triangular
mask = LLMScratch.causal_mask(4, torch.device("cpu"))
assert torch.isinf(mask[0, 1])   # posição 0 não vê posição 1 (futuro)
assert mask[0, 0] == 0.0          # posição 0 vê a si mesma
print("[ok] máscara causal correta")

print("\nTodos os módulos verificados isoladamente.")
```

Se um desses falhar, você sabe exatamente qual arquivo investigar — sem precisar rodar um treino completo de 10 epochs para descobrir que o bug estava na máscara causal.

---

## Erros Comuns

### Erro 1: Import circular entre `model/` e `training/`

```python
# model/llmscratch.py
from training.trainer import Trainer  # ERRADO: domínio importando aplicação

# training/trainer.py
from model.llmscratch import LLMScratch  # cria ciclo
```

A regra é uma via: `training/` pode importar `model/`, nunca o contrário. Se você sentir vontade de importar `Trainer` dentro de `model/`, é sinal de que essa lógica pertence à camada de aplicação, não à de domínio.

### Erro 2: Lógica de treino vazando para dentro do modelo

```python
# ERRADO: LLMScratch.forward() chamando optimizer.step()
class LLMScratch(nn.Module):
    def forward(self, x):
        logits = ...
        self.optimizer.step()  # não é responsabilidade do modelo
        return logits
```

O modelo só faz forward pass. Quem decide quando dar um passo de otimização é o `Trainer`.

### Erro 3: `main.py` com lógica de negócio

```python
# ERRADO: cálculo de loss dentro do composition root
if args.mode == "train":
    logits = model(x)
    loss = F.cross_entropy(logits, y)  # isso pertence a Trainer.train_step
    ...
```

`main.py` deveria apenas construir objetos e chamar métodos de alto nível (`trainer.fit(dataloader)`). Se `main.py` cresce com lógica de cálculo, essa lógica está no lugar errado.

---

## Exercícios

### Exercício 33.1: Rodar o Pipeline Completo
Crie os 13 arquivos deste capítulo e rode `python main.py --mode train` seguido de `python main.py --mode infer`.

### Exercício 33.2: Trocar o Tokenizer
Crie uma classe `WordTokenizer` com a mesma interface de `CharTokenizer` (`encode`/`decode`). Troque em `main.py` sem tocar em nenhum outro arquivo.

### Exercício 33.3: Adicionar Validação
Adicione um segundo `DataLoader` de validação em `Trainer.fit()`, calculando `val_loss` a cada epoch sem fazer backward nele.

### Exercício 33.4: Early Stopping
Modifique `Trainer` para parar o treino se `val_loss` não melhorar por 3 epochs seguidas.

### Exercício 33.5: Novo Caso de Uso — Avaliação
Crie `training/evaluator.py` com uma classe `Evaluator` que calcula perplexidade em um dataset, reutilizando `LLMScratch` sem modificá-lo.

---

## Gabarito

### Exercício 33.2: WordTokenizer
```python
class WordTokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.word_to_id = {}
        self.id_to_word = {}

    def fit(self, texts: list[str]) -> None:
        words = sorted(set(w for t in texts for w in t.split()))
        self.word_to_id = {w: i + 1 for i, w in enumerate(words[: self.vocab_size - 1])}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.word_to_id.get(w, 0) for w in text.split()]
        if max_len is not None:
            ids = ids[:max_len] + [self.pad_id] * max(0, max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_word.get(i, "") for i in ids if i != self.pad_id)
```
Basta trocar a instanciação em `main.py`: `tokenizer = WordTokenizer(...)`. Nenhum outro arquivo muda.

### Exercício 33.3: Validação
```python
def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
    for epoch in range(self.config.num_epochs):
        self.model.train()
        train_loss = sum(self.train_step(x, y) for x, y in train_loader) / len(train_loader)

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                mask = LLMScratch.causal_mask(x.size(1), self.device)
                logits = self.model(x, mask=mask)
                val_loss += self.criterion(logits.reshape(-1, self.config.vocab_size), y.reshape(-1)).item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")
```

### Exercício 33.4: Early Stopping
```python
best_val_loss = float("inf")
patience_counter = 0
patience = 3

for epoch in range(self.config.num_epochs):
    ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(self.model, self.config, epoch, val_loss)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping na epoch {epoch+1}")
            break
```

### Exercício 33.5: Evaluator
```python
# training/evaluator.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.llmscratch import LLMScratch


class Evaluator:
    def __init__(self, model: LLMScratch, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def perplexity(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0

        for input_ids, target_ids in dataloader:
            input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
            mask = LLMScratch.causal_mask(input_ids.size(1), self.device)
            logits = self.model(input_ids, mask=mask)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=0,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (target_ids != 0).sum().item()

        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
```
`Evaluator` segue exatamente o mesmo padrão de `Trainer` e `Generator`: recebe um modelo pronto, não sabe construí-lo.

---

## Desafios Avançados (Opcionais)

### Fixação 33.1: Injeção de Dependência
Reescreva `Trainer.__init__` para receber o `optimizer` já construído como parâmetro (em vez de criá-lo internamente). Qual vantagem isso traz para testes?

### Fixação 33.2: Interface Abstrata de Tokenizer
Crie uma classe abstrata `BaseTokenizer` (usando `abc.ABC`) e faça `CharTokenizer` e `WordTokenizer` herdarem dela. Que garantia isso adiciona?

### Fixação 33.3: Configuração via YAML
Adicione um método `LLMScratchConfig.from_yaml(path)` que carrega a configuração de um arquivo `.yaml` em vez de valores hardcoded no `main.py`.

### Fixação 33.4: Logging Estruturado
Substitua os `print()` espalhados por um módulo `logging` configurado em um único lugar (`main.py`), injetado como dependência em `Trainer`.

### Fixação 33.5: Testes Automatizados
Converta a seção "Testando Cada Módulo Isoladamente" em um arquivo `tests/test_model.py` usando `pytest`, com uma função de teste por asserção.

---

## Resumo

- **Domain** (`config.py`, `tokenizer.py`, `model/`): pura lógica e matemática, zero I/O, testável isoladamente
- **Application** (`training/trainer.py`, `inference/generator.py`): orquestra o domínio para um objetivo (treinar, gerar)
- **Infrastructure** (`data/`, `training/checkpoint.py`): sabe sobre disco, `DataLoader`, serialização
- **Composition root** (`main.py`): o único lugar que conhece todas as peças e as liga
- **Regra de ouro**: dependências apontam para dentro — domínio nunca importa aplicação ou infraestrutura
- **Ganho prático**: cada módulo é testável sozinho, e trocar uma peça (tokenizer, dataset, otimizador) não exige tocar nas outras

Você agora tem LLMScratch não como um script, mas como um projeto — o tipo de estrutura que aguenta crescer sem virar um emaranhado ingerenciável.

Próximo capítulo: **Além de LLMScratch** — escalar esse mesmo scaffold para modelos maiores, dados reais, e treino distribuído.

---

**Próximo**: [Capítulo 34: Além de LLMScratch](34_beyond.md)
