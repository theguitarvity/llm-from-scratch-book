# Capítulo 33: Projeto James Jr. — Modelo Completo Funcional

## Objetivos

Ao final deste capítulo, você será capaz de:

1. Implementar uma LLM pequenininha completa (James Jr.)
2. Treinar em dados de verdade (simples)
3. Fazer inferência e gerar texto
4. Salvar e carregar checkpoints
5. Debugar e melhorar o modelo

---

## Por Que Isso Importa

Finalmente, juntamos TUDO que aprendemos em um modelo de verdade.

James Jr. é pequeno (1-10M parâmetros), mas funciona:
- Tokenização
- Embeddings
- Transformer blocks
- Cabeça de predição
- Treinamento autoregressivo
- Geração de texto

---

## Arquitetura

```
Input: "O gato"
  ↓
Tokenização: [1, 2]
  ↓
Embedding: [1, 2, 64] (embedding_dim=64)
  ↓
Positional Embedding: [1, 2, 64]
  ↓
[Transformer Block x N] (num_layers=2)
  ├─ Self-Attention
  ├─ Feed-Forward
  ├─ Residual + LayerNorm
  └─ Repeat
  ↓
Output: [1, 2, d_model]
  ↓
Logits Head: [1, 2, vocab_size]
  ↓
Loss (Cross-Entropy)
```

---

## Implementação Completa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========== CONFIGURAÇÃO ==========
CONFIG = {
    "vocab_size": 1000,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 2,
    "d_ff": 256,  # Feed-forward inner dim
    "max_seq_len": 128,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 10,
}

# ========== COMPONENTES ==========

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention com residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward com residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x

class JamesJr(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        self.token_embed = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_embed = PositionalEmbedding(config["d_model"], config["max_seq_len"])
        
        # Transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["num_heads"],
                config["d_ff"],
                config["dropout"]
            )
            for _ in range(config["num_layers"])
        ])
        
        # Output head
        self.norm = nn.LayerNorm(config["d_model"])
        self.head = nn.Linear(config["d_model"], config["vocab_size"])
    
    def forward(self, input_ids, mask=None):
        # input_ids: [batch, seq_len]
        x = self.token_embed(input_ids)  # [batch, seq_len, d_model]
        x = self.pos_embed(x)
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x, mask=mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def generate(self, prompt_ids, max_len=100, temperature=1.0):
        """Gera texto dado um prompt"""
        self.eval()
        
        for _ in range(max_len):
            # Forward
            logits = self.forward(prompt_ids[:, -self.config["max_seq_len"]:])
            
            # Próximo token (último position)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
            
            # Stop condition
            if next_token.item() == 0:  # EOS token
                break
        
        return prompt_ids

# ========== DATASET ==========

class TextDataset(Dataset):
    def __init__(self, texts, max_len=128, vocab_size=1000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        # Simular tokenização: usar character codes
        self.data = []
        for text in texts:
            ids = [ord(c) % vocab_size for c in text[:max_len]]
            ids = ids + [0] * (max_len - len(ids))  # Padding
            self.data.append(torch.tensor(ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx]
        # Input é tudo exceto último token
        # Target é tudo exceto primeiro token
        return ids[:-1], ids[1:]

# ========== TREINAMENTO ==========

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dados (dummy)
    texts = [
        "O gato dormia no sofá" * 5,
        "Um cachorro corria no parque" * 5,
        "A chuva caía sobre a cidade" * 5,
        "O sol brilhava no céu azul" * 5,
    ]
    
    dataset = TextDataset(texts, max_len=CONFIG["max_seq_len"], vocab_size=CONFIG["vocab_size"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # Modelo
    model = JamesJr(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    print(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    
    # Treinar
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward
            logits = model(input_ids)  # [batch, seq_len-1, vocab]
            
            # Loss
            loss = criterion(
                logits.reshape(-1, CONFIG["vocab_size"]),
                target_ids.reshape(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: loss = {avg_loss:.4f}")
    
    # Salvar
    torch.save({
        "model_state": model.state_dict(),
        "config": CONFIG,
    }, "james_jr.pt")
    print("Modelo salvo em james_jr.pt")
    
    # Gerar
    print("\nGerando texto:")
    prompt = "O gato"
    prompt_ids = torch.tensor([[ord(c) % CONFIG["vocab_size"] for c in prompt]], dtype=torch.long).to(device)
    
    output_ids = model.generate(prompt_ids, max_len=50)
    generated_text = "".join([chr(id.item()) for id in output_ids[0]])
    print(f"Prompt: {prompt}")
    print(f"Gerado: {generated_text}")

if __name__ == "__main__":
    train()
```

---

## Teste Rápido

```python
# Carregar modelo treinado
checkpoint = torch.load("james_jr.pt")
model = JamesJr(checkpoint["config"])
model.load_state_dict(checkpoint["model_state"])

# Inferência
prompt = "O"
prompt_ids = torch.tensor([[ord(c) % CONFIG["vocab_size"] for c in prompt]], dtype=torch.long)
output = model.generate(prompt_ids, max_len=30)
print(output)
```

---

## Exercícios

### Exercício 33.1: Treine e Gere
Rode o código acima. Gere texto.

### Exercício 33.2: Adicione Dropout
Implemente dropout em PositionalEmbedding. Qual é o efeito?

### Exercício 33.3: Diferentes Configurações
Treine com CONFIG alterado (menos layers, menos heads). Compare velocidade.

### Exercício 33.4: Salve Checkpoint
Durante treinamento, salve modelo a cada 10 epochs.

### Exercício 33.5: Avalie
Compute perplexidade no dataset.

---

## 🎓 Resumo

- ✅ Você construiu uma LLM funcional do zero
- ✅ Você treinou em dados de verdade
- ✅ Você gerou texto
- ✅ Agora você entende como ChatGPT/Claude funcionam em alto nível

Próximo: **Além de James Jr.** — otimizações, distributed training, fine-tuning.

---

**Parabéns! Você chegou ao fim da jornada fundamental.** 🎉

---

**Próximo**: [Capítulo 34: Além de James Jr.](34_beyond.md)
