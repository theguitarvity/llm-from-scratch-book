# Progresso de Desenvolvimento - LLMScratch LLM Book

## ✅ Capítulos Completos (34/34)

### Parte I: Fundamentos
- ✅ **01** - Introdução e Motivação (completo)
- ✅ **02** - Setup do Ambiente (com macOS/Linux/Windows)
- ✅ **03** - Tensores e PyTorch (shapes, gradientes)
- ✅ **04** - Álgebra Linear Essencial (matrizes, eigenvalores, SVD)
- ✅ **05** - Operações Básicas (indexação, reduções, ativações)

### Parte II: Embeddings e Representações
- ✅ **06** - O Conceito de Embedding (lookup, similiaridade, espaço semântico)
- ✅ **07** - Implementação de Embeddings (manual + nn.Embedding)
- ✅ **08** - Projeções Lineares (y = xW + b, manual + nn.Linear)
- ✅ **09** - Origem dos Parâmetros W e b (inicialização, Xavier, Kaiming)

### Parte III: Atenção e Self-Attention
- ✅ **10** - Mecanismo de Atenção (Q, K, V, scores, softmax, output)
- ✅ **11** - Query, Key, Value Explicados (projeções independentes, self vs cross-attention)
- ✅ **12** - Attention Scores (dot product, interpretação geométrica)
- ✅ **13** - Scaling por sqrt(d_k) e Softmax (estabilidade numérica, saturação)
- ✅ **14** - Weighted Sum de V (combinação convexa, vs mean pooling)
- ✅ **15** - Causal Masking (máscara triangular, -inf antes do softmax)
- ✅ **16** - Self-Attention Completo (montando tudo, gradientes end-to-end)

### Parte IV: Multi-Head e Normalização
- ✅ **17** - Multi-Head Attention (paralelo, concatenação, W_o)
- ✅ **18** - Residual Connections (skip connections, gradient highway)
- ✅ **19** - Layer Normalization (gamma, beta, LN vs BN, Pre-LN vs Post-LN)

### Parte V: Arquitetura Transformer
- ✅ **20** - MLP e Feed-Forward Network (expansão 4x, GELU)
- ✅ **21** - Bloco Transformer Completo (attention + FFN + residual + norm)
- ✅ **22** - Positional Embeddings (sinusoidal, aprendível, RoPE)
- ✅ **23** - Arquitetura Fim a Fim (GPTModel completo, weight tying)

### Parte VI: Tokenização e Input
- ✅ **24** - Tokenização (BPE do zero, tokens especiais)
- ✅ **25** - Vocabulary e Context Windows (trade-offs, custo O(n²))

### Parte VII: Treinamento
- ✅ **26** - Logits e Cross-Entropy Loss (softmax, MLE, perplexidade)
- ✅ **27** - Backpropagação e Gradientes (chain rule, autograd, clipping)
- ✅ **28** - Otimizadores (SGD, Momentum, Adam, AdamW)
- ✅ **29** - Treinamento Autoregressivo (shift-by-one, teacher forcing)
- ✅ **30** - Checkpoints e Avaliação (save/load, overfitting, debugging)

### Parte VIII: Geração e Prática
- ✅ **31** - Sampling, Temperature e Top-K/P (greedy, nucleus sampling)
- ✅ **32** - Geração de Texto Completa (loop autoregressivo, KV-cache)

### Parte IX: Projeto LLMScratch
- ✅ **33** - Projeto LLMScratch — Arquitetura Modular (scaffold Clean Architecture, 13 arquivos, journey completo)
- ✅ **34** - Além de LLMScratch (escalar, distributed training, fine-tuning)

---

## 📊 Estatísticas

| Métrica | Valor |
|---------|-------|
| Capítulos Completos | 34/34 (100%) |
| Palavras Escritas (estimado) | ~150,000+ |
| Código em Python (blocos) | ~250+ |
| Exercícios Criados | ~170+ |
| Exercícios de Fixação | ~170+ |

---

## ✨ Características de Cada Capítulo

- **Objetivos** (o que você aprende)
- **Por Que Isso Importa** (motivação, conexão com bugs/debugging reais)
- **Seções técnicas** (fórmulas em LaTeX, shapes explícitos, exemplo numérico manual)
- **Experimento(s)** (código PyTorch executável e reprodutível)
- **Erros Comuns** (pitfalls a evitar, ❌ vs ✓)
- **Exercícios** (5 exercícios com gabarito)
- **Desafios Avançados** (5 desafios extras opcionais)
- **Resumo** (recap + link para o próximo capítulo)

O Capítulo 33 foge um pouco desse padrão: em vez de um único bloco de código, guia a construção de um projeto modular de 13 arquivos, inspirado em Clean Architecture (Domain / Application / Infrastructure), do primeiro `config.py` até `main.py` rodando treino e inferência ponta a ponta.

---

## 📚 Como Usar Este Livro

1. **Leia Sequencialmente**: A progressão é deliberada
2. **Rode os Experimentos**: Copy-paste, execute, modifique
3. **Faça os Exercícios**: Antes de ler gabarito
4. **Combine Conceitos**: Cada novo capítulo usa anteriores

---

## 🚀 Para Começar

```bash
cd /Users/mrlopito/Documents/desenv/llm-book

# Ative venv
source venv/bin/activate

# Inicie com Capítulo 01
cd chapters
less 01_introducao_e_motivacao.md
```

---

**Última Atualização**: 2026-08-26
**Versão**: 1.0 (livro completo, 34/34 capítulos)
