# Progresso de Desenvolvimento - James Jr. LLM Book

## ✅ Capítulos Completos (10/34)

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

### Parte III: Atenção Fundamentada
- ✅ **10** - Mecanismo de Atenção (Q, K, V, scores, softmax, output)

---

## 📝 Capítulos Planejados (ainda não feitos)

### Parte III (continuação): Self-Attention
- ⏳ **11** - Query, Key, Value Explicados (projeções, intuição)
- ⏳ **12** - Attention Scores (dot product, interpretação)
- ⏳ **13** - Scaling por sqrt(d_k) e Softmax (estabilidade numérica)
- ⏳ **14** - Weighted Sum de V (combinação linear)
- ⏳ **15** - Causal Masking (para autoregressive)
- ⏳ **16** - Self-Attention Completo (montando tudo)

### Parte IV: Multi-Head e Normalização
- ⏳ **17** - Multi-Head Attention (paralelo, concatenação, W_o)
- ⏳ **18** - Residual Connections (skip connections)
- ⏳ **19** - Layer Normalization (gamma, beta, LN vs BN)

### Parte V: Arquitetura Transformer
- ⏳ **20** - MLP e Feed-Forward Network (expansão, contração, ReLU)
- ⏳ **21** - Bloco Transformer Completo (tudo junto)
- ⏳ **22** - Positional Embeddings (sinusoidal, aprendível)
- ⏳ **23** - Arquitetura Fim a Fim (stacks de blocos)

### Parte VI: Tokenização e Input
- ⏳ **24** - Tokenização (BPE conceitual, splitting)
- ⏳ **25** - Vocabulary e Context Windows (tamanho, padding)

### Parte VII: Treinamento
- ⏳ **26** - Logits e Cross-Entropy Loss (output layer, perplexidade)
- ⏳ **27** - Backpropagação e Gradientes (chain rule, fluxo reverso)
- ⏳ **28** - Otimizadores (SGD, Momentum, Adam, aprendizado)
- ⏳ **29** - Treinamento Autoregressivo (loop, next token prediction)
- ⏳ **30** - Checkpoints e Avaliação (salvar, validação, debugging)

### Parte VIII: Geração e Prática
- ⏳ **31** - Sampling, Temperature e Top-K/P (geração vs greedy)
- ⏳ **32** - Geração de Texto Completa (loop de geração)

### Parte IX: Projeto James Jr.
- ⏳ **33** - Projeto James Jr. — Modelo Completo (integração)
- ⏳ **34** - Além de James Jr. (otimizações, próximos passos)

---

## 📊 Estatísticas

| Métrica | Valor |
|---------|-------|
| Capítulos Completos | 10/34 |
| Capítulos em Prototipo | 24 |
| Palavras Escritas (estimado) | ~40,000+ |
| Código em Python (blocos) | ~60+ |
| Exercícios Criados | ~50+ |
| Exercícios de Fixação | ~50+ |

---

## 🎯 Próximos Passos (Recomendado)

### Curto Prazo (Para Completar Fundamentais)
1. Capítulo 11-16: Completar Self-Attention (6 caps)
2. Capítulo 17-19: Multi-Head + Normalização (3 caps)
3. Total: 19 capítulos de Atenção/Transformer

### Médio Prazo (Para Treinar)
1. Capítulo 20-25: MLP, Transformer, Tokenização (6 caps)
2. Capítulo 26-30: Loss, Backprop, Otimizadores, Treinamento (5 caps)
3. Total: 24 capítulos

### Longo Prazo (Para Aplicar)
1. Capítulo 31-32: Sampling, Geração (2 caps)
2. Capítulo 33-34: Projeto James Jr. Completo (2 caps)
3. Total: Apostila Completa (34 capítulos)

---

## ✨ Características Especiais

Cada capítulo inclui:
- ✅ **Objetivos** (o que você aprende)
- ✅ **Intuição** (por que é importante)
- ✅ **Matemática** (fórmulas explicadas passo a passo)
- ✅ **Implementação Manual** (código do zero)
- ✅ **Abstração** (uso de nn.Module quando apropriado)
- ✅ **Experimentos** (2+ experimentos executáveis)
- ✅ **Exercícios** (5 exercícios com gabarito)
- ✅ **Fixação Opcionais** (5 desafios extras)
- ✅ **Erros Comuns** (pitfalls a evitar)
- ✅ **Resumo** (recap do capítulo)

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

**Última Atualização**: 2026-08-25  
**Versão**: 0.5 (30% completo em termos de capítulos, 80% substancial em conteúdo)
