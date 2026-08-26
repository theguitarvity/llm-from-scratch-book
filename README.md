# LLMScratch — Construindo uma LLM do Zero

Uma apostila completa em português brasileiro sobre os fundamentos de Large Language Models, com foco em entender os conceitos antes de abstrair.

**Objetivo**: Reconstruir, do zero até a inferência completa, os mecanismos de uma language model pequena e funcional. Neste livro, você implementará explicitamente cada camada, operação e conceito — começando com tensores crus até chegar a um modelo treinável e gerando texto.

**Filosofia**: Entender antes de abstrair. Antes de usar `nn.Linear` ou `nn.MultiheadAttention`, você implementará as operações com tensores e numpy/PyTorch direto, vendo o que realmente acontece nos números.

---

## 📚 Estrutura do Livro

### **Parte I: Fundamentos**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [01](chapters/01_introducao_e_motivacao.md) | Introdução e Motivação | O que é uma LLM? Por que estudar do zero? Contexto histórico. |
| [02](chapters/02_setup_ambiente.md) | Setup do Ambiente | PyTorch, versões, GPU/CPU, verificação de instalação. |
| [03](chapters/03_tensores_e_pytorch.md) | Tensores e PyTorch | Criação de tensores, operações, shapes, gradientes. |
| [04](chapters/04_algebra_linear_essencial.md) | Álgebra Linear Essencial | Matrizes, vetores, multiplicação, transposição, norms. |
| [05](chapters/05_operacoes_basicas.md) | Operações Básicas | Broadcasting, redução, indexação, manipulação de shapes. |

### **Parte II: Embeddings e Representações**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [06](chapters/06_embedding_conceito.md) | O Conceito de Embedding | O que é embedding? Por que projetamos em espaços densos? |
| [07](chapters/07_embedding_implementacao.md) | Implementação de Embeddings | Matriz de embedding, lookup, gradientes, treinamento. |
| [08](chapters/08_projecoes_lineares.md) | Projeções Lineares | Transformações lineares, W, bias, linearidade. |
| [09](chapters/09_origem_parametros.md) | Origem dos Parâmetros W e b | Por que W tem forma [d_in, d_out]? Inicialização. |

### **Parte III: Atenção e Self-Attention**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [10](chapters/10_mecanismo_atencao.md) | Mecanismo de Atenção | Query, Key, Value; scores; por que isso funciona. |
| [11](chapters/11_query_key_value.md) | Q, K, V Explicados | Projeções lineares para Q/K/V; shapes; intuição. |
| [12](chapters/12_atencao_scores.md) | Attention Scores | Produto escalar entre Q e K; interpretação. |
| [13](chapters/13_scaling_softmax.md) | Scaling por sqrt(d_k) e Softmax | Por que dividir por raiz quadrada? Softmax revisitado. |
| [14](chapters/14_weighted_sum.md) | Weighted Sum de V | Combinação linear pesada dos Values. |
| [15](chapters/15_causal_masking.md) | Causal Masking | Máscara causal; por que precisamos; implementação. |
| [16](chapters/16_self_attention.md) | Self-Attention Completo | Montando tudo junto; forward pass completo. |

### **Parte IV: Multi-Head e Normalização**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [17](chapters/17_multihead_attention.md) | Multi-Head Attention | Múltiplas cabeças em paralelo; concatenação; W_o. |
| [18](chapters/18_residual_connections.md) | Residual Connections | Skip connections; como viabilizam treinamento profundo. |
| [19](chapters/19_layer_normalization.md) | Layer Normalization | Normalização por camada; gamma e beta; estabilidade. |

### **Parte V: Arquitetura Transformer**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [20](chapters/20_mlp_feedforward.md) | MLP e Feed-Forward Network | Rede densa com ativação; expansão e contração. |
| [21](chapters/21_bloco_transformer.md) | Bloco Transformer Completo | Unindo attention, MLP, residuals, layer norm. |
| [22](chapters/22_positional_embeddings.md) | Positional Embeddings | Por que precisamos de posição? Sinusoidal; treinável. |
| [23](chapters/23_arquitetura_fim_a_fim.md) | Arquitetura Fim a Fim | Stack de blocos; embedding de tokens; cabeça de predição. |

### **Parte VI: Tokenização e Input**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [24](chapters/24_tokenizacao.md) | Tokenização | Dividindo texto em tokens; vocabulary; BPE conceitual. |
| [25](chapters/25_vocabulary_context.md) | Vocabulary e Context Windows | Tamanho do vocabulário; tamanho máximo de sequência. |

### **Parte VII: Treinamento**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [26](chapters/26_logits_e_loss.md) | Logits e Cross-Entropy Loss | Output; probs; cross-entropy; por quê usamos. |
| [27](chapters/27_backpropagacao.md) | Backpropagação e Gradientes | Chain rule; fluxo reverso; cálculo manual de gradientes. |
| [28](chapters/28_otimizadores.md) | Otimizadores | SGD, Momentum, Adam; intuição e implementação. |
| [29](chapters/29_treinamento_autoregressivo.md) | Treinamento Autoregressivo | Loop de treinamento; predição de próximo token. |
| [30](chapters/30_checkpoints_avaliacao.md) | Checkpoints e Avaliação | Salvando modelos; métricas; overfitting; debugging. |

### **Parte VIII: Geração e Prática**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [31](chapters/31_sampling_temperature.md) | Sampling, Temperature e Top-K/P | Greedy vs. sampling; temperature; filtragem. |
| [32](chapters/32_geracao_texto.md) | Geração de Texto Completa | Loop de geração; concatenação de sequências; qualidade. |

### **Parte IX: Projeto LLMScratch**

| Capítulo | Título | Tópicos |
|----------|--------|---------|
| [33](chapters/33_projeto_james_jr.md) | Projeto LLMScratch — Arquitetura Modular | Scaffold inspirado em Clean Architecture; projeto arquivo por arquivo; treino e inferência end-to-end. |
| [34](chapters/34_beyond.md) | Além de LLMScratch | Otimizações; distributed training; fine-tuning; próximos passos. |

### **Guias Extras**

| Guia | Tópicos |
|------|---------|
| [Treinamento com Dados Reais](GUIA_TREINAMENTO_DADOS_REAIS.md) | Datasets públicos (TinyStories, Wikipedia PT, OSCAR, C4); tokenizer BPE real; escalando o projeto modular do Capítulo 33 para treino em corpus real, com warmup, cosine decay e avaliação. |

---

## 🎯 Como Usar Este Livro

1. **Leia sequencialmente**: A progressão é deliberada. Cada capítulo usa conceitos dos anteriores.

2. **Acompanhe os experimentos**: Todo capítulo tem um trecho de código executável. Rode-o, entenda o output.

3. **Faça os exercícios**: No final de cada capítulo, há exercícios com gabarito. Tente resolver antes de olhar a solução.

4. **Implemente do zero**: Não use `nn.MultiheadAttention` nos primeiros capítulos — implemente você mesmo.

5. **Modifique os números**: Os exemplos usam shapes pequenos ([1, 3, 4], [4, 4], etc.). Mude-os e veja como tudo muda.

---

## 💻 Ambiente

### Requisitos Mínimos

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib (para gráficos opcionais)

### Instalação (macOS Apple Silicon/MPS)

```bash
# Clone ou navegue ao diretório do livro
cd llm-book

# Crie um virtual environment
python3 -m venv venv
source venv/bin/activate

# Instale PyTorch (MPS para Apple Silicon)
pip install torch torchvision torchaudio

# Instale dependências
pip install numpy matplotlib
```

### Teste de Instalação

```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
print(f"Device: {device}")
print(f"Tensor: {x}")
EOF
```

---

## 📖 Notas Importantes

- **Matemática é tudo**: Você verá fórmulas. Elas não são opcionais — entender a matemática é entender a IA.

- **Código é implementação da matemática**: Toda fórmula tem um bloco de código correspondente. Eles falam a mesma língua.

- **Exemplos são pequenos propositalmente**: Usamos pequenos shapes (batch size 1, sequências curtas) para você entender o que está acontecendo. Em produção, tudo fica maior, mas o conceito não muda.

- **Gradientes são sua ferramenta de debugging**: Se algo não funciona, veja os gradientes. Eles contam histórias.

- **LLMScratch não é SOTA**: Ele é educacional. O objetivo é entender como LLMs funcionam, não competir com GPT-4.

---

## 🔗 Convenções de Notação

- **x, y, z**: Escalares (números)
- **v, u, w**: Vetores (matrizes 1D)
- **X, W, K**: Matrizes (2D ou mais)
- **[n, d]**: Notação de shape (n linhas, d colunas)
- **⊙**: Elemento-wise (Hadamard) product
- **⊗**: Outer product
- **T**: Transposta
- **∘**: Composição de funções
- **∇**: Gradiente

---

## 📝 Licença e Uso

Este material é educacional. Use, modifique e compartilhe livremente.

---

## 🚀 Começar

Abra [Capítulo 01: Introdução e Motivação](chapters/01_introducao_e_motivacao.md) e comece sua jornada em LLMs.

---

**Última atualização**: 2026-08-25  
**Versão**: 1.0 (em construção)
