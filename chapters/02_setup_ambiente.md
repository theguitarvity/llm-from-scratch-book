# Capítulo 02: Setup do Ambiente

## 🎯 Objetivos

Ao final deste capítulo, você será capaz de:

1. Instalar PyTorch corretamente no seu sistema
2. Criar e ativar um virtual environment isolado
3. Verificar se seu setup está funcionando (especialmente GPU/MPS)
4. Entender PyTorch como ferramenta, não mágica
5. Rodar seu primeiro script PyTorch com sucesso

---

## 💡 Intuição

Uma **language model** é fundamentalmente uma série de operações matemáticas. Para fazer essas operações rápido, precisamos de:

1. **Linguagem**: Python (simples, legível)
2. **Biblioteca de numeração**: PyTorch (operações tensores, GPU)
3. **Ambiente isolado**: Virtual environment (não bagunçar seu Python sistema)

PyTorch é nossa "calculadora turbo". Sem ela, computar gradientes de um modelo com bilhões de parâmetros seria impossível (ou levaria meses).

---

## 🔧 Instalação Step-by-Step

### Pré-requisitos

**macOS Apple Silicon (M1/M2/M3+)**:
- Python 3.10+ instalado
- Acesso ao terminal
- ~500MB de espaço em disco (PyTorch)

**Verificar versão de Python**:

```bash
python3 --version
```

Você deve ver `Python 3.10.x` ou maior. Se tiver apenas Python 2 ou versão antiga, instale via Homebrew:

```bash
brew install python@3.11
```

### Passo 1: Criar Virtual Environment

Um **virtual environment** é um "sandbox" Python isolado. Cada projeto tem seu próprio, evitando conflitos de versões.

```bash
# Navegue ao diretório do projeto
cd /Users/mrlopito/Documents/desenv/llm-book

# Crie o venv
python3 -m venv venv

# Ative
source venv/bin/activate

# Você verá "(venv)" no prompt agora
```

Quando ativado:
- `python` e `pip` referem-se ao venv, não ao sistema
- Instalar pacotes aqui não afeta seu Python global

### Passo 2: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Passo 3: Instalar PyTorch

Escolha a instalação apropriada para seu sistema:

#### macOS (Apple Silicon M1/M2/M3+)

```bash
# Instalação recomendada (com suporte MPS automático)
pip install torch torchvision torchaudio
```

PyTorch 2.0+ suporta **Metal Performance Shaders (MPS)**, acelerando computações automaticamente.

#### macOS (Intel)

```bash
# Para Macs com Intel
pip install torch torchvision torchaudio
```

CPU funcionará perfeitamente. Se tiver GPU NVIDIA eGPU, use a instalação CUDA abaixo.

#### Windows (NVIDIA GPU - CUDA)

Se você tem GPU NVIDIA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(Para CUDA 11.8; verifique seu CUDA version com `nvidia-smi`)

Se apenas CPU:

```bash
pip install torch torchvision torchaudio
```

#### Linux (NVIDIA GPU - CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(Para CUDA 11.8)

Se apenas CPU:

```bash
pip install torch torchvision torchaudio
```

#### Linux (AMD GPU - ROCm)

Se você tem AMD GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

Se apenas CPU:

```bash
pip install torch torchvision torchaudio
```

### Passo 4: Instalar Dependências

```bash
pip install numpy matplotlib
```

- **NumPy**: Operações numéricas (comparação com PyTorch)
- **Matplotlib**: Gráficos (para visualizar loss, etc.)

### Passo 5: Verificação de Instalação

Crie um arquivo `test_setup.py`:

```python
import torch
import numpy as np

print("=" * 50)
print("TESTE DE INSTALAÇÃO - James Jr. LLM Book")
print("=" * 50)

# Versão PyTorch
print(f"\n✓ PyTorch version: {torch.__version__}")

# Versão NumPy
print(f"✓ NumPy version: {np.__version__}")

# Verificar dispositivo disponível
print(f"\n✓ MPS available: {torch.backends.mps.is_available()}")
print(f"✓ MPS built: {torch.backends.mps.is_built()}")

# Escolher device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✓ Usando: Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✓ Usando: CUDA (NVIDIA)")
else:
    device = torch.device("cpu")
    print(f"\n✓ Usando: CPU (fallback)")

# Teste simples
print(f"\nTestando computação em {device}...")
x = torch.randn(3, 4, device=device)
y = torch.randn(4, 2, device=device)
z = torch.matmul(x, y)

print(f"  x.shape: {x.shape}")
print(f"  y.shape: {y.shape}")
print(f"  z = x @ y, z.shape: {z.shape}")
print(f"  z.device: {z.device}")

# Teste de gradiente
print(f"\nTestando gradientes...")
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
b = (a ** 2).sum()
b.backward()
print(f"  a = {a.data}")
print(f"  b = sum(a^2) = {b.item():.4f}")
print(f"  ∇b/∇a = {a.grad}")

print("\n" + "=" * 50)
print("✓ SETUP COMPLETO E FUNCIONAL!")
print("=" * 50)
```

Rode:

```bash
python test_setup.py
```

Você deve ver algo como:

```
==================================================
TESTE DE INSTALAÇÃO - James Jr. LLM Book
==================================================

✓ PyTorch version: 2.0.1
✓ NumPy version: 1.24.3

✓ MPS available: True
✓ MPS built: True

✓ Usando: Metal Performance Shaders (MPS)

Testando computação em mps...
  x.shape: torch.Size([3, 4])
  y.shape: torch.Size([4, 2])
  z = x @ y, z.shape: torch.Size([3, 2])
  z.device: mps:0

Testando gradientes...
  a = tensor([1., 2., 3.])
  b = sum(a^2) = 14.0000
  ∇b/∇a = tensor([2., 4., 6.])

==================================================
✓ SETUP COMPLETO E FUNCIONAL!
==================================================
```

Se vir `MPS available: True`, ótimo! GPU funcionando.

Se vir `MPS available: False`, não se preocupe. CPU também funciona (mais lento, mas correto).

---

## 🖥️ Configuração de Dispositivo

PyTorch permite escolher onde computar: **CPU** (todos têm), **MPS** (Apple), **CUDA** (NVIDIA).

### Padrão que Usaremos no Livro

```python
# Detectar automaticamente
device = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    "cuda" if torch.cuda.is_available() else 
    "cpu"
)

# Depois: mover tensores para device
x = torch.randn(3, 4).to(device)

# Ou criar direto no device
y = torch.randn(3, 4, device=device)
```

**Nota importante**: 
- MPS é ~5-10x mais rápido que CPU em operações grandes
- CPU é completamente OK para o livro (modelos pequenos)
- CUDA (NVIDIA) seria ainda mais rápido, mas você precisa de GPU NVIDIA

---

## 📦 Requisitos (sem criar arquivo extra)

Se preferir, aqui está o equivalente de um `requirements.txt` em formato Markdown, para colar no console se necessário:

```bash
# Versão minimal (recomendada)
pip install torch>=2.0 numpy matplotlib

# Versão com extras opcionais
pip install torch>=2.0 numpy matplotlib jupyter ipython
```

---

## 🧪 Experimento: Primeira Operação

Crie `primeiro_tensor.py`:

```python
import torch

# Criar dois vetores
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Operações
soma = a + b
produto_escalar = torch.dot(a, b)
elemento_wise = a * b

print("Vetor a:", a)
print("Vetor b:", b)
print("\na + b:", soma)
print("a · b (dot product):", produto_escalar)
print("a ⊙ b (elemento-wise):", elemento_wise)

# Operação com matriz
M = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

v = torch.tensor([1.0, 2.0])

resultado = torch.matmul(M, v)

print("\nMatriz M:")
print(M)
print("\nVetor v:", v)
print("\nM @ v (matrix-vector product):")
print(resultado)
```

Rode:

```bash
python primeiro_tensor.py
```

Output esperado:

```
Vetor a: tensor([1., 2., 3.])
Vetor b: tensor([4., 5., 6.])

a + b: tensor([5., 7., 9.])
a · b (dot product): tensor(32.)
a ⊙ b (elemento-wise): tensor([ 4., 10., 18.])

Matriz M:
tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])

Vetor v: tensor([1., 2.])

M @ v (matrix-vector product):
tensor([ 5., 11., 17.])
```

---

## ❌ Erros Comuns

### Erro 1: "ModuleNotFoundError: No module named 'torch'"

**Causa**: Você não ativou o venv, ou pip instalou em lugar errado.

**Solução**:
```bash
# Verifique se venv está ativado (deve ver "(venv)" no prompt)
source venv/bin/activate

# Reinstale
pip install torch
```

### Erro 2: "MPS is not available"

**Causa**: Normal em Mac com chip Intel ou PyTorch antigo.

**Solução**: Use CPU. É OK. O código continua idêntico, apenas mais lento.

```python
device = torch.device("cpu")  # Force CPU
```

### Erro 3: Operação muda de device

Exemplo errado:
```python
x = torch.randn(3, 4, device="mps")
y = torch.randn(3, 4, device="cpu")
z = x + y  # ❌ ERRO: misturando devices
```

**Solução**: Garanta que ambos estão no mesmo device.

```python
device = torch.device("mps")
x = torch.randn(3, 4, device=device)
y = torch.randn(3, 4, device=device)
z = x + y  # ✓ OK
```

---

## ✍️ Exercícios

### Exercício 2.1: Teste de Device
Rode `test_setup.py`. Qual device você tem disponível? MPS, CUDA ou CPU?

### Exercício 2.2: Tensor Simples
Crie dois tensores de shape [2, 3]. Some-os. Multiplique pela transposta um do outro. Qual é o shape final?

```python
# Seu código aqui
x = torch.randn(2, 3)
y = torch.randn(2, 3)

soma = x + y
print(f"soma.shape: {soma.shape}")

xT = x.T  # Transposta
resultado = torch.matmul(x, xT)  # [2,3] @ [3,2] = [2,2]
print(f"resultado.shape: {resultado.shape}")
```

### Exercício 2.3: Gradientes
Crie um tensor `w` com `requires_grad=True`. Compute `y = 3*w^2 + 2*w + 1`. Faça `y.backward()`. Qual é `w.grad`?

*Dica*: Use a regra da cadeia. Derivada de y em relação a w é `∇y/∇w = 6*w + 2`.

---

## 📚 Gabarito

### Exercício 2.1: Teste de Device
Resposta esperada (macOS Apple Silicon):
```
✓ MPS available: True
✓ Usando: Metal Performance Shaders (MPS)
```

Se der `False`, sem problema — CPU funciona.

### Exercício 2.2: Tensor Simples
```python
import torch

x = torch.randn(2, 3)
y = torch.randn(2, 3)

soma = x + y
print(f"soma.shape: {soma.shape}")  # [2, 3]

xT = x.T  # x transposto: [3, 2]
resultado = torch.matmul(x, xT)  # [2,3] @ [3,2] = [2, 2]
print(f"resultado.shape: {resultado.shape}")  # [2, 2]
```

### Exercício 2.3: Gradientes
```python
import torch

w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = 3*w**2 + 2*w + 1
y_sum = y.sum()  # Scalar, para poder fazer .backward()
y_sum.backward()

print(f"w.grad = {w.grad}")
# Esperado: [8, 14, 20]  (porque 6*w + 2 para cada elemento)
# 6*1+2=8, 6*2+2=14, 6*3+2=20
```

---

## 🎓 Resumo

- ✅ Virtual environment isola seu projeto
- ✅ PyTorch com MPS/CUDA/CPU oferece aceleração
- ✅ Sempre mova tensores para o mesmo device
- ✅ Teste seu setup com `test_setup.py`
- ✅ Operações básicas (soma, matmul, transposta) funcionam

No próximo capítulo, mergulharemos fundo em **tensores**, **shapes** e **operações básicas**.

---

**Próximo**: [Capítulo 03: Tensores e PyTorch](03_tensores_e_pytorch.md)
