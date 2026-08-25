# Capítulo 04: Álgebra Linear Essencial

## 🎯 Objetivos

Ao final deste capítulo, você será capaz de:

1. Entender vetores, matrizes e suas operações fundamentais
2. Computar multiplicação matricial explicitamente
3. Entender conceitos de rank, determinante e inversa
4. Usar normas e distâncias
5. Aplicar álgebra linear a problemas de deep learning

---

## 💡 Intuição

**Álgebra linear é a linguagem de deep learning.** Toda rede neural é uma série de transformações lineares (matrizes) combinadas com ativações (não-linearidades).

Quando você treina um modelo, você está essencialmente:
1. Passando dados através de transformações lineares
2. Computando uma loss (distância entre predição e realidade)
3. Atualizando pesos para diminuir essa distância

Sem entender álgebra linear, você está fazendo magia. Com ela, você vê os mecanismos.

---

## 📐 Vetores e Matrizes

### Notação

- **Escalar**: $a, b, c$ (números)
- **Vetor**: $\mathbf{v}, \mathbf{w}$ (matriz Nx1)
- **Matriz**: $\mathbf{X}, \mathbf{W}$ (matriz MxN)
- **Transposta**: $\mathbf{X}^T$ (inverte linhas e colunas)

Exemplo:
$$\mathbf{v} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}, \quad \mathbf{X} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$

$$\mathbf{X}^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

### Dimensões

Uma matriz com shape [m, n] tem:
- m linhas
- n colunas
- m*n elementos totais

---

## ⭐ Multiplicação Matricial (o coração de tudo)

### Multiplicação Matriz-Vetor

Seja $\mathbf{X} \in \mathbb{R}^{m \times n}$ e $\mathbf{v} \in \mathbb{R}^n$.

$$\mathbf{y} = \mathbf{X} \mathbf{v}$$

onde cada elemento é:

$$y_i = \sum_{j=1}^n X_{ij} v_j$$

Ou seja: cada linha de X é multiplicada **escalarmente** (dot product) com v.

**Exemplo**:

$$\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 1 \cdot 1 + 2 \cdot 2 + 3 \cdot 3 \\ 4 \cdot 1 + 5 \cdot 2 + 6 \cdot 3 \end{pmatrix} = \begin{pmatrix} 14 \\ 32 \end{pmatrix}$$

### Multiplicação Matriz-Matriz

Seja $\mathbf{A} \in \mathbb{R}^{m \times n}$ e $\mathbf{B} \in \mathbb{R}^{n \times p}$.

$$\mathbf{C} = \mathbf{A} \mathbf{B}$$

onde:

$$C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$

**Shape rule**: $[m, n] \times [n, p] = [m, p]$

**Exemplo**:

$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$$

### Propriedades

```
(A @ B) @ C = A @ (B @ C)  # Associatividade
(A @ B)^T = B^T @ A^T      # Transposta inverte ordem
A @ I = A                  # Identidade
A @ 0 = 0                  # Zero
```

Mas note: **$A \times B \neq B \times A$** em geral. Não é comutativa!

---

## 🎯 Produto Escalar e Normas

### Produto Escalar (Dot Product)

Entre dois vetores $\mathbf{v}, \mathbf{w} \in \mathbb{R}^n$:

$$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^n v_i w_i = \mathbf{v}^T \mathbf{w}$$

**Interpretação geométrica**:

$$\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \|\mathbf{w}\| \cos(\theta)$$

onde $\theta$ é o ângulo entre eles.

- Se $\theta = 0°$ (mesma direção): dot product é máximo
- Se $\theta = 90°$ (perpendicular): dot product é 0
- Se $\theta = 180°$ (direção oposta): dot product é negativo

**Exemplo**:

```python
v = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([4.0, 5.0, 6.0])
dot = torch.dot(v, w)  # 1*4 + 2*5 + 3*6 = 32
```

### Normas (Comprimento de Vetor)

**Norma L2** (Euclidiana):

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$$

**Norma L1** (Manhattan):

$$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

**Norma L∞** (Máximo):

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

```python
v = torch.tensor([3.0, 4.0])
norm_l2 = torch.norm(v, p=2)  # sqrt(9 + 16) = 5
norm_l1 = torch.norm(v, p=1)  # 3 + 4 = 7
norm_linf = torch.norm(v, p=float('inf'))  # 4
```

### Distância Euclidiana

Entre dois vetores:

$$d(\mathbf{v}, \mathbf{w}) = \|\mathbf{v} - \mathbf{w}\|_2$$

---

## 🔄 Inversão e Rank

### Inversa

Para uma matriz quadrada $\mathbf{A} \in \mathbb{R}^{n \times n}$, sua inversa $\mathbf{A}^{-1}$ satisfaz:

$$\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}$$

Nem toda matriz tem inversa. Apenas **matrizes de rank completo** (full rank) têm.

```python
A = torch.randn(3, 3)
A_inv = torch.linalg.inv(A)

# Verificar
identity = A @ A_inv
print(identity)  # Deve ser aproximadamente I
```

### Rank

O **rank** de uma matriz é o número de linhas (ou colunas) linearmente independentes.

Para uma matriz $[m, n]$:
- Rank mínimo: 0
- Rank máximo: min(m, n)

**Exemplo**:

$$\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$$

Rank = 1 (segunda linha é 2x primeira linha, não adiciona nova informação).

```python
A = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
rank = torch.linalg.matrix_rank(A)  # 1
```

---

## 🧠 Decomposição de Matrizes

### Eigendecomposição

Para uma matriz quadrada $\mathbf{A}$, um **eigenvector** $\mathbf{v}$ satisfaz:

$$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$

onde $\lambda$ é o **eigenvalue** (autovalor).

Interpretação: multiplicar por A apenas escala v, não muda direção.

```python
A = torch.randn(3, 3)
eigenvalues, eigenvectors = torch.linalg.eigh(A)  # Para matrizes simétricas
```

### SVD (Singular Value Decomposition)

Toda matriz pode ser fatorada:

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

onde:
- $\mathbf{U}$: vetores singulares à esquerda
- $\mathbf{\Sigma}$: valores singulares (diagonais)
- $\mathbf{V}$: vetores singulares à direita

SVD é fundamental para compressão, análise de componentes principais (PCA), etc.

```python
A = torch.randn(4, 6)
U, S, Vt = torch.linalg.svd(A)
# A ≈ U @ torch.diag(S) @ Vt
```

---

## 🧪 Experimento: Álgebra Linear Explícita

Crie `experimento_algebra_linear.py`:

```python
import torch
import numpy as np

print("=" * 70)
print("EXPERIMENTO: Álgebra Linear Essencial")
print("=" * 70)

# ========== 1. MULTIPLICAÇÃO MATRICIAL ==========
print("\n1. MULTIPLICAÇÃO MATRICIAL")
print("-" * 70)

A = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
b = torch.tensor([1.0, 2.0, 3.0])

# Manual: A @ b
# Linha 1: 1*1 + 2*2 + 3*3 = 14
# Linha 2: 4*1 + 5*2 + 6*3 = 32
resultado_manual = torch.tensor([14.0, 32.0])

# Usando PyTorch
resultado_pytorch = A @ b

print(f"A = \n{A}")
print(f"b = {b}")
print(f"A @ b (manual) = {resultado_manual}")
print(f"A @ b (PyTorch) = {resultado_pytorch}")
print(f"Iguais? {torch.allclose(resultado_manual, resultado_pytorch)}")

# ========== 2. PRODUTO ESCALAR ==========
print("\n2. PRODUTO ESCALAR")
print("-" * 70)

v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

# Manual: 1*4 + 2*5 + 3*6 = 32
dot_manual = 1*4 + 2*5 + 3*6
dot_pytorch = torch.dot(v1, v2)

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 · v2 (manual) = {dot_manual}")
print(f"v1 · v2 (PyTorch) = {dot_pytorch}")

# Ângulo entre vetores
cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
angle_rad = torch.acos(cos_angle)
angle_deg = angle_rad * 180 / np.pi

print(f"Ângulo entre v1 e v2: {angle_deg:.2f}°")

# ========== 3. NORMAS ==========
print("\n3. NORMAS")
print("-" * 70)

v = torch.tensor([3.0, 4.0])

norm_l2 = torch.norm(v, p=2)
norm_l1 = torch.norm(v, p=1)
norm_linf = torch.norm(v, p=float('inf'))

print(f"v = {v}")
print(f"||v||_2 (Euclidiana) = {norm_l2:.4f}")  # sqrt(9 + 16) = 5
print(f"||v||_1 (Manhattan) = {norm_l1:.4f}")   # 3 + 4 = 7
print(f"||v||_∞ (Máximo) = {norm_linf:.4f}")    # 4

# ========== 4. DISTÂNCIA ==========
print("\n4. DISTÂNCIA EUCLIDIANA")
print("-" * 70)

p1 = torch.tensor([0.0, 0.0])
p2 = torch.tensor([3.0, 4.0])

distancia = torch.norm(p2 - p1)
print(f"Ponto 1: {p1}")
print(f"Ponto 2: {p2}")
print(f"Distância: {distancia:.4f}")

# ========== 5. RANK ==========
print("\n5. RANK")
print("-" * 70)

# Matriz de rank completo
A_full = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [2.0, 3.0]
])

# Matriz de rank incompleto
A_low = torch.tensor([
    [1.0, 2.0],
    [2.0, 4.0],
    [3.0, 6.0]
])

rank_full = torch.linalg.matrix_rank(A_full)
rank_low = torch.linalg.matrix_rank(A_low)

print(f"A_full =\n{A_full}")
print(f"Rank: {rank_full}")
print()
print(f"A_low =\n{A_low}")
print(f"Rank: {rank_low} (linhas 2 e 3 são múltiplos de linha 1)")

# ========== 6. INVERSA ==========
print("\n6. INVERSA")
print("-" * 70)

A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

A_inv = torch.linalg.inv(A)
identity = A @ A_inv

print(f"A =\n{A}")
print(f"A_inv =\n{A_inv}")
print(f"A @ A_inv (deve ser ~I) =\n{identity}")

# ========== 7. EIGENDECOMPOSIÇÃO ==========
print("\n7. EIGENVALORES E EIGENVECTORS")
print("-" * 70)

# Matriz simétrica (necessária para eigh)
A_sym = torch.tensor([
    [4.0, 2.0],
    [2.0, 3.0]
])

eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)

print(f"A =\n{A_sym}")
print(f"Eigenvalores: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verificar: A @ v = lambda * v
v = eigenvectors[:, 0]
lambda_ = eigenvalues[0]
print(f"\nVerificação para primeira eigenvector:")
print(f"A @ v = {A_sym @ v}")
print(f"λ * v = {lambda_ * v}")
print(f"Iguais? {torch.allclose(A_sym @ v, lambda_ * v)}")

# ========== 8. SVD ==========
print("\n8. SINGULAR VALUE DECOMPOSITION")
print("-" * 70)

A = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

U, S, Vt = torch.linalg.svd(A, full_matrices=False)

print(f"A =\n{A}")
print(f"U shape: {U.shape}")
print(f"S shape: {S.shape}")
print(f"Vt shape: {Vt.shape}")

# Reconstruir A
A_reconstructed = U @ torch.diag(S) @ Vt
print(f"\nA original:\n{A}")
print(f"A reconstruída:\n{A_reconstructed}")
print(f"Iguais? {torch.allclose(A, A_reconstructed)}")

print("\n" + "=" * 70)
print("✓ Experimento Completo!")
print("=" * 70)
```

Rode:

```bash
python experimento_algebra_linear.py
```

---

## ❌ Erros Comuns

### Erro 1: Shapes Incompatíveis em Matmul

```python
# ❌ Errado
A = torch.randn(3, 4)
B = torch.randn(3, 5)
C = A @ B  # ERRO: (3,4) @ (3,5) não funciona

# ✓ Certo
B = torch.randn(4, 5)
C = A @ B  # OK: (3,4) @ (4,5) = (3,5)
```

### Erro 2: Esquecendo que Matrizes Não Comutam

```python
# ❌ Errado
A @ B == B @ A  # False (em geral)

# ✓ Certo: Siga a ordem
resultado_1 = A @ B
resultado_2 = B @ A
# Diferentes! (exceto em casos especiais)
```

### Erro 3: Inversão de Matrizes Não Full-Rank

```python
# ❌ Errado
A = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
A_inv = torch.linalg.inv(A)  # ERRO: Matriz singular

# ✓ Certo: Verificar rank antes
rank = torch.linalg.matrix_rank(A)  # 1, não full rank
# Usar pseudo-inversa ao invés
A_pinv = torch.linalg.pinv(A)
```

---

## ✍️ Exercícios

### Exercício 4.1: Multiplicação Manual
Compute manualmente $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 \\ 6 \end{pmatrix}$. Depois verifique com PyTorch.

### Exercício 4.2: Produto Escalar
Dados $\mathbf{v} = [1, 2, 3]$ e $\mathbf{w} = [4, 5, 6]$, compute $\mathbf{v} \cdot \mathbf{w}$.

### Exercício 4.3: Norma
Qual é a norma L2 de $[3, 4]$?

### Exercício 4.4: Distância
Qual é a distância Euclidiana entre pontos $(0, 0)$ e $(3, 4)$?

### Exercício 4.5: Rank
Qual é o rank de $\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$?

---

## 📚 Gabarito

### Exercício 4.1: Multiplicação Manual

$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 \\ 6 \end{pmatrix} = \begin{pmatrix} 1 \cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4 \cdot 6 \end{pmatrix} = \begin{pmatrix} 17 \\ 39 \end{pmatrix}$$

```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([5.0, 6.0])
print(A @ b)  # tensor([17., 39.])
```

### Exercício 4.2: Produto Escalar

$$[1, 2, 3] \cdot [4, 5, 6] = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32$$

### Exercício 4.3: Norma

$$\|[3, 4]\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

### Exercício 4.4: Distância

$$d = \sqrt{(3-0)^2 + (4-0)^2} = \sqrt{9 + 16} = 5$$

### Exercício 4.5: Rank

Rank = 1. (Segunda linha é 2x primeira linha)

---

## 🎯 Exercícios de Fixação (Opcionais)

### Fixação 4.1: Matriz de Rotação
Uma matriz de rotação 2D por ângulo $\theta$ é:
$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

Crie essa matriz para $\theta = 45°$. Verifique que $R @ R^T = I$ (propriedade de matrizes ortogonais).

**Dica**: Use `torch.cos()`, `torch.sin()`, `torch.pi`.

### Fixação 4.2: Projeção
Dados $\mathbf{v} = [3, 4]$ e $\mathbf{u} = [1, 0]$, compute a projeção de v em u:

$$\text{proj}_\mathbf{u} \mathbf{v} = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}$$

### Fixação 4.3: Matrizes Simétricas
Uma matriz simétrica satisfaz $\mathbf{A} = \mathbf{A}^T$. 

Crie uma matriz simétrica [3, 3]. Verifique que todos seus eigenvalores são reais (propriedade de matrizes simétricas).

### Fixação 4.4: Condicionamento Numérico
Crie uma matriz "quase-singular":
$$\mathbf{A} = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$$

Compute sua inversa. Observe que pequenas mudanças em A causam grandes mudanças em $A^{-1}$ (mal-condicionada).

### Fixação 4.5: Rank-1 Update
Mostre que adicionar um vetor externo (outer product) muda o rank:

$$\text{rank}(\mathbf{A}) \leq \text{rank}(\mathbf{A} + \mathbf{u} \mathbf{v}^T) \leq \text{rank}(\mathbf{A}) + 1$$

Crie exemplo prático.

---

## 🎓 Resumo

- **Multiplicação matricial**: $[m, n] @ [n, p] = [m, p]$
- **Produto escalar**: Mede similaridade entre vetores
- **Normas**: Medem "tamanho" de vetores
- **Rank**: Mede quantidade de informação independente
- **Inversa**: Existe apenas se rank completo
- **Eigen e SVD**: Decomposições fundamentais

Álgebra linear não é adicional — é **central** em deep learning.

No próximo capítulo: continuamos com operações em PyTorch, focando em casos práticos.

---

**Próximo**: [Capítulo 05: Operações Básicas](05_operacoes_basicas.md)
