# 🚀 Quick Start — LLMScratch LLM Book

Guia de 5 minutos para começar.

---

## 1️⃣ Verifique Ambiente

```bash
# Navegue ao diretório
cd /Users/mrlopito/Documents/desenv/llm-book

# Ative venv (já existe)
source venv/bin/activate

# Verifique PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}, Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')"
```

Se houver erro, execute `python test_setup.py` (veja Capítulo 02).

---

## 2️⃣ Leia Primeiro Capítulo

```bash
less chapters/01_introducao_e_motivacao.md
```

**O que você aprende**: Motivação, história, estrutura do livro.  
**Tempo**: 15 minutos

---

## 3️⃣ Complete Fundamentos (Caps 02-05)

```bash
# Capítulo 02: Setup
less chapters/02_setup_ambiente.md

# Capítulo 03: Tensores
less chapters/03_tensores_e_pytorch.md
# Execute: python experimento_shapes.py

# Capítulo 04: Álgebra Linear
less chapters/04_algebra_linear_essencial.md
# Execute: python experimento_algebra_linear.py

# Capítulo 05: Operações
less chapters/05_operacoes_basicas.md
# Execute: python experimento_operacoes.py
```

**Tempo Total**: 2-3 horas  
**Competência**: Confortável com tensores, shapes, operações

---

## 4️⃣ Entenda Atenção (Caps 06-10)

```bash
# Capítulo 06-10 formam a sequência de atenção
for i in {06..10}; do
  echo "=== Capítulo $i ==="
  less chapters/${i}_*.md
  # Execute experimentos quando encontrar
done
```

**Tempo Total**: 3-4 horas  
**Competência**: Entender como atenção funciona

---

## 5️⃣ Implemente Self-Attention (Cap 16)

```bash
less chapters/16_self_attention.md

# Rode experimento
python3 << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F

# (Copie código do capítulo 16)
# Implemente SelfAttention
# Execute forward pass
# Verifique shapes

print("Self-Attention funcionando!")
EOF
```

**Tempo**: 1 hora  
**Competência**: Implementar atenção manualmente

---

## 6️⃣ Treine LLMScratch (Cap 33)

```bash
# Copie código do Capítulo 33 e execute
python3 << 'EOF'
# (Copie implementação de LLMScratch do Capítulo 33)
# Chame train()
EOF

# Ou salve em arquivo
cat > train_james_jr.py << 'EOF'
# Copie cap 33 aqui
if __name__ == "__main__":
    train()
EOF

python train_james_jr.py
```

**Tempo**: 30-60 minutos (dependendo de hardware)  
**Resultado**: Modelo treinado, checkpoint salvo, texto gerado

---

## 7️⃣ Explore e Modifique

Agora que entende:

1. **Modifique hyperparameters** de LLMScratch
2. **Experimente com seus dados**
3. **Implemente novas features**
4. **Compare com HuggingFace Transformers**

---

## 📚 Índice Completo (Quando Quiser Mais)

Ver `README.md` para estrutura completa de 34 capítulos.

---

## ⏱️ Timeline Recomendada

| Fase | Capítulos | Horas | Resultado |
|------|-----------|-------|-----------|
| Dia 1 | 01-05 | 3 | Confortável com tensores |
| Dia 2 | 06-10 | 4 | Entende atenção |
| Dia 3 | 16 | 2 | Implementa self-att |
| Dia 4 | 33-34 | 3 | Treina LLM, gera texto |
| **Total** | **13 caps** | **12 horas** | **LLM do zero** |

---

## 🎯 Meta: Você Entenderá

Após completar Quick Start, você será capaz de:

1. ✅ Criar tensores e fazer operações
2. ✅ Explicar atenção a um amigo
3. ✅ Implementar self-attention manualmente
4. ✅ Treinar uma pequena LLM
5. ✅ Gerar texto com modelo treinado
6. ✅ Salvar/carregar checkpoints

---

## 🆘 Se Travar

1. **Erro de shape**: Veja "Erros Comuns" no capítulo
2. **NaN/Inf**: Verifique inicialização (Cap 09)
3. **Código não roda**: Verifique imports e versões PyTorch
4. **Não entendi conceito**: Releia "Intuição" seção do capítulo

---

## 🚀 Próximas Fases (Após Quick Start)

- **Fase 2**: Capítulos 11-32 (tokenização, otimizadores, etc)
- **Fase 3**: Treinar em dados reais (OpenWebText)
- **Fase 4**: Fine-tune para tarefa específica
- **Fase 5**: Contribuir para open source LLMs

---

## 💡 Dica Final

Não just read, **execute e modifique**. A maior parte do aprendizado vem de:

1. Rodar o código
2. Mudar um parâmetro
3. Ver o que muda
4. Entender o porquê

**Feliz aprendizado! 🎓**

---

Próximo passo: Abra `chapters/01_introducao_e_motivacao.md` e comece.
