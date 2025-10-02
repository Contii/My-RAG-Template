# Guia de Instalação e Uso

## 1. Crie o ambiente virtual

```bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
```

## 2. Instale dependências

```bash
pip install -U "huggingface_hub[cli]"
pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4
pip install accelerate
```

## 3. Faça login no Hugging Face

```bash
huggingface-cli login
```
Cole seu token Hugging Face quando solicitado (não adicione como credencial no projeto).

## 4. Instale PyTorch

Acesse [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) e selecione a versão adequada para seu sistema.

Exemplo para CUDA 12.8:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 5. Teste a instalação do PyTorch

```python
import torch
print(torch.__version__)
```

## 6. Baixe e teste o modelo

- Vá até a página do modelo desejado no Hugging Face.
- Copie o exemplo de código para um novo arquivo `test.py`.
- Adicione o argumento `device_map="auto"` na chamada de `AutoModelForCausalLM.from_pretrained`.

Exemplo:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nome-do-modelo", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("nome-do-modelo")
```

- Execute o arquivo usando o interpretador do ambiente virtual:

```bash
python test.py
```

## 7. Observações para Expansão

Para adicionar novos LLMs, repita o processo de instalação e ajuste o código conforme a documentação do modelo desejado.

Para integração com novas bases de dados, crie módulos específicos na estrutura do projeto.

Consulte a documentação oficial dos modelos e bibliotecas para dúvidas específicas e boas práticas.

- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/cli)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [Transformers GitHub](https://github.com/huggingface/transformers)
