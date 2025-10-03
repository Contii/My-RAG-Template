# My-RAG Template

Laboratório para desenvolvimento de sistemas de Retrieval-Augmented Generation (RAG) com LLMs leves, integrando modelos do [Hugging Face](https://huggingface.co/) e PyTorch.

## Características

- Arquitetura moderna para IA generativa baseada em busca e geração.
- Base para soluções escaláveis de NLP, pesquisa semântica e chatbots inteligentes.
- Estrutura modular e extensível, facilitando integração com diferentes fontes de dados e modelos.
- Foco em eficiência e execução local de LLMs.
- Pronto para expansão: múltiplos bancos de dados, adaptação de modelos, integração com APIs.
- Código limpo, organizado e pensado para evolução profissional.

## Aplicações

- Chatbots avançados
- Pesquisa semântica em *corpora extensa*
- Sistemas de recomendação baseados em linguagem natural

## Instalação

Consulte as [instruções detalhadas](./docs/INSTALL.md) de instalação e configuração do ambiente inicial.

## Configuração

Os principais parâmetros estão em `config/config.yaml`.  
Exemplo:

```yaml
llm_model: "microsoft/bitnet-b1.58-2B-4T"
retriever_type: "stub"
generator_type: "stub"
max_tokens: 250
data_path: "data/documents/"
temperature: 0.7
```

## Uso

Execute o pipeline principal:

```sh
python main.py
```

## Referências

- [Hugging Face](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)
- [Retrieval-Augmented Generation (RAG)](https://huggingface.co/docs/transformers/model_doc/rag)



