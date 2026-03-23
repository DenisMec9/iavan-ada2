# Resumo da Implementacao - AV02 (Busca por Similaridade com LLM)

## Objetivo entregue
Construimos uma aplicacao completa para busca semantica sobre o tema MPC (Musica Popular Carioca), com:
- pipeline de ingestao e indexacao vetorial;
- API de busca;
- interface web (front-end);
- arquivos de deploy para publicar a aplicacao.

## Pipeline de dados (CLI)
Implementamos fluxo ponta a ponta em Python:
1. leitura de URLs em `data/urls_mpc.txt`;
2. scraping e limpeza de texto;
3. chunking com overlap;
4. geracao de embeddings;
5. criacao de indice FAISS;
6. consulta por similaridade com top-k.

Arquivos principais:
- `src/main.py`
- `src/scraping.py`
- `src/chunking.py`
- `src/embeddings.py`
- `src/indexer.py`
- `src/search.py`
- `src/preprocess.py`

## API (Back-end)
Criamos API com FastAPI:
- `GET /api/health` para status;
- `GET /api/search` para consulta semantica;
- `GET /` servindo a interface web.

Melhorias de qualidade:
- limpeza adicional de encoding/mojibake;
- resumo de trecho (`excerpt`) mais legivel;
- metadados por resultado: nome da fonte, dominio, tipo de fonte e relevancia;
- rotulos didaticos (ex.: Fonte oficial, Fonte jornalistica, Base enciclopedica);
- filtro de baixa qualidade textual para evitar retorno ruim.

Arquivo principal:
- `backend/app.py`

## Front-end
Construimos interface web integrada a API:
- campo de pergunta;
- busca assíncrona;
- cards com titulo, trecho, fonte e score;
- visual responsivo e legivel para demonstracao.

Arquivos:
- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`
- `frontend/favicon.svg`

## Deploy
Deixamos projeto pronto para deploy (front + back no mesmo servico):
- `Procfile`
- `render.yaml`
- `requirements.txt`
- instrucoes em `README.md`

## Dados e index pronto
Incluimos artefatos necessarios para consulta imediata:
- `data/index/faiss.index`
- `data/processed/chunks.jsonl`

## Documentacao de apoio
- `AV02_Busca_por_Similaridade_com_LLM.md` (estrutura da atividade)
- `PLANO_CONTRIBUICAO_VITORIA.md` (divisao de contribuicoes reais)
- `README.md` (execucao local e deploy)
