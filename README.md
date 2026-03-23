[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/4L0ralYh)

# AV02 - Busca por Similaridade com LLM (Tema: MPC / Papatinho)

Projeto completo para a atividade: pipeline de embeddings + indice vetorial + API + interface web.

## O que este projeto faz
1. Le URLs de `data/urls_mpc.txt` (3 a 10 paginas).
2. Faz scraping do conteudo textual.
3. Limpa e quebra o texto em chunks.
4. Gera embeddings para cada chunk.
5. Cria um indice vetorial FAISS.
6. Exponibiliza busca por API (`/api/search`).
7. Mostra resultados em front-end web.

## Estrutura
```txt
backend/
  app.py                    # FastAPI (API + servidor do front)
frontend/
  index.html                # interface web
  styles.css
  app.js
data/
  raw/scraped_docs.jsonl
  processed/chunks.jsonl
  processed/embeddings.npy
  index/faiss.index
  urls_mpc.txt
src/
  scraping.py
  preprocess.py
  chunking.py
  embeddings.py
  indexer.py
  search.py
  main.py                   # pipeline CLI
render.yaml                 # deploy Render
Procfile                    # deploy estilo Heroku/Render
requirements.txt
```

## Instalacao
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Etapa 1: construir o indice vetorial
```bash
python src/main.py --build-index
```

## Etapa 2: testar busca pelo terminal
```bash
python src/main.py --query "Qual e a proposta do projeto MPC?" --top-k 5
```

## Etapa 3: rodar back-end + front-end local
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Depois abra: `http://localhost:8000`

## Endpoints da API
- `GET /api/health`
- `GET /api/search?q=sua+pergunta&top_k=5`

## Deploy (extra +0,5)
### Opcao recomendada: Render (1 deploy, front + back juntos)
1. Suba este projeto no GitHub.
2. No Render, clique em **New +** -> **Web Service**.
3. Conecte seu repositorio.
4. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Deploy.
6. Teste no navegador:
   - `/` (front-end)
   - `/api/health` e `/api/search?...` (back-end)

## Parametros uteis da pipeline
- `--chunk-size 900`
- `--overlap 120`
- `--metric ip` (cosseno aproximado com normalizacao L2)
- `--model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `--urls-file data/urls_mpc.txt`

Para deploy no Render free (evitar OOM):
- `EMBEDDING_BACKEND=hf_api`
- `MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Observacoes importantes
- Algumas paginas podem bloquear scraping ou renderizar conteudo via JavaScript.
- Se uma URL falhar, ela sera ignorada e o pipeline continua.
- O primeiro build baixa o modelo e pode demorar.
- Para a entrega, mantenha entre 3 e 10 fontes validas e salve evidencias (print da web + print da API).
