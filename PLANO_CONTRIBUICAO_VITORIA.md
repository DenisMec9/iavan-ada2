# Plano de Contribuicao - Vitoria

Este plano foi montado com base no estado atual do projeto para evitar tarefas duplicadas.

## 1) O que ja esta feito (nao usar como tarefa)
- Front-end base pronto:
  - `frontend/index.html` com formulario de busca.
  - `frontend/app.js` consumindo `/api/search`.
  - `frontend/styles.css` com layout e cards.
- Back-end principal pronto:
  - `GET /api/health`
  - `GET /api/search`
  - limpeza de texto + excerpt + rotulos de fonte.
- Deploy ja preparado:
  - `Procfile`
  - `render.yaml`
- Dados ja processados:
  - `data/index/faiss.index`
  - `data/processed/chunks.jsonl`

## 2) Tarefas reais para a Vitoria (pendentes)

## Tarefa A - Front mais didatico
Arquivos:
- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`

Objetivo:
- Adicionar uma secao "Como interpretar os resultados" com texto curto.
- Adicionar "Perguntas sugeridas" (3 a 5 botoes clicaveis).
- Ao clicar em pergunta sugerida, preencher a textarea e executar busca.

Criterio de pronto:
- Secao explicativa aparece na tela.
- Botoes de pergunta sugerida funcionam.
- Nao quebrar busca manual existente.

Mensagem de commit sugerida:
- `feat(frontend): adiciona perguntas sugeridas e secao didatica de interpretacao`

## Tarefa B - API com filtro de fonte
Arquivo:
- `backend/app.py`

Objetivo:
- Incluir parametro opcional `source_type` em `/api/search`.
- Filtrar resultados por tipo quando esse parametro for enviado.
- Tipos aceitos: `enciclopedia`, `release-oficial`, `materia-critica`, `site-web`, `catalogo-musical`.

Criterio de pronto:
- `/api/search?q=mpc&source_type=materia-critica` retorna apenas esse tipo.
- Sem `source_type`, comportamento atual continua igual.

Mensagem de commit sugerida:
- `feat(api): adiciona filtro opcional por tipo de fonte na busca`

## Tarefa C - Endpoint de transparencia das fontes
Arquivo:
- `backend/app.py`

Objetivo:
- Criar endpoint `GET /api/sources`.
- Retornar resumo com contagem por dominio e por tipo de fonte.

Exemplo de saida esperada:
- total de chunks
- lista por dominio (`billboard.com.br`, `universalmusic.com.br`, etc.)
- lista por tipo (`materia-critica`, `release-oficial`, etc.)

Criterio de pronto:
- Endpoint responde JSON valido.
- Contagens batem com `data/processed/chunks.jsonl`.

Mensagem de commit sugerida:
- `feat(api): cria endpoint de resumo e contagem de fontes`

## Tarefa D - Evidencia para entrega
Arquivos novos:
- `docs/AVALIACAO_VITORIA.md`
- opcional: `docs/prints/` (imagens)

Objetivo:
- Documentar 8 a 10 queries reais e resumo dos resultados.
- Incluir observacoes curtas de qualidade (o que funcionou e o que pode melhorar).

Criterio de pronto:
- Arquivo com queries, trechos de retorno e conclusao.
- Material pronto para mostrar ao professor.

Mensagem de commit sugerida:
- `docs: adiciona avaliacao de consultas e evidencias de uso`

## 3) Ordem recomendada de trabalho dela
1. Tarefa A (front didatico)
2. Tarefa B (filtro no search)
3. Tarefa C (endpoint /api/sources)
4. Tarefa D (documentacao de evidencia)

## 4) Comandos rapidos para validar antes de commit
```bash
python -m compileall backend src
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Testes manuais:
- `http://localhost:8000`
- `http://localhost:8000/api/health`
- `http://localhost:8000/api/search?q=o%20que%20e%20mpc`
- `http://localhost:8000/api/search?q=mpc&source_type=materia-critica`
- `http://localhost:8000/api/sources`

## 5) Observacao importante
Evitar editar os arquivos de indice/dados (`data/index/faiss.index`, `data/processed/embeddings.npy`) para nao perder reproducibilidade.
