# AV02 - Busca por Similaridade com LLM

## 1) Objetivo da atividade
Construir uma aplicacao em Python que encontre trechos de texto semanticamente parecidos com uma consulta (`query`) usando embeddings e busca por similaridade.

## 2) Entregavel esperado
- Codigo Python funcional da pipeline completa.
- Indice vetorial salvo localmente.
- Script para receber uma query e retornar os `n` chunks mais similares.
- Evidencias de teste (prints, logs e exemplos de busca).

## 3) Pipeline (como sera feita)
1. Scraping de 3 a 10 paginas de um mesmo tema.
2. Limpeza do texto (remover HTML, ruido e espacos extras).
3. Chunking (quebrar texto em blocos menores com sobreposicao).
4. Geracao de embeddings de cada chunk.
5. Indexacao vetorial (salvar vetores + metadados).
6. Entrada de query do usuario.
7. Embedding da query.
8. Busca dos `top-k` chunks mais proximos.
9. Exibicao dos resultados com score de similaridade e fonte.

## 4) Stack sugerida (Python)
- `requests` + `beautifulsoup4` (scraping)
- regex basica ou `nltk` (limpeza)
- `langchain-text-splitters` ou splitter manual (chunking)
- `sentence-transformers` (embeddings locais) ou API de embeddings
- `faiss-cpu` (indice vetorial)
- `numpy` e `pandas` (apoio)

## 5) Estrutura de pastas sugerida
```txt
av02-similaridade-llm/
  data/
    raw/                  # html/texto bruto
    processed/            # textos limpos/chunks
    index/                # indice FAISS e metadados
  src/
    scraping.py
    preprocess.py
    chunking.py
    embeddings.py
    indexer.py
    search.py
    main.py
  requirements.txt
  README.md
```

## 6) Especificacao de cada etapa

### 6.1 Scraping
- Receber lista de URLs (3 a 10).
- Baixar conteudo HTML.
- Extrair texto principal.
- Salvar em `data/raw/` com nome por pagina.

Saida: arquivos `.txt` ou `.json` com `url`, `titulo`, `texto`.

### 6.2 Limpeza
- Remover scripts, estilos e tags residuais.
- Normalizar espacos e quebras de linha.
- Opcional: remover caracteres especiais repetidos.

Saida: texto limpo pronto para chunking.

### 6.3 Chunking
- Definir `chunk_size` (ex.: 500-1000 caracteres).
- Definir `chunk_overlap` (ex.: 50-150 caracteres).
- Manter metadados por chunk: `id`, `url`, `titulo`, `posicao`.

Saida: lista de chunks em JSON ou CSV.

### 6.4 Embeddings
- Gerar vetor para cada chunk com modelo escolhido.
- Armazenar matriz de embeddings e metadados alinhados.

Saida: `embeddings.npy` + `metadata.json`.

### 6.5 Indexacao
- Criar indice vetorial (FAISS `IndexFlatIP` ou `IndexFlatL2`).
- Adicionar embeddings ao indice.
- Persistir indice em disco.

Saida: arquivo do indice em `data/index/`.

### 6.6 Busca por similaridade
- Receber query por terminal (input).
- Gerar embedding da query.
- Buscar `top_k` no indice.
- Retornar texto do chunk + score + fonte.

Saida: ranking dos chunks mais similares.

## 7) Checklist de avaliacao
- [ ] Fez scraping de 3 a 10 paginas
- [ ] Limpou e dividiu em chunks
- [ ] Gerou embeddings dos chunks
- [ ] Criou e salvou indice vetorial
- [ ] Recebe query e busca similares
- [ ] Mostra top-k com score e origem
- [ ] Codigo organizado e reproduzivel

## 8) Exemplo de fluxo de execucao
```bash
python src/main.py --build-index
python src/main.py --query "O que e busca por similaridade?" --top_k 5
```

## 9) Pseudocodigo (resumo)
```python
# build
urls = carregar_urls()
textos = [scrapear(url) for url in urls]
limpos = [limpar(t) for t in textos]
chunks = quebrar_em_chunks(limpos, chunk_size=800, overlap=100)
X = embedar(chunks)
indice = criar_indice(X)
salvar(indice, chunks)

# search
q = input("Digite sua query: ")
q_vec = embedar([q])
ids, scores = buscar(indice, q_vec, top_k=5)
mostrar_resultados(ids, scores, chunks)
```

## 10) Riscos comuns e como evitar
- Chunk grande demais: piora precisao. Use tamanho moderado.
- Sem overlap: pode quebrar contexto. Use sobreposicao.
- Dados sujos: embeddings ruins. Caprichar na limpeza.
- Modelo inadequado: testar ao menos 1 alternativa de embedding.

## 11) Cronograma rapido (para entregar hoje)
1. Implementar scraping + limpeza (30-45 min).
2. Implementar chunking + salvar metadados (20-30 min).
3. Integrar embeddings + FAISS (30-45 min).
4. Implementar busca por query e testar (20-30 min).
5. Ajustar README e evidencias finais (15-20 min).

## 12) Texto curto para relatorio
Nesta atividade, os textos coletados sao convertidos em vetores numericos por um modelo de embeddings. Esses vetores representam o significado semantico dos chunks. Quando o usuario envia uma query, ela tambem e transformada em vetor e comparada com os vetores do indice vetorial. Assim, o sistema retorna os trechos semanticamente mais proximos, mesmo sem depender apenas de palavras exatas.
