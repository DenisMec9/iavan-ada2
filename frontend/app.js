const form = document.getElementById("search-form");
const queryEl = document.getElementById("query");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const buttonEl = document.getElementById("search-btn");
const DEFAULT_TOP_K = 5;

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeLabel(value, fallback) {
  const label = (value || "").toString().trim();
  return label || fallback;
}

function renderResults(items) {
  if (!items.length) {
    resultsEl.innerHTML = "<p>Nenhum resultado encontrado.</p>";
    return;
  }

  resultsEl.innerHTML = items
    .map((item, i) => {
      const preview = escapeHtml(normalizeLabel(item.excerpt, ""));
      const title = escapeHtml(normalizeLabel(item.title, "Sem titulo"));
      const url = escapeHtml(normalizeLabel(item.url, "#"));
      const sourceName = escapeHtml(normalizeLabel(item.source_name, "Fonte"));
      const sourceType = escapeHtml(
        normalizeLabel(item.source_type_label || item.source_type, "Fonte web"),
      );
      const relevance = escapeHtml(normalizeLabel(item.relevance, "moderada"));
      const score = Number(item.score || 0).toFixed(4);

      return `
        <article class="card" style="animation-delay:${i * 0.03}s">
          <div class="meta">
            <span class="chip">#${item.rank}</span>
            <span class="chip">${sourceName}</span>
            <span class="chip">${sourceType}</span>
            <span class="chip relevance-${relevance}">relevancia ${relevance}</span>
          </div>
          <div class="title">${title}</div>
          <div class="text">${preview}</div>
          <p class="source-row">
            <a href="${url}" target="_blank" rel="noreferrer">Abrir fonte</a>
            <span class="score">score ${score}</span>
          </p>
        </article>
      `;
    })
    .join("");
}

async function search(query, topK) {
  const params = new URLSearchParams({
    q: query,
    top_k: String(topK),
    _ts: String(Date.now()),
  });
  const res = await fetch(`/api/search?${params.toString()}`, {
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache",
      Pragma: "no-cache",
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erro na busca");
  }
  return res.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryEl.value.trim();
  const topK = DEFAULT_TOP_K;

  if (!query) {
    statusEl.textContent = "Digite uma pergunta antes de buscar.";
    return;
  }

  buttonEl.disabled = true;
  buttonEl.textContent = "Buscando...";
  statusEl.textContent = "Consultando indice...";

  try {
    const data = await search(query, topK);
    const notice = data.notice ? ` | ${data.notice}` : "";
    statusEl.textContent = `${data.count} resultado(s) para: "${query}"${notice}`;
    renderResults(data.results || []);
  } catch (error) {
    statusEl.textContent = error.message || "Erro inesperado.";
    resultsEl.innerHTML = "";
  } finally {
    buttonEl.disabled = false;
    buttonEl.textContent = "Buscar";
  }
});
