const form = document.getElementById('analyze-form');
const resultEl = document.getElementById('result');
const docsEl = document.getElementById('documents');
const refreshBtn = document.getElementById('refresh-btn');

async function fetchDocuments() {
  const res = await fetch('/api/v1/documents');
  if (!res.ok) {
    docsEl.innerHTML = '<p>Failed to load documents.</p>';
    return;
  }

  const docs = await res.json();
  if (!docs.length) {
    docsEl.innerHTML = '<p>No analyzed documents yet.</p>';
    return;
  }

  docsEl.innerHTML = docs
    .map((doc) => {
      const confidence = doc.confidence != null ? (doc.confidence * 100).toFixed(2) : 'n/a';
      return `
        <article class="doc-card">
          <h3>${escapeHtml(doc.title)}</h3>
          <p>${escapeHtml(doc.content.slice(0, 180))}${doc.content.length > 180 ? '...' : ''}</p>
          <p class="doc-meta">Sentiment: <strong>${doc.label ?? 'n/a'}</strong> | Confidence: ${confidence}%</p>
        </article>
      `;
    })
    .join('');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const title = document.getElementById('title').value.trim();
  const content = document.getElementById('content').value.trim();

  const res = await fetch('/api/v1/documents/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, content }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: 'Request failed' }));
    resultEl.classList.remove('hidden');
    resultEl.innerHTML = `<strong>Error:</strong> ${escapeHtml(data.detail ?? 'Unknown error')}`;
    return;
  }

  const data = await res.json();
  resultEl.classList.remove('hidden');
  resultEl.innerHTML = `
    <strong>Prediction:</strong> ${escapeHtml(data.label)}<br />
    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br />
    <strong>Model:</strong> ${escapeHtml(data.model_name)} (${escapeHtml(data.model_version)})
  `;

  form.reset();
  await fetchDocuments();
});

refreshBtn.addEventListener('click', fetchDocuments);

fetchDocuments();
