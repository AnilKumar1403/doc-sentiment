const authScreen = document.getElementById('auth-screen');
const appShell = document.getElementById('app-shell');
const authMessage = document.getElementById('auth-message');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const tabLogin = document.getElementById('tab-login');
const tabRegister = document.getElementById('tab-register');
const sidebarUser = document.getElementById('sidebar-user');
const viewTitle = document.getElementById('view-title');
const viewSubtitle = document.getElementById('view-subtitle');

let currentUser = null;
const ACCESS_TOKEN_KEY = 'sentiment_access_token';

const viewMeta = {
  dashboard: { title: 'Dashboard', subtitle: 'Your document emotion intelligence overview.' },
  analyze: { title: 'Analyze', subtitle: 'Analyze text/files with custom emotion metrics.' },
  history: { title: 'History', subtitle: 'Review all processed reports and suggestions.' },
  profile: { title: 'Profile', subtitle: 'Your account details and access profile.' },
};

const routeToView = {
  '/dashboard': 'dashboard',
  '/analyze': 'analyze',
  '/history': 'history',
  '/profile': 'profile',
  '/login': null,
};

const viewToRoute = {
  dashboard: '/dashboard',
  analyze: '/analyze',
  history: '/history',
  profile: '/profile',
};

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

function showMessage(target, msg, isError = false) {
  target.classList.remove('hidden');
  target.style.borderColor = isError ? '#d36d6d' : '#b9d7c4';
  target.innerHTML = msg;
}

async function apiFetch(url, options = {}) {
  const token = localStorage.getItem(ACCESS_TOKEN_KEY);
  const headers = new Headers(options.headers || {});
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({ detail: 'Request failed' }));
    if (res.status === 401) {
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      showAuth();
    }
    throw new Error(payload.detail || 'Request failed');
  }
  return res.json();
}

function setAuthMode(mode) {
  const isLogin = mode === 'login';
  loginForm.classList.toggle('hidden', !isLogin);
  registerForm.classList.toggle('hidden', isLogin);
  tabLogin.classList.toggle('active', isLogin);
  tabRegister.classList.toggle('active', !isLogin);
  authMessage.classList.add('hidden');
}

function showApp(user) {
  currentUser = user;
  authScreen.classList.add('hidden');
  appShell.classList.remove('hidden');
  sidebarUser.textContent = `${user.display_name} (${user.email})`;
  document.getElementById('profile-name').textContent = user.display_name;
  document.getElementById('profile-email').textContent = user.email;
  document.getElementById('profile-joined').textContent = new Date(user.created_at).toLocaleString();

  const pathView = routeToView[window.location.pathname] || 'dashboard';
  openView(pathView, { updateUrl: true });
}

function showAuth() {
  currentUser = null;
  appShell.classList.add('hidden');
  authScreen.classList.remove('hidden');
  setAuthMode('login');
  if (window.location.pathname !== '/login') {
    window.history.replaceState({}, '', '/login');
  }
}

function openView(viewName, options = {}) {
  const { updateUrl = true } = options;
  if (!currentUser) {
    showAuth();
    return;
  }

  document.querySelectorAll('.view').forEach((view) => view.classList.add('hidden'));
  document.querySelectorAll('.menu-btn[data-view]').forEach((btn) => btn.classList.remove('active'));

  const viewEl = document.getElementById(`view-${viewName}`);
  if (!viewEl) return;

  viewEl.classList.remove('hidden');
  const btn = document.querySelector(`.menu-btn[data-view="${viewName}"]`);
  if (btn) btn.classList.add('active');

  viewTitle.textContent = viewMeta[viewName].title;
  viewSubtitle.textContent = viewMeta[viewName].subtitle;

  if (updateUrl) {
    const targetPath = viewToRoute[viewName] || '/dashboard';
    if (window.location.pathname !== targetPath) {
      window.history.pushState({ view: viewName }, '', targetPath);
    }
  }

  if (viewName === 'dashboard') loadDashboard();
  if (viewName === 'history') loadHistory();
}

function formatEmotionScores(scores = []) {
  return scores
    .slice(0, 6)
    .map((item) => `${escapeHtml(item.emotion)} (${(item.score * 100).toFixed(1)}%)`)
    .join(', ');
}

function renderAnalyzeResult(data) {
  const scores = formatEmotionScores(data.emotion_scores || []);
  const suggestions = (data.suggestions || [])
    .map((s) => `<li>${escapeHtml(s)}</li>`)
    .join('');

  return `
    <div><strong>Primary Emotion:</strong> ${escapeHtml(data.label)} (${(data.confidence * 100).toFixed(2)}%)</div>
    <div><strong>Selected Metrics:</strong> ${escapeHtml((data.selected_metrics || []).join(', ') || 'all')}</div>
    <div><strong>Top Scores:</strong> ${scores || 'n/a'}</div>
    <div><strong>Summary:</strong> ${escapeHtml(data.summary || '')}</div>
    <div><strong>Suggestions:</strong><ul>${suggestions || '<li>No suggestions</li>'}</ul></div>
  `;
}

async function loadDashboard() {
  try {
    const data = await apiFetch('/api/v1/dashboard/summary');
    document.getElementById('stat-total').textContent = data.total_documents;
    document.getElementById('stat-alert').textContent = data.high_alert_documents;
    document.getElementById('stat-last').textContent = data.last_analysis_at
      ? new Date(data.last_analysis_at).toLocaleString()
      : '-';
    document.getElementById('stat-top-emotions').textContent = formatEmotionScores(data.top_emotions || []) || '-';
    await loadModelDetails();
  } catch (err) {
    showMessage(document.getElementById('analyze-result'), escapeHtml(err.message), true);
  }
}

async function loadModelDetails() {
  const metaEl = document.getElementById('model-meta');
  const metricsEl = document.getElementById('model-metrics');
  try {
    const details = await apiFetch('/api/v1/model/details');
    metaEl.textContent = `${details.model_name} (${details.model_version}) | labels: ${details.labels.length}`;
    const micro = details.train_metrics?.micro_f1;
    const macro = details.train_metrics?.macro_f1;
    const samples = details.train_metrics?.samples;
    metricsEl.textContent = `training metrics -> micro-F1: ${micro ?? 'n/a'}, macro-F1: ${macro ?? 'n/a'}, samples: ${samples ?? 'n/a'}`;
  } catch (err) {
    metaEl.textContent = 'Model details unavailable';
    metricsEl.textContent = String(err.message || '');
  }
}

async function loadHistory() {
  const historyList = document.getElementById('history-list');
  historyList.innerHTML = '<p class="muted">Loading...</p>';

  try {
    const items = await apiFetch('/api/v1/documents/history');
    if (!items.length) {
      historyList.innerHTML = '<p class="muted">No history yet.</p>';
      return;
    }

    historyList.innerHTML = items
      .map((item) => {
        const confidence = item.confidence != null ? `${(item.confidence * 100).toFixed(2)}%` : 'n/a';
        const topScores = formatEmotionScores(item.emotion_scores || []);
        const suggestions = (item.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');

        return `
          <article class="history-item">
            <h3>${escapeHtml(item.title)}</h3>
            <p class="muted tiny">${new Date(item.created_at).toLocaleString()} | Source: ${escapeHtml(item.source_type)}${item.file_name ? ` | File: ${escapeHtml(item.file_name)}` : ''}</p>
            <p><span class="tag">Primary: ${escapeHtml(item.label || 'n/a')}</span> Confidence: ${confidence}</p>
            <p><strong>Metrics:</strong> ${escapeHtml((item.selected_metrics || []).join(', ') || 'all')}</p>
            <p><strong>Top Scores:</strong> ${topScores || 'n/a'}</p>
            <p><strong>Summary:</strong> ${escapeHtml(item.summary || '')}</p>
            <ul>${suggestions || '<li>No suggestions</li>'}</ul>
          </article>
        `;
      })
      .join('');
  } catch (err) {
    historyList.innerHTML = `<p class="muted">${escapeHtml(err.message)}</p>`;
  }
}

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const email = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value;

  try {
    const data = await apiFetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    showApp(data.user);
  } catch (err) {
    showMessage(authMessage, escapeHtml(err.message), true);
  }
});

registerForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const display_name = document.getElementById('register-name').value.trim();
  const email = document.getElementById('register-email').value.trim();
  const password = document.getElementById('register-password').value;

  try {
    const data = await apiFetch('/api/v1/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name, email, password }),
    });
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    showApp(data.user);
  } catch (err) {
    showMessage(authMessage, escapeHtml(err.message), true);
  }
});

document.getElementById('text-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('analyze-result');
  const title = document.getElementById('text-title').value.trim();
  const content = document.getElementById('text-content').value.trim();
  const emotion_metrics = document.getElementById('text-metrics').value.trim();

  try {
    const data = await apiFetch('/api/v1/documents/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content, emotion_metrics }),
    });
    showMessage(target, renderAnalyzeResult(data));
    event.target.reset();
    loadDashboard();
  } catch (err) {
    showMessage(target, escapeHtml(err.message), true);
  }
});

document.getElementById('file-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('analyze-result');
  const title = document.getElementById('file-title').value.trim();
  const emotion_metrics = document.getElementById('file-metrics').value.trim();
  const fileInput = document.getElementById('file-input');
  const file = fileInput.files[0];

  if (!file) {
    showMessage(target, 'Please select a file.', true);
    return;
  }

  const formData = new FormData();
  formData.append('title', title);
  formData.append('emotion_metrics', emotion_metrics);
  formData.append('file', file);

  try {
    const data = await apiFetch('/api/v1/documents/analyze-file', {
      method: 'POST',
      body: formData,
    });
    showMessage(target, renderAnalyzeResult(data));
    event.target.reset();
    loadDashboard();
  } catch (err) {
    showMessage(target, escapeHtml(err.message), true);
  }
});

document.querySelectorAll('.menu-btn[data-view]').forEach((button) => {
  button.addEventListener('click', () => openView(button.dataset.view));
});

document.getElementById('history-refresh').addEventListener('click', loadHistory);

document.getElementById('logout-btn').addEventListener('click', async () => {
  try {
    await apiFetch('/api/v1/auth/logout', { method: 'POST' });
  } catch (_err) {
    // Ignore logout API errors and reset UI anyway.
  }
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  showAuth();
});

tabLogin.addEventListener('click', () => setAuthMode('login'));
tabRegister.addEventListener('click', () => setAuthMode('register'));

window.addEventListener('popstate', () => {
  const pathView = routeToView[window.location.pathname];
  if (!currentUser) {
    showAuth();
    return;
  }
  if (pathView) {
    openView(pathView, { updateUrl: false });
  } else {
    openView('dashboard');
  }
});

async function bootstrap() {
  const requestedPath = window.location.pathname;
  try {
    const me = await apiFetch('/api/v1/auth/me');
    showApp(me);
    if (requestedPath === '/login') {
      openView('dashboard');
    }
  } catch (_err) {
    showAuth();
  }
}

bootstrap();
