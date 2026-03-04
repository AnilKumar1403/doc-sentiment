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

const viewMeta = {
  dashboard: { title: 'Dashboard', subtitle: 'Your document sentiment overview.' },
  analyze: { title: 'Analyze', subtitle: 'Analyze direct text or uploaded files with OCR.' },
  history: { title: 'History', subtitle: 'Review all your processed documents and sentiments.' },
  profile: { title: 'Profile', subtitle: 'Your account details and access profile.' },
};

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

function showMessage(target, msg, isError = false) {
  target.classList.remove('hidden');
  target.style.borderColor = isError ? '#d36d6d' : '#b9d7c4';
  target.innerHTML = escapeHtml(msg);
}

async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const payload = await res.json().catch(() => ({ detail: 'Request failed' }));
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
  openView('dashboard');
}

function showAuth() {
  currentUser = null;
  appShell.classList.add('hidden');
  authScreen.classList.remove('hidden');
  setAuthMode('login');
}

function openView(viewName) {
  document.querySelectorAll('.view').forEach((view) => view.classList.add('hidden'));
  document.querySelectorAll('.menu-btn[data-view]').forEach((btn) => btn.classList.remove('active'));

  const viewEl = document.getElementById(`view-${viewName}`);
  if (!viewEl) return;

  viewEl.classList.remove('hidden');
  const btn = document.querySelector(`.menu-btn[data-view="${viewName}"]`);
  if (btn) btn.classList.add('active');

  viewTitle.textContent = viewMeta[viewName].title;
  viewSubtitle.textContent = viewMeta[viewName].subtitle;

  if (viewName === 'dashboard') loadDashboard();
  if (viewName === 'history') loadHistory();
}

async function loadDashboard() {
  try {
    const data = await apiFetch('/api/v1/dashboard/summary');
    document.getElementById('stat-total').textContent = data.total_documents;
    document.getElementById('stat-positive').textContent = data.positive_documents;
    document.getElementById('stat-negative').textContent = data.negative_documents;
    document.getElementById('stat-last').textContent = data.last_analysis_at
      ? new Date(data.last_analysis_at).toLocaleString()
      : '-';
  } catch (err) {
    showMessage(document.getElementById('analyze-result'), err.message, true);
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
        return `
          <article class="history-item">
            <h3>${escapeHtml(item.title)}</h3>
            <p class="muted tiny">${new Date(item.created_at).toLocaleString()} | Source: ${escapeHtml(item.source_type)}${item.file_name ? ` | File: ${escapeHtml(item.file_name)}` : ''}</p>
            <p>${escapeHtml((item.content || '').slice(0, 240))}${(item.content || '').length > 240 ? '...' : ''}</p>
            <p><span class="tag ${escapeHtml(item.label || '')}">${escapeHtml(item.label || 'n/a')}</span> Confidence: ${confidence} | Chars: ${item.extracted_char_count}</p>
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
    showApp(data.user);
  } catch (err) {
    showMessage(authMessage, err.message, true);
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
    showApp(data.user);
  } catch (err) {
    showMessage(authMessage, err.message, true);
  }
});

document.getElementById('text-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('analyze-result');
  const title = document.getElementById('text-title').value.trim();
  const content = document.getElementById('text-content').value.trim();

  try {
    const data = await apiFetch('/api/v1/documents/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content }),
    });
    showMessage(
      target,
      `Text analysis completed. Sentiment: ${data.label}, Confidence: ${(data.confidence * 100).toFixed(2)}%, Chars: ${data.extracted_char_count}`
    );
    event.target.reset();
    loadDashboard();
  } catch (err) {
    showMessage(target, err.message, true);
  }
});

document.getElementById('file-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('analyze-result');
  const title = document.getElementById('file-title').value.trim();
  const fileInput = document.getElementById('file-input');
  const file = fileInput.files[0];

  if (!file) {
    showMessage(target, 'Please select a file.', true);
    return;
  }

  const formData = new FormData();
  formData.append('title', title);
  formData.append('file', file);

  try {
    const data = await apiFetch('/api/v1/documents/analyze-file', {
      method: 'POST',
      body: formData,
    });
    showMessage(
      target,
      `File analysis completed. Sentiment: ${data.label}, Confidence: ${(data.confidence * 100).toFixed(2)}%, Source: ${data.source_type}, Chars: ${data.extracted_char_count}`
    );
    event.target.reset();
    loadDashboard();
  } catch (err) {
    showMessage(target, err.message, true);
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
  showAuth();
});

tabLogin.addEventListener('click', () => setAuthMode('login'));
tabRegister.addEventListener('click', () => setAuthMode('register'));

async function bootstrap() {
  try {
    const me = await apiFetch('/api/v1/auth/me');
    showApp(me);
  } catch (_err) {
    showAuth();
  }
}

bootstrap();
