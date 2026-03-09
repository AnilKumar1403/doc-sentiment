const authScreen = document.getElementById('auth-screen');
const appShell = document.getElementById('app-shell');
const authMessage = document.getElementById('auth-message');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const authModeLoginBtn = document.getElementById('auth-mode-login');
const authModeRegisterBtn = document.getElementById('auth-mode-register');
const goRegisterBtn = document.getElementById('go-register-btn');
const goLoginBtn = document.getElementById('go-login-btn');
const sidebarUser = document.getElementById('sidebar-user');
const viewTitle = document.getElementById('view-title');
const viewSubtitle = document.getElementById('view-subtitle');

let currentUser = null;
const ACCESS_TOKEN_KEY = 'sentiment_access_token';
const API_BASE_STORAGE_KEY = 'aqualearning_api_base';

function normalizedApiBase(raw) {
  const value = String(raw || '').trim();
  if (!value) return '';
  return value.endsWith('/') ? value.slice(0, -1) : value;
}

function configuredApiBase() {
  const fromStorage = normalizedApiBase(window.localStorage.getItem(API_BASE_STORAGE_KEY));
  if (fromStorage) return fromStorage;

  const fromGlobal = normalizedApiBase(window.AQUALearning_API_BASE || window.AQUALEARNING_API_BASE);
  if (fromGlobal) return fromGlobal;

  const fromQuery = normalizedApiBase(new URLSearchParams(window.location.search).get('api_base'));
  if (fromQuery) {
    window.localStorage.setItem(API_BASE_STORAGE_KEY, fromQuery);
    return fromQuery;
  }
  return '';
}

const viewMeta = {
  dashboard: { title: 'Dashboard', subtitle: 'Analytics-only view for your account activity and model insights.' },
  sentiment: { title: 'Sentiment Module', subtitle: 'Dedicated sentiment flow with text/file analysis.' },
  relevance: { title: 'Relevance Studio', subtitle: 'Compare CV/document vs target context and generate revised resumes for JD alignment.' },
  learning: { title: 'Learning Coach', subtitle: 'Math and Indian Social storytelling, weak-topic analysis, and student Q&A solver.' },
  history: { title: 'History', subtitle: 'History-only view of completed analyses.' },
  profile: { title: 'Profile', subtitle: 'Profile-only view of login and account details.' },
};

const routeToView = {
  '/dashboard': 'dashboard',
  '/sentiment': 'sentiment',
  '/relevance': 'relevance',
  '/learning': 'learning',
  '/history': 'history',
  '/profile': 'profile',
  '/login': null,
};

const viewToRoute = {
  dashboard: '/dashboard',
  sentiment: '/sentiment',
  relevance: '/relevance',
  learning: '/learning',
  history: '/history',
  profile: '/profile',
};

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

function showMessage(target, html, isError = false) {
  target.classList.remove('hidden');
  target.style.borderColor = isError ? '#e2939e' : '#b7dfd0';
  target.innerHTML = html;
}

function userPlanLabel(user) {
  return user?.is_unlimited ? 'Unlimited Access' : 'Standard Access';
}

function creditsLabel(user) {
  if (!user) return '-';
  if (user.is_unlimited) return 'Unlimited';
  return String(user.credits_remaining ?? 0);
}

async function apiFetch(url, options = {}) {
  const token = localStorage.getItem(ACCESS_TOKEN_KEY);
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const tryFetch = async (targetUrl) => {
    try {
      return await fetch(targetUrl, { ...options, headers });
    } catch (_err) {
      return null;
    }
  };

  const isApiPath = typeof url === 'string' && url.startsWith('/api/');
  const overrideBase = configuredApiBase();
  let res = null;

  if (isApiPath && overrideBase) {
    res = await tryFetch(`${overrideBase}${url}`);
  }

  if (!res) {
    res = await tryFetch(url);
  }

  if ((!res || res.status === 404) && isApiPath) {
    const isHttpsPage = window.location.protocol === 'https:';
    let fallbackBases = [
      // Production backend (Render)
      'https://sentiment-backend-latest-r5du.onrender.com',
    
      // Local dev (optional - keep these if you run backend locally sometimes)
      'http://127.0.0.1:8000',
      'http://localhost:8000',
    ];
    if (overrideBase) {
      fallbackBases.unshift(overrideBase);
    }
    if (isHttpsPage) {
      fallbackBases = fallbackBases.filter((base) => base.startsWith('https://'));
    }
    fallbackBases = fallbackBases.filter(
      (base, idx) => base !== window.location.origin && fallbackBases.indexOf(base) === idx,
    );
    for (const base of fallbackBases) {
      const candidate = `${base}${url}`;
      const fallbackRes = await tryFetch(candidate);
      if (fallbackRes && fallbackRes.status !== 404) {
        res = fallbackRes;
        break;
      }
      if (fallbackRes && fallbackRes.ok) {
        res = fallbackRes;
        break;
      }
    }
  }

  if (!res) {
    if (window.location.protocol === 'https:') {
      throw new Error(
        'Backend not reachable from this HTTPS page. Set ?api_base=https://<backend-host> or localStorage aqualearning_api_base.',
      );
    }
    throw new Error('Backend not reachable. Ensure backend is running and API base/port is correct.');
  }
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

function isNotFoundError(err) {
  const msg = String(err?.message || '').toLowerCase();
  return msg.includes('not found') || msg.includes('404');
}

async function postJsonWithRouteFallback(paths, bodyObj) {
  let lastErr = null;
  const payload = JSON.stringify(bodyObj);
  for (const path of paths) {
    try {
      return await apiFetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      });
    } catch (err) {
      lastErr = err;
      if (!isNotFoundError(err)) throw err;
    }
  }
  throw lastErr || new Error('Requested API route was not found.');
}

async function postFormWithRouteFallback(paths, buildFormData) {
  let lastErr = null;
  for (const path of paths) {
    try {
      return await apiFetch(path, {
        method: 'POST',
        body: buildFormData(),
      });
    } catch (err) {
      lastErr = err;
      if (!isNotFoundError(err)) throw err;
    }
  }
  throw lastErr || new Error('Requested API route was not found.');
}

function buildFallbackResumeFromRelevance(data, context) {
  const candidateName = (context.candidate_name || '').trim() || (currentUser?.display_name || 'Candidate');
  const role = context.role || 'Target Role';
  const company = context.company || 'Target Company';
  const strengths = (data.strengths || []).slice(0, 6).map((s) => `- ${s}`).join('\n');
  const actions = (data.priority_actions || data.suggestions || []).slice(0, 8).map((s) => `- ${s}`).join('\n');
  const sourceCv = (context.cv_text || '').trim();

  return [
    `${candidateName}`,
    `Applying for: ${role} | ${company}`,
    '',
    'PROFESSIONAL SUMMARY',
    data.summary || 'Strong profile with clear fit to the target role.',
    '',
    'CORE ALIGNMENT HIGHLIGHTS',
    strengths || '- Demonstrated role-relevant experience and outcomes.',
    '',
    'STRATEGIC IMPROVEMENTS TO APPLY',
    actions || '- Strengthen role-specific achievements and measurable business outcomes.',
    '',
    'ORIGINAL CV SOURCE',
    sourceCv || '[Attach/paste CV text for a fuller revised draft]',
  ].join('\n');
}

function adaptRelevanceToResumeResult(data, context) {
  const resumeDraft = buildFallbackResumeFromRelevance(data, context);
  const improvementKeywords = (data.gaps || []).slice(0, 12);
  const fallbackMods = (data.priority_actions || []).slice(0, 6).map((action, idx) => ({
    line_number: idx + 1,
    current_line: 'Update required in role-alignment section',
    proposed_line: action,
    why_change: 'Improve clarity of relevance against target JD requirements.',
    impact: 'Improves recruiter readability and semantic ATS match.',
    priority: idx < 2 ? 'high' : 'medium',
  }));
  const fallbackActionPlan = (data.priority_actions || data.suggestions || []).slice(0, 8).map((item, idx) => ({
    area: idx < 2 ? 'Critical Alignment' : 'Optimization',
    where_to_add: idx < 3 ? 'Summary + Experience' : 'Experience + Projects',
    what_to_add: item,
    why_it_matters: 'Directly improves relevance against JD requirements.',
    expected_impact: 'Improves ATS and recruiter confidence.',
    estimated_score_lift: idx < 2 ? 4.0 : 2.0,
    sample_line: null,
    priority: idx < 3 ? 'high' : 'medium',
  }));
  const fallbackKeywordCoverage = (data.gaps || []).slice(0, 14).map((keyword, idx) => ({
    keyword,
    present_in_cv: false,
    recommended_section: idx < 5 ? 'CORE SKILLS + EXPERIENCE' : 'EXPERIENCE BULLETS',
    action: `Add '${keyword}' with a measurable proof point.`,
    priority: idx < 6 ? 'high' : 'medium',
  }));
  const relevanceScore = Number(data.relevance_score || 0);
  const targetScore = 75;
  return {
    title: context.title,
    role: context.role,
    company: context.company,
    relevance_score: relevanceScore,
    target_relevance_score: targetScore,
    gap_to_target: Math.max(0, targetScore - relevanceScore),
    estimated_post_update_score: Math.max(targetScore, relevanceScore + 12),
    baseline_summary: data.summary || '',
    detailed_strategy: data.detailed_summary || '',
    revised_resume: resumeDraft,
    revision_rationale: (data.priority_actions || data.suggestions || []).slice(0, 12),
    ats_keywords_added: improvementKeywords,
    strategic_action_plan: fallbackActionPlan,
    jd_keyword_coverage: fallbackKeywordCoverage,
    line_level_modifications: fallbackMods,
    generated_cover_letter: data.generated_cover_letter || null,
    credits_remaining: currentUser?.is_unlimited ? null : Number(currentUser?.credits_remaining ?? 0),
    is_unlimited: Boolean(currentUser?.is_unlimited),
    llm_enhanced: Boolean(data.llm_enhanced),
  };
}

function renderUserIdentity(user) {
  const plan = userPlanLabel(user);
  const credits = creditsLabel(user);
  sidebarUser.textContent = `${user.display_name} (${user.email}) | ${plan} | Credits: ${credits}`;
  document.getElementById('profile-name').textContent = user.display_name;
  document.getElementById('profile-email').textContent = user.email;
  document.getElementById('profile-plan').textContent = plan;
  document.getElementById('profile-credits').textContent = credits;
  document.getElementById('profile-joined').textContent = new Date(user.created_at).toLocaleString();
}

function setAuthMode(mode) {
  const isRegister = mode === 'register';
  loginForm.classList.toggle('hidden', isRegister);
  registerForm.classList.toggle('hidden', !isRegister);
  authModeLoginBtn.classList.toggle('active', !isRegister);
  authModeRegisterBtn.classList.toggle('active', isRegister);
  authMessage.classList.add('hidden');
}

async function refreshCurrentUser() {
  if (!currentUser) return;
  try {
    const me = await apiFetch('/api/v1/auth/me');
    currentUser = me;
    renderUserIdentity(me);
  } catch (_err) {}
}

function showApp(user) {
  currentUser = user;
  authScreen.classList.add('hidden');
  appShell.classList.remove('hidden');
  renderUserIdentity(user);
  const pathView = routeToView[window.location.pathname] || 'dashboard';
  openView(pathView, { updateUrl: true });
}

function showAuth() {
  currentUser = null;
  appShell.classList.add('hidden');
  authScreen.classList.remove('hidden');
  setAuthMode('login');
  if (window.location.pathname !== '/login') window.history.replaceState({}, '', '/login');
}

function openView(viewName, options = {}) {
  const { updateUrl = true } = options;
  if (!currentUser) return showAuth();

  document.querySelectorAll('.view').forEach((v) => v.classList.add('hidden'));
  document.querySelectorAll('.menu-btn[data-view]').forEach((b) => b.classList.remove('active'));

  const viewEl = document.getElementById(`view-${viewName}`);
  if (!viewEl) return;

  viewEl.classList.remove('hidden');
  const btn = document.querySelector(`.menu-btn[data-view="${viewName}"]`);
  if (btn) btn.classList.add('active');

  viewTitle.textContent = viewMeta[viewName].title;
  viewSubtitle.textContent = viewMeta[viewName].subtitle;

  if (updateUrl) {
    const targetPath = viewToRoute[viewName] || '/dashboard';
    if (window.location.pathname !== targetPath) window.history.pushState({ view: viewName }, '', targetPath);
  }

  if (viewName === 'dashboard') loadDashboard();
  if (viewName === 'history') loadHistory();
}

function formatEmotionScores(scores = []) {
  return scores.slice(0, 6).map((i) => `${escapeHtml(i.emotion)} (${(i.score * 100).toFixed(1)}%)`).join(', ');
}

function formatHistoryScore(item) {
  if (item.score == null) return 'n/a';
  if (item.module === 'sentiment') return `${(Number(item.score) * 100).toFixed(2)}% confidence`;
  if (item.module === 'learning') return `${Number(item.score).toFixed(2)} mastery`;
  return `${Number(item.score).toFixed(2)} relevance`;
}

function renderModuleAnalytics(items = []) {
  if (!items.length) return '<p class="muted">No analytics yet.</p>';
  return items.map((m) => `
    <article class="history-item">
      <p><span class="tag">${escapeHtml((m.module || 'module').toUpperCase())}</span></p>
      <p><strong>Total Analyses:</strong> ${Number(m.total_analyses || 0)}</p>
      <p><strong>Average Score:</strong> ${m.average_score != null ? Number(m.average_score).toFixed(2) : 'n/a'}</p>
      <p class="muted tiny"><strong>Last Run:</strong> ${m.last_run_at ? new Date(m.last_run_at).toLocaleString() : '-'}</p>
    </article>
  `).join('');
}

async function loadDashboard() {
  const data = await apiFetch('/api/v1/dashboard/summary');
  document.getElementById('stat-total').textContent = data.total_documents;
  document.getElementById('stat-total-analyses').textContent = data.total_analyses ?? 0;
  document.getElementById('stat-alert').textContent = data.high_alert_documents;
  document.getElementById('stat-last').textContent = data.last_analysis_at ? new Date(data.last_analysis_at).toLocaleString() : '-';
  document.getElementById('stat-top-emotions').textContent = formatEmotionScores(data.top_emotions || []) || '-';
  document.getElementById('dashboard-module-analytics').innerHTML = renderModuleAnalytics(data.module_analytics || []);
  await loadModelDetails();
}

async function loadModelDetails() {
  const meta = document.getElementById('model-meta');
  const metrics = document.getElementById('model-metrics');
  try {
    const d = await apiFetch('/api/v1/model/details');
    meta.textContent = `${d.model_name} (${d.model_version}) | labels: ${d.labels.length}`;
    metrics.textContent = `micro-F1: ${d.train_metrics?.micro_f1 ?? 'n/a'}, macro-F1: ${d.train_metrics?.macro_f1 ?? 'n/a'}, samples: ${d.train_metrics?.samples ?? 'n/a'}`;
  } catch (err) {
    meta.textContent = 'Model details unavailable';
    metrics.textContent = String(err.message || '');
  }
}

async function loadHistory() {
  const list = document.getElementById('history-list');
  list.innerHTML = '<p class="muted">Loading...</p>';
  try {
    const items = await apiFetch('/api/v1/documents/history');
    if (!items.length) {
      list.innerHTML = '<p class="muted">No history yet.</p>';
      return;
    }
    list.innerHTML = items.map((item) => {
      const moduleTag = escapeHtml((item.module || 'general').toUpperCase());
      const suggestions = (item.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
      const details = item.details || {};
      let detailBlock = '';
      if (item.module === 'sentiment') {
        detailBlock = `<p><strong>Top Scores:</strong> ${formatEmotionScores(details.emotion_scores || []) || 'n/a'}</p>`;
      } else if (item.module === 'relevance') {
        if ((item.label || '').toLowerCase() === 'resume_generation' || details.revised_resume) {
          const rationale = (item.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const keywords = (details.ats_keywords_added || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const strategicPlan = (details.strategic_action_plan || []).map((a, idx) => `
            <article class="resume-mod-item">
              <p><strong>Action ${idx + 1} | Priority:</strong> ${escapeHtml((a.priority || 'medium').toUpperCase())}</p>
              <p><strong>Area:</strong> ${escapeHtml(a.area || '')}</p>
              <p><strong>Where:</strong> ${escapeHtml(a.where_to_add || '')}</p>
              <p><strong>What to Add:</strong> ${escapeHtml(a.what_to_add || '')}</p>
              <p><strong>Why:</strong> ${escapeHtml(a.why_it_matters || '')}</p>
              <p><strong>Expected Impact:</strong> ${escapeHtml(a.expected_impact || '')}</p>
              <p><strong>Estimated Score Lift:</strong> ${Number(a.estimated_score_lift || 0).toFixed(2)}%</p>
              ${a.sample_line ? `<p><strong>Sample Line:</strong> ${escapeHtml(a.sample_line)}</p>` : ''}
            </article>
          `).join('');
          const keywordCoverage = (details.jd_keyword_coverage || []).map((k) => `
            <li>
              <strong>${escapeHtml(k.keyword || '')}</strong> | Present: ${k.present_in_cv ? 'Yes' : 'No'} |
              Section: ${escapeHtml(k.recommended_section || '')} | Priority: ${escapeHtml((k.priority || 'medium').toUpperCase())}
              <br />Action: ${escapeHtml(k.action || '')}
            </li>
          `).join('');
          const mods = (details.line_level_modifications || []).map((m) => `
            <article class="resume-mod-item">
              <p><strong>Line ${Number(m.line_number || 0)} | Priority:</strong> ${escapeHtml((m.priority || 'medium').toUpperCase())}</p>
              <p><strong>Current:</strong> ${escapeHtml(m.current_line || '')}</p>
              <p><strong>Proposed:</strong> ${escapeHtml(m.proposed_line || '')}</p>
              <p><strong>Why:</strong> ${escapeHtml(m.why_change || '')}</p>
              <p><strong>Impact:</strong> ${escapeHtml(m.impact || '')}</p>
            </article>
          `).join('');
          detailBlock = `
            <p><strong>Current Score:</strong> ${Number(item.score || 0).toFixed(2)}% | <strong>Target:</strong> ${Number(details.target_relevance_score || 75).toFixed(2)}% | <strong>Gap:</strong> ${Number(details.gap_to_target || 0).toFixed(2)}% | <strong>Projected:</strong> ${Number(details.estimated_post_update_score || item.score || 0).toFixed(2)}%</p>
            <pre>${escapeHtml(details.detailed_strategy || '')}</pre>
            <p><strong>Strategic Action Plan:</strong></p>
            ${strategicPlan || '<p class="muted">No strategic plan captured.</p>'}
            <p><strong>JD Keyword Coverage:</strong></p><ul>${keywordCoverage || '<li>n/a</li>'}</ul>
            <p><strong>Line-by-Line Modification Plan:</strong></p>
            ${mods || '<p class="muted">No line-level modifications captured.</p>'}
            <p><strong>ATS Keywords Added:</strong></p><ul>${keywords || '<li>n/a</li>'}</ul>
            <p><strong>Revision Rationale:</strong></p><ul>${rationale || '<li>n/a</li>'}</ul>
            <div><strong>Revised Resume:</strong><pre>${escapeHtml(details.revised_resume || '')}</pre></div>
          `;
        } else {
          const priorities = (details.priority_actions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const risks = (details.risk_flags || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const detailedSummary = details.detailed_summary ? `<pre>${escapeHtml(details.detailed_summary)}</pre>` : '';
          detailBlock = `
            ${detailedSummary}
            <p><strong>Tone:</strong> ${escapeHtml(details.communication_tone || 'n/a')}</p>
            <p><strong>Priority Actions:</strong></p><ul>${priorities || '<li>n/a</li>'}</ul>
            <p><strong>Risk Flags:</strong></p><ul>${risks || '<li>n/a</li>'}</ul>
          `;
        }
      } else if (item.module === 'learning') {
        if ((item.label || '').toLowerCase() === 'learning_qa' || details.question_text) {
          const steps = (details.logical_steps || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const concepts = (details.key_concepts || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const mistakes = (details.common_mistakes || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          const refs = (details.references || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          detailBlock = `
            <p><strong>Question:</strong> ${escapeHtml(details.question_text || 'n/a')}</p>
            <div class="qa-compare-grid">
              <article class="qa-box qa-box-student">
                <h4>Student Current Answer</h4>
                <p>${escapeHtml(details.current_answer || 'Not provided')}</p>
              </article>
              <article class="qa-box qa-box-correct">
                <h4>Correct Answer</h4>
                <p>${escapeHtml(details.correct_answer || 'n/a')}</p>
              </article>
            </div>
            <p><strong>Verdict:</strong> <span class="qa-verdict qa-verdict-${escapeHtml(details.answer_verdict || 'review_required')}">${escapeHtml(details.answer_verdict || 'review_required')}</span></p>
            <pre>${escapeHtml(details.answer_feedback || '')}</pre>
            <pre>${escapeHtml(details.detailed_explanation || '')}</pre>
            <p><strong>Logical Steps:</strong></p><ul>${steps || '<li>n/a</li>'}</ul>
            <p><strong>Key Concepts:</strong></p><ul>${concepts || '<li>n/a</li>'}</ul>
            <p><strong>Common Mistakes:</strong></p><ul>${mistakes || '<li>n/a</li>'}</ul>
            <p><strong>References:</strong></p><ul>${refs || '<li>n/a</li>'}</ul>
          `;
        } else {
          const retained = (details.retained_topics || []).join(', ');
          const weak = (details.weak_topics || []).join(', ');
          const plan = (details.study_plan || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
          detailBlock = `
            <p><strong>Retained Topics:</strong> ${escapeHtml(retained || 'n/a')}</p>
            <p><strong>Weak Topics:</strong> ${escapeHtml(weak || 'n/a')}</p>
            <p><strong>Study Plan:</strong></p><ul>${plan || '<li>n/a</li>'}</ul>
          `;
        }
      }
      return `
        <article class="history-item">
          <h3>${escapeHtml(item.title)}</h3>
          <p class="muted tiny">${new Date(item.created_at).toLocaleString()} | ${escapeHtml(item.source_type)}</p>
          <p><span class="tag">${moduleTag}</span> <span class="tag">${escapeHtml(item.analysis_type || 'analysis')}</span> <span class="tag">${escapeHtml(item.label || 'n/a')}</span></p>
          <p><strong>Score:</strong> ${escapeHtml(formatHistoryScore(item))}</p>
          <pre>${escapeHtml(item.summary || '')}</pre>
          ${detailBlock}
          <ul>${suggestions || '<li>No suggestions</li>'}</ul>
        </article>
      `;
    }).join('');
  } catch (err) {
    list.innerHTML = `<p class="muted">${escapeHtml(err.message)}</p>`;
  }
}

function renderRelevanceResult(data) {
  const metricRows = Object.entries(data.metrics || {}).map(([k, v]) => `<li>${escapeHtml(k)}: ${Number(v).toFixed(2)}</li>`).join('');
  const suggestions = (data.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const priority = (data.priority_actions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const risks = (data.risk_flags || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const strengths = (data.strengths || []).slice(0, 10).join(', ');
  const gaps = (data.gaps || []).slice(0, 10).join(', ');
  const cover = data.generated_cover_letter ? `<pre>${escapeHtml(data.generated_cover_letter)}</pre>` : '';
  const detailed = data.detailed_summary ? `<pre>${escapeHtml(data.detailed_summary)}</pre>` : '';
  return `
    <div><strong>Relevance Score:</strong> ${Number(data.relevance_score).toFixed(2)}%</div>
    <div><strong>Type:</strong> ${escapeHtml(data.analysis_type)} | <strong>LLM Enhanced:</strong> ${data.llm_enhanced ? 'Yes' : 'No'}</div>
    <pre>${escapeHtml(data.summary)}</pre>
    ${detailed}
    <div><strong>Communication Tone:</strong> ${escapeHtml(data.communication_tone || 'n/a')}</div>
    <div><strong>Strengths:</strong> ${escapeHtml(strengths || 'n/a')}</div>
    <div><strong>Gaps:</strong> ${escapeHtml(gaps || 'n/a')}</div>
    <div><strong>Metrics:</strong><ul>${metricRows}</ul></div>
    <div><strong>Priority Actions:</strong><ul>${priority || '<li>n/a</li>'}</ul></div>
    <div><strong>Risk Flags:</strong><ul>${risks || '<li>n/a</li>'}</ul></div>
    <div><strong>Suggestions:</strong><ul>${suggestions}</ul></div>
    ${cover ? `<div><strong>Generated Cover Letter:</strong>${cover}</div>` : ''}
  `;
}

function renderResumeGenerationResult(data) {
  const rationale = (data.revision_rationale || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const keywords = (data.ats_keywords_added || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const strategicPlan = (data.strategic_action_plan || []).map((a, idx) => `
    <article class="resume-mod-item">
      <p><strong>Action ${idx + 1} | Priority:</strong> ${escapeHtml((a.priority || 'medium').toUpperCase())}</p>
      <p><strong>Area:</strong> ${escapeHtml(a.area || '')}</p>
      <p><strong>Where:</strong> ${escapeHtml(a.where_to_add || '')}</p>
      <p><strong>What to Add:</strong> ${escapeHtml(a.what_to_add || '')}</p>
      <p><strong>Why:</strong> ${escapeHtml(a.why_it_matters || '')}</p>
      <p><strong>Expected Impact:</strong> ${escapeHtml(a.expected_impact || '')}</p>
      <p><strong>Estimated Score Lift:</strong> ${Number(a.estimated_score_lift || 0).toFixed(2)}%</p>
      ${a.sample_line ? `<p><strong>Sample Line:</strong> ${escapeHtml(a.sample_line)}</p>` : ''}
    </article>
  `).join('');
  const keywordCoverage = (data.jd_keyword_coverage || []).map((k) => `
    <li>
      <strong>${escapeHtml(k.keyword || '')}</strong> | Present: ${k.present_in_cv ? 'Yes' : 'No'} |
      Section: ${escapeHtml(k.recommended_section || '')} | Priority: ${escapeHtml((k.priority || 'medium').toUpperCase())}
      <br />Action: ${escapeHtml(k.action || '')}
    </li>
  `).join('');
  const mods = (data.line_level_modifications || []).map((m) => `
    <article class="resume-mod-item">
      <p><strong>Line ${Number(m.line_number || 0)} | Priority:</strong> ${escapeHtml((m.priority || 'medium').toUpperCase())}</p>
      <p><strong>Current:</strong> ${escapeHtml(m.current_line || '')}</p>
      <p><strong>Proposed:</strong> ${escapeHtml(m.proposed_line || '')}</p>
      <p><strong>Why:</strong> ${escapeHtml(m.why_change || '')}</p>
      <p><strong>Impact:</strong> ${escapeHtml(m.impact || '')}</p>
    </article>
  `).join('');
  const credits = data.is_unlimited ? 'Unlimited' : String(data.credits_remaining ?? 0);
  const plan = data.is_unlimited ? 'Unlimited Access' : 'Standard Access';
  const cover = data.generated_cover_letter ? `<pre>${escapeHtml(data.generated_cover_letter)}</pre>` : '';
  return `
    <div><strong>Role:</strong> ${escapeHtml(data.role || 'n/a')} | <strong>Company:</strong> ${escapeHtml(data.company || 'n/a')}</div>
    <div><strong>Relevance Score:</strong> ${Number(data.relevance_score).toFixed(2)}% | <strong>Target:</strong> ${Number(data.target_relevance_score || 75).toFixed(2)}%+ | <strong>Gap:</strong> ${Number(data.gap_to_target || 0).toFixed(2)}% | <strong>Projected Post-Update:</strong> ${Number(data.estimated_post_update_score || data.relevance_score).toFixed(2)}%</div>
    <div><strong>Plan:</strong> ${plan} | <strong>Credits Remaining:</strong> ${escapeHtml(credits)}</div>
    <pre>${escapeHtml(data.baseline_summary || '')}</pre>
    <pre>${escapeHtml(data.detailed_strategy || '')}</pre>
    <div><strong>Strategic Action Plan (to exceed 75% relevance):</strong>${strategicPlan || '<p class="muted">No strategic actions generated.</p>'}</div>
    <div><strong>JD Keyword Coverage Matrix:</strong><ul>${keywordCoverage || '<li>n/a</li>'}</ul></div>
    <div><strong>Line-by-Line Modification Plan:</strong>${mods || '<p class="muted">No line-level modifications generated.</p>'}</div>
    <div><strong>Revision Rationale:</strong><ul>${rationale || '<li>n/a</li>'}</ul></div>
    <div><strong>ATS Keywords Added:</strong><ul>${keywords || '<li>n/a</li>'}</ul></div>
    <div><strong>Revised Resume:</strong><pre>${escapeHtml(data.revised_resume || '')}</pre></div>
    ${cover ? `<div><strong>Generated Cover Letter:</strong>${cover}</div>` : ''}
  `;
}

function renderLearningResult(data) {
  const retained = (data.retained_topics || []).join(', ');
  const weak = (data.weak_topics || []).join(', ');
  const suggestions = (data.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const plan = (data.study_plan || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  return `
    <div><strong>Subject:</strong> ${escapeHtml(data.subject)} | <strong>LLM Enhanced:</strong> ${data.llm_enhanced ? 'Yes' : 'No'}</div>
    <div><strong>Mastery Score:</strong> ${data.mastery_score != null ? Number(data.mastery_score).toFixed(2) : 'n/a'}</div>
    <pre>${escapeHtml(data.storytelling_summary)}</pre>
    <pre>${escapeHtml(data.detailed_feedback || '')}</pre>
    <div><strong>Retained Topics:</strong> ${escapeHtml(retained || 'n/a')}</div>
    <div><strong>Weak Topics:</strong> ${escapeHtml(weak || 'n/a')}</div>
    <div><strong>Study Plan:</strong><ul>${plan || '<li>n/a</li>'}</ul></div>
    <div><strong>Suggestions:</strong><ul>${suggestions}</ul></div>
  `;
}

function renderLearningQAResult(data) {
  const steps = (data.logical_steps || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const concepts = (data.key_concepts || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const mistakes = (data.common_mistakes || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const practice = (data.practice_questions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  const refs = (data.references || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  return `
    <div><strong>Subject:</strong> ${escapeHtml(data.subject)} | <strong>Complexity:</strong> ${escapeHtml(data.complexity_level || 'n/a')} | <strong>LLM Enhanced:</strong> ${data.llm_enhanced ? 'Yes' : 'No'}</div>
    <p><strong>Question:</strong> ${escapeHtml(data.question_text || '')}</p>
    <div class="qa-compare-grid">
      <article class="qa-box qa-box-student">
        <h4>Student Current Answer</h4>
        <p>${escapeHtml(data.current_answer || 'Not provided')}</p>
      </article>
      <article class="qa-box qa-box-correct">
        <h4>Correct Answer</h4>
        <p>${escapeHtml(data.correct_answer || 'n/a')}</p>
      </article>
    </div>
    <p><strong>Verdict:</strong> <span class="qa-verdict qa-verdict-${escapeHtml(data.answer_verdict || 'review_required')}">${escapeHtml(data.answer_verdict || 'review_required')}</span></p>
    <pre>${escapeHtml(data.answer_feedback || '')}</pre>
    <pre>${escapeHtml(data.concise_answer || '')}</pre>
    <pre>${escapeHtml(data.detailed_explanation || '')}</pre>
    <div><strong>Logical Steps:</strong><ul>${steps || '<li>n/a</li>'}</ul></div>
    <div><strong>Key Concepts:</strong><ul>${concepts || '<li>n/a</li>'}</ul></div>
    <div><strong>Common Mistakes:</strong><ul>${mistakes || '<li>n/a</li>'}</ul></div>
    <div><strong>Practice Questions:</strong><ul>${practice || '<li>n/a</li>'}</ul></div>
    <div><strong>References:</strong><ul>${refs || '<li>n/a</li>'}</ul></div>
  `;
}

function renderSentimentResult(data) {
  const suggestions = (data.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join('');
  return `
    <div><strong>Primary Emotion:</strong> ${escapeHtml(data.label)} (${(data.confidence * 100).toFixed(2)}%)</div>
    <div><strong>Selected Metrics:</strong> ${escapeHtml((data.selected_metrics || []).join(', ') || 'all')}</div>
    <div><strong>Top Scores:</strong> ${formatEmotionScores(data.emotion_scores || []) || 'n/a'}</div>
    <div><strong>Summary:</strong> ${escapeHtml(data.summary || '')}</div>
    <div><strong>Suggestions:</strong><ul>${suggestions || '<li>No suggestions</li>'}</ul></div>
  `;
}

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const data = await apiFetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: document.getElementById('login-email').value.trim(),
        password: document.getElementById('login-password').value,
      }),
    });
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    showApp(data.user);
  } catch (err) {
    showMessage(authMessage, escapeHtml(err.message), true);
  }
});

registerForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const displayName = document.getElementById('register-name').value.trim();
  const email = document.getElementById('register-email').value.trim();
  const password = document.getElementById('register-password').value;
  const passwordConfirm = document.getElementById('register-password-confirm').value;

  if (password !== passwordConfirm) {
    showMessage(authMessage, 'Passwords do not match.', true);
    return;
  }

  try {
    const data = await postJsonWithRouteFallback(
      [
        '/api/v1/auth/register',
        '/api/v1/auth/register/',
        '/api/auth/register',
        '/auth/register',
      ],
      { display_name: displayName, email, password },
    );
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    showApp(data.user);
  } catch (err) {
    const msg = String(err?.message || '');
    if (msg.toLowerCase().includes('already exists')) {
      showMessage(authMessage, 'Account already exists. Please login with your email and password.', true);
      setAuthMode('login');
      document.getElementById('login-email').value = email;
      return;
    }
    showMessage(authMessage, escapeHtml(msg), true);
  }
});

authModeLoginBtn.addEventListener('click', () => setAuthMode('login'));
authModeRegisterBtn.addEventListener('click', () => setAuthMode('register'));
goRegisterBtn.addEventListener('click', () => setAuthMode('register'));
goLoginBtn.addEventListener('click', () => setAuthMode('login'));

document.getElementById('relevance-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('relevance-result');

  const docFile = document.getElementById('rel-doc-file').files[0];
  const refFile = document.getElementById('rel-ref-file').files[0];
  const docText = document.getElementById('rel-doc').value;
  const refText = document.getElementById('rel-ref').value;

  try {
    let data;
    if (docFile || refFile) {
      const fd = new FormData();
      fd.append('title', document.getElementById('rel-title').value.trim());
      fd.append('analysis_type', document.getElementById('rel-type').value);
      fd.append('role', document.getElementById('rel-role').value.trim());
      fd.append('company', document.getElementById('rel-company').value.trim());
      fd.append('context_notes', document.getElementById('rel-notes').value);
      fd.append('document_text', docText);
      fd.append('reference_text', refText);
      if (docFile) fd.append('document_file', docFile);
      if (refFile) fd.append('reference_file', refFile);
      data = await apiFetch('/api/v1/relevance/analyze-file', { method: 'POST', body: fd });
    } else {
      data = await apiFetch('/api/v1/relevance/analyze-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: document.getElementById('rel-title').value.trim(),
          analysis_type: document.getElementById('rel-type').value,
          role: document.getElementById('rel-role').value.trim(),
          company: document.getElementById('rel-company').value.trim(),
          document_text: docText,
          reference_text: refText,
          context_notes: document.getElementById('rel-notes').value,
        }),
      });
    }
    showMessage(target, renderRelevanceResult(data));
    loadHistory();
    loadDashboard();
  } catch (err) {
    const msg = (err.message || '').toLowerCase().includes('not found')
      ? 'Relevance endpoint was not found. Verify backend is running and restart if needed.'
      : err.message;
    showMessage(target, escapeHtml(msg), true);
  }
});

document.getElementById('resume-generate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('resume-generate-result');

  const cvFile = document.getElementById('resume-cv-file').files[0];
  const jdFile = document.getElementById('resume-jd-file').files[0];
  const cvText = document.getElementById('resume-cv-text').value;
  const jdText = document.getElementById('resume-jd-text').value;

  try {
    const requestContext = {
      title: document.getElementById('resume-title').value.trim(),
      role: document.getElementById('resume-role').value.trim(),
      company: document.getElementById('resume-company').value.trim(),
      candidate_name: document.getElementById('resume-candidate-name').value.trim(),
      context_notes: document.getElementById('resume-context-notes').value.trim(),
      cv_text: cvText,
      jd_text: jdText,
    };
    let data;
    if (cvFile || jdFile) {
      const buildResumeForm = () => {
        const fd = new FormData();
        fd.append('title', requestContext.title);
        fd.append('role', requestContext.role);
        fd.append('company', requestContext.company);
        fd.append('candidate_name', requestContext.candidate_name);
        fd.append('context_notes', requestContext.context_notes);
        fd.append('cv_text', requestContext.cv_text);
        fd.append('jd_text', requestContext.jd_text);
        if (cvFile) fd.append('cv_file', cvFile);
        if (jdFile) fd.append('jd_file', jdFile);
        return fd;
      };
      try {
        data = await postFormWithRouteFallback(
          [
            '/api/v1/relevance/generate-resume-file',
            '/api/v1/relevance/generate-resume-file/',
            '/api/v1/relevance/generate_resume_file',
            '/api/v1/relevance/generate_resume_file/',
            '/api/v1/relevance/resume-generator',
            '/api/relevance/generate-resume-file',
            '/api/relevance/generate_resume_file',
            '/api/relevance/resume-generator',
            '/relevance/generate-resume-file',
            '/relevance/generate_resume_file',
            '/relevance/resume-generator',
          ],
          buildResumeForm,
        );
      } catch (routeErr) {
        if (!isNotFoundError(routeErr)) throw routeErr;
        const buildAnalyzeForm = () => {
          const fd = new FormData();
          fd.append('title', requestContext.title);
          fd.append('analysis_type', 'resume_jd');
          fd.append('role', requestContext.role);
          fd.append('company', requestContext.company);
          fd.append('context_notes', requestContext.context_notes);
          fd.append('document_text', requestContext.cv_text);
          fd.append('reference_text', requestContext.jd_text);
          if (cvFile) fd.append('document_file', cvFile);
          if (jdFile) fd.append('reference_file', jdFile);
          return fd;
        };
        const relevanceData = await postFormWithRouteFallback(
          ['/api/v1/relevance/analyze-file', '/relevance/analyze-file'],
          buildAnalyzeForm,
        );
        data = adaptRelevanceToResumeResult(relevanceData, requestContext);
      }
    } else {
      try {
        data = await postJsonWithRouteFallback(
          [
            '/api/v1/relevance/generate-resume',
            '/api/v1/relevance/generate-resume/',
            '/api/v1/relevance/generate_resume',
            '/api/v1/relevance/generate_resume/',
            '/api/v1/relevance/resume-generator',
            '/api/relevance/generate-resume',
            '/api/relevance/generate_resume',
            '/api/relevance/resume-generator',
            '/relevance/generate-resume',
            '/relevance/generate_resume',
            '/relevance/resume-generator',
          ],
          requestContext,
        );
      } catch (routeErr) {
        if (!isNotFoundError(routeErr)) throw routeErr;
        const relevanceData = await postJsonWithRouteFallback(
          ['/api/v1/relevance/analyze-text', '/relevance/analyze-text'],
          {
            title: requestContext.title,
            analysis_type: 'resume_jd',
            role: requestContext.role,
            company: requestContext.company,
            document_text: requestContext.cv_text,
            reference_text: requestContext.jd_text,
            context_notes: requestContext.context_notes,
          },
        );
        data = adaptRelevanceToResumeResult(relevanceData, requestContext);
      }
    }
    showMessage(target, renderResumeGenerationResult(data));
    await refreshCurrentUser();
    loadHistory();
    loadDashboard();
  } catch (err) {
    const msg = (err.message || '').toLowerCase().includes('not found')
      ? 'Resume generation endpoint was not found. Verify backend/API base and retry.'
      : err.message;
    showMessage(target, escapeHtml(msg), true);
  }
});

document.getElementById('sentiment-text-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('sentiment-result');
  try {
    const data = await apiFetch('/api/v1/sentiment/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: document.getElementById('sentiment-text-title').value.trim(),
        content: document.getElementById('sentiment-text-content').value.trim(),
        emotion_metrics: document.getElementById('sentiment-metrics').value.trim(),
      }),
    });
    showMessage(target, renderSentimentResult(data));
    event.target.reset();
    loadDashboard();
  } catch (err) {
    showMessage(target, escapeHtml(err.message), true);
  }
});

document.getElementById('sentiment-file-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('sentiment-result');
  const file = document.getElementById('sentiment-file-input').files[0];
  if (!file) return showMessage(target, 'Please select a file.', true);
  const fd = new FormData();
  fd.append('title', document.getElementById('sentiment-file-title').value.trim());
  fd.append('emotion_metrics', document.getElementById('sentiment-file-metrics').value.trim());
  fd.append('file', file);
  try {
    const data = await apiFetch('/api/v1/sentiment/analyze-file', { method: 'POST', body: fd });
    showMessage(target, renderSentimentResult(data));
    event.target.reset();
    loadDashboard();
  } catch (err) {
    showMessage(target, escapeHtml(err.message), true);
  }
});

document.getElementById('learning-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('learning-result');

  const chapterFile = document.getElementById('learn-chapter-file').files[0];
  const notesFile = document.getElementById('learn-notes-file').files[0];

  try {
    let data;
    if (chapterFile || notesFile) {
      const fd = new FormData();
      fd.append('subject', document.getElementById('learn-subject').value);
      fd.append('chapter_text', document.getElementById('learn-chapter').value);
      fd.append('student_notes', document.getElementById('learn-notes').value);
      if (chapterFile) fd.append('chapter_file', chapterFile);
      if (notesFile) fd.append('student_notes_file', notesFile);
      data = await apiFetch('/api/v1/learning/story-file', { method: 'POST', body: fd });
    } else {
      data = await apiFetch('/api/v1/learning/story', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject: document.getElementById('learn-subject').value,
          chapter_text: document.getElementById('learn-chapter').value,
          student_notes: document.getElementById('learn-notes').value,
        }),
      });
    }
    showMessage(target, renderLearningResult(data));
    loadHistory();
    loadDashboard();
  } catch (err) {
    showMessage(target, escapeHtml(err.message), true);
  }
});

document.getElementById('learning-qa-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = document.getElementById('learning-qa-result');
  const body = {
    subject: document.getElementById('qa-subject').value,
    question_text: document.getElementById('qa-question').value.trim(),
    student_attempt: document.getElementById('qa-attempt').value.trim(),
    assignment_context: document.getElementById('qa-context').value.trim(),
    grade_level: document.getElementById('qa-grade').value.trim(),
  };
  try {
    let data;
    try {
      data = await postJsonWithRouteFallback(
        [
          '/api/v1/learning/question-answer',
          '/api/v1/learning/question-answer/',
          '/api/v1/learning/qa',
          '/api/v1/learning/qa/',
          '/api/v1/learning/question_answer',
          '/api/v1/learning/question_answer/',
          '/api/learning/question-answer',
          '/api/learning/qa',
          '/api/learning/question_answer',
          '/v1/learning/question-answer',
          '/v1/learning/qa',
          '/v1/learning/question_answer',
          '/learning/question-answer',
          '/learning/qa',
          '/learning/question_answer',
        ],
        body,
      );
    } catch (routeErr) {
      if (!isNotFoundError(routeErr)) throw routeErr;
      throw new Error('Student Q&A endpoint was not found. Restart backend, open /docs, and verify a POST route exists for learning question-answer.');
    }
    showMessage(target, renderLearningQAResult(data));
    loadHistory();
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
  } catch (_err) {}
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  showAuth();
});

window.addEventListener('popstate', () => {
  const v = routeToView[window.location.pathname];
  if (!currentUser) return showAuth();
  if (v) openView(v, { updateUrl: false });
  else openView('dashboard');
});

async function bootstrap() {
  try {
    const me = await apiFetch('/api/v1/auth/me');
    showApp(me);
    if (window.location.pathname === '/login') openView('dashboard');
  } catch (_err) {
    showAuth();
  }
}

bootstrap();
