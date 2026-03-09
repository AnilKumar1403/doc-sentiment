const DEFAULT_BACKEND = "https://sentiment-backend-latest-r5du.onrender.com";

function backendBase() {
  const raw =
    process.env.AQUALEARNING_BACKEND_URL ||
    process.env.BACKEND_API_BASE ||
    DEFAULT_BACKEND;
  return String(raw || DEFAULT_BACKEND).replace(/\/+$/, "");
}

function buildTargetUrl(req) {
  const path = String(req.query.path || "").replace(/^\/+/, "");
  const incoming = new URL(req.url || "/", "http://localhost");
  const params = new URLSearchParams(incoming.search);
  params.delete("path");
  const query = params.toString();
  return `${backendBase()}/api/${path}${query ? `?${query}` : ""}`;
}

function filteredRequestHeaders(req) {
  const headers = { ...req.headers };
  delete headers.host;
  delete headers.connection;
  delete headers["content-length"];
  delete headers["x-forwarded-for"];
  delete headers["x-forwarded-host"];
  delete headers["x-forwarded-port"];
  delete headers["x-forwarded-proto"];
  return headers;
}

async function readRawBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

function copyResponseHeaders(upstream, res, bodyLength) {
  upstream.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (k === "transfer-encoding" || k === "content-length") return;
    res.setHeader(key, value);
  });
  if (typeof bodyLength === "number") {
    res.setHeader("content-length", String(bodyLength));
  }
}

module.exports = async function handler(req, res) {
  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  try {
    const target = buildTargetUrl(req);
    const method = String(req.method || "GET").toUpperCase();
    const body = method === "GET" || method === "HEAD" ? undefined : await readRawBody(req);

    const upstream = await fetch(target, {
      method,
      headers: filteredRequestHeaders(req),
      body,
      redirect: "manual",
    });

    const payload = Buffer.from(await upstream.arrayBuffer());
    copyResponseHeaders(upstream, res, payload.length);
    res.status(upstream.status).send(payload);
  } catch (err) {
    res.status(502).json({
      detail: "Proxy upstream error",
      error: String(err?.message || err || "unknown"),
    });
  }
};

module.exports.config = {
  api: {
    bodyParser: false,
  },
};
