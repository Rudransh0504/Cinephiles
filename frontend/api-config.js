// api-config.js — shared across all pages
const API_BASE = "http://127.0.0.1:8000/api";

function getToken() {
  return localStorage.getItem("cinephiles_token");
}

function setToken(token) {
  localStorage.setItem("cinephiles_token", token);
}

function clearToken() {
  localStorage.removeItem("cinephiles_token");
}

function isLoggedIn() {
  return !!getToken();
}

function requireAuth() {
  if (!isLoggedIn()) {
    window.location.href = "auth.html";
  }
}

function logout() {
  clearToken();
  window.location.href = "auth.html";
}

// Wrapper around fetch that attaches the Bearer token automatically
async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (res.status === 401) {
    // Token expired or invalid — force re-login
    clearToken();
    window.location.href = "auth.html";
    return null;
  }

  if (res.status === 204) {
    return null; // No content (e.g. successful DELETE)
  }

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    throw new Error(data?.detail || "Request failed");
  }

  return data;
}