/** Minimal fetch wrapper for the FastAPI backend (same-origin /api via Vite proxy). */
export async function getHealth() {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function getRules() {
  const r = await fetch("/api/rules");
  if (!r.ok) throw new Error(`rules ${r.status}`);
  return r.json();
}

export async function postAdvise(file, target, preferRepeated) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("target", target);
  fd.append("prefer_repeated", preferRepeated ? "true" : "false");
  const r = await fetch("/api/advise", { method: "POST", body: fd });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `advise failed (${r.status})`);
  }
  return r.json();
}
