export type Building = { id: string; title: string; lat: number; lon: number; source: string; image_url: string; score?: number; geometry?: GeoJSON.Geometry | null };
export type State = { version: string; library: { total: number; indexed: number }; models: { image_loaded: boolean; text_loaded: boolean; device: string; image_model: string; text_model: string }; center: { lat: number; lon: number }; data_directory: string; provider: string };
export type Job = { id: string; kind: string; status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'; message: string; progress: number; error?: string; result?: { buildings?: Building[]; message?: string; provider_capped?: boolean; center?: { lat: number; lon: number } } };

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...options, credentials: 'same-origin' });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : `درخواست کامل نشد (${response.status}).`);
  }
  return response.json();
}

export const post = <T,>(path: string, body?: unknown) => api<T>(path, { method: 'POST', ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
export const fa = (value: number, digits = 0) => new Intl.NumberFormat('fa-IR', { maximumFractionDigits: digits }).format(value);
