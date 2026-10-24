export type Rect = { x: number; y: number; width: number; height: number };
export type Scene = { id: string; name: string; width: number; height: number; preview_url: string; storage?: 'linked' | 'managed'; approximate_preview?: boolean };
export type Concept = { id: string; name: string; positive: number; negative: number; revision: number; trained_revision: number | null; ready: boolean; window_width: number; window_height: number };
export type Example = { id: string; scene: string | null; bbox: Rect | null; label: 0 | 1; image_url: string; width: number; height: number };
export type Detection = { index: number; bbox: Rect; score: number };
export type Scan = { run: string; scene: string; concept: string | null; revision: number | null; detections: Detection[]; message: string; settings: { windows: number; scope: Rect; width: number; height: number } };
export type ImageFormats = { accept: string; groups: {name: string; extensions: string[]}[]; ecw_available: boolean };
export type State = { desktop_picker: boolean; image_formats: ImageFormats; version: string; offline: true; scenes: Scene[]; concepts: Concept[]; data_directory: string; models: { image_available: boolean; text_available: boolean; image_loaded: boolean; text_loaded: boolean } };
export type Job = { id: string; kind: string; status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'; progress: number; message: string; error?: string; result?: Omit<Partial<Scan>, 'scene' | 'concept'> & { scene?: Scene | string; concept?: string | null } };
export const fa = (value: number, digits = 0) => new Intl.NumberFormat('fa-IR', { maximumFractionDigits: digits }).format(value);
export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { credentials: 'same-origin', ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : `درخواست کامل نشد (${response.status}).`);
  }
  return response.json();
}
export const post = <T,>(path: string, value?: unknown) => api<T>(path, { method: 'POST', ...(value === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) }) });
export async function download(path: string, filename: string) {
  const response = await fetch(`/api${path}`, { credentials: 'same-origin' });
  if (!response.ok) { const body = await response.json().catch(() => null); throw new Error(body?.detail || 'خروجی ساخته نشد.'); }
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
