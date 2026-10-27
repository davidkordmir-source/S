import { useEffect, useRef, useState } from 'react';
import { ClipboardPaste, ImagePlus, LoaderCircle, ScanLine, SquareDashed, Target, X } from 'lucide-react';
import { api, fa, Job, percent, post, Rect, Reference, Scene } from './offline-api';
import type { Tool } from './ImageCanvas';

export type SimilarControl = { settings: unknown; ready: boolean; label: string };
type Props = { onSearchChange: (control: SimilarControl) => void;
  scene: Scene | null; reference: Reference | null; setReference: (item: Reference) => void;
  revision: number; busy: boolean; available: boolean; scope: Rect | null; clearScope: () => void;
  setTool: (tool: Tool) => void; start: (request: () => Promise<Job>) => void;
};
type Plan = { windows: number; completed: number };
export default function SimilarityPanel({ scene, reference, setReference, revision, busy, available, scope, clearScope, setTool, onSearchChange }: Props) {
  const [references, setReferences] = useState<Reference[]>([]);
  const [width, setWidth] = useState(128), [height, setHeight] = useState(128);
  const [threshold, setThreshold] = useState(.55), [broad, setBroad] = useState(false);
  const [rotate, setRotate] = useState(true), [intelligent, setIntelligent] = useState(true), [locked, setLocked] = useState(true);
  const [uploading, setUploading] = useState(false), [error, setError] = useState('');
  const [plan, setPlan] = useState<Plan | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const load = async () => {
    const result = await api<{ references: Reference[] }>('/workspace/references');
    setReferences(result.references);
    if (!reference && result.references.length) setReference(result.references[0]);
  };
  useEffect(() => { load().catch(e => setError(e.message)); }, [revision]);
  useEffect(() => {
    if (reference) { setWidth(Math.min(8192, reference.width)); setHeight(Math.min(8192, reference.height)); }
  }, [reference?.id]);
  const upload = async (file?: File) => {
    if (!file || busy || uploading) return;
    if (file.size > 15 * 1024 * 1024) { setError('فایل نمونه حداکثر ۱۵ مگابایت است؛ تصویر اصلی را از «انتخاب تصویر» باز کن.'); return; }
    setUploading(true); setError('');
    try {
      const result = await api<Reference>(`/workspace/references?name=${encodeURIComponent(file.name)}`, { method: 'POST', body: file });
      setReference(result);
      setReferences(previous => [result, ...previous]);
    } catch (e) { setError((e as Error).message); } finally { setUploading(false); }
  };
  useEffect(() => {
    const paste = (event: ClipboardEvent) => {
      const file = Array.from(event.clipboardData?.files ?? []).find(file => file.type.startsWith('image/'));
      if (file) { event.preventDefault(); upload(file); }
    };
    window.addEventListener('paste', paste);
    return () => window.removeEventListener('paste', paste);
  }, [busy, uploading]);
  const settings = { scene: scene?.id, reference: reference?.id, width, height, threshold, scope, broad, rotate, intelligent };
  useEffect(() => {
    setPlan(null);
    if (!scene || !reference) return;
    let stopped = false;
    const timer = setTimeout(() => post<Plan>('/workspace/similar/plan', settings).then(next => {
      if (!stopped) { setPlan(next); setError(''); }
    }).catch(e => { if (!stopped) setError(e.message); }), 250);
    return () => { stopped = true; clearTimeout(timer); };
  }, [scene?.id, reference?.id, width, height, scope, broad, rotate, intelligent, busy]);
  useEffect(() => {
    onSearchChange({ settings, ready: !uploading && !!scene && !!reference && !!plan && (!intelligent || available), label: plan && plan.completed === plan.windows ? 'به‌روزرسانی نتایج مشابه' : plan?.completed ? 'ادامهٔ جست‌وجوی مشابه' : 'پیداکردن موارد مشابه' });
  }, [scene?.id, reference?.id, width, height, threshold, scope, broad, rotate, intelligent, available, plan, uploading, onSearchChange]);
  const changeWidth = (value: number) => { setWidth(value); if (locked && reference) setHeight(Math.max(8, Math.min(8192, Math.round(value * reference.height / reference.width)))); };
  const changeHeight = (value: number) => { setHeight(value); if (locked && reference) setWidth(Math.max(8, Math.min(8192, Math.round(value * reference.width / reference.height)))); };
  return <section className="similarity-panel step-section" aria-label="پیداکردن مشابه با یک نمونه">
    <div className="step-heading"><span className="step-number">۲</span><h2>این را برایم پیدا کن</h2><Target size={17} /></div>
    <p className="hint">یک قسمت از زمین را نشان بده؛ موارد شبیه آن روی تصویر علامت می‌خورند.</p>
    <button className="reference-drop" disabled={busy || uploading} onClick={() => input.current?.click()}
      onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); upload(event.dataTransfer.files[0]); }}>
      {uploading ? <LoaderCircle size={26} className="spin" /> : <ImagePlus size={26} />}<strong>اسکرین‌شات یا عکس نمونه را بده</strong>
      <small>انتخاب فایل، کشیدن و رهاکردن، یا Ctrl + V</small>
    </button>
    <input ref={input} type="file" hidden accept="image/*,.ecw,.jp2,.heic,.heif" data-testid="reference-input" onChange={event => { upload(event.target.files?.[0]); event.target.value = ''; }} />
    <button className="reference-draw" disabled={!scene || busy} onClick={() => setTool('reference')}><Target size={16} />انتخاب نمونه از روی تصویر</button>
    {references.length > 0 && <label className="field-label">نمونهٔ مرجع<select aria-label="نمونهٔ مرجع" disabled={busy} value={reference?.id ?? ''} onChange={event => { const item = references.find(item => item.id === event.target.value); if (item) setReference(item); }}>{references.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
    {reference && <div className="reference-preview"><img src={reference.image_url} alt="نمونهٔ مرجع جست‌وجو" /><div><b>{reference.name}</b><small>{fa(reference.width)} × {fa(reference.height)} پیکسل</small><span><ClipboardPaste size={12} />یک نمونه برای شروع کافی است</span></div></div>}
    <label className="field-label">اندازهٔ تقریبی نمونه روی تصویر اصلی</label>
    <div className="reference-size" dir="ltr"><label>عرض<input aria-label="عرض نمونه روی تصویر" type="number" min={8} max={8192} value={width} disabled={busy} onChange={event => changeWidth(Number(event.target.value))} /></label><span>×</span><label>ارتفاع<input aria-label="ارتفاع نمونه روی تصویر" type="number" min={8} max={8192} value={height} disabled={busy} onChange={event => changeHeight(Number(event.target.value))} /></label></div>
    <label className="check-field"><input type="checkbox" checked={locked} onChange={event => setLocked(event.target.checked)} />حفظ نسبت عرض و ارتفاع</label>
    <p className="hint">اندازه از نمونه پیشنهاد می‌شود. اگر اسکرین‌شات با زوم دیگری گرفته شده، اندازهٔ واقعی آن را روی تصویر تنظیم کن.</p>
    <div className="similar-scope"><span>{scope ? 'داخل محدودهٔ انتخابی' : 'کل تصویر'}</span><button disabled={!scene || busy} onClick={() => setTool('scope')} title="انتخاب محدوده"><SquareDashed size={16} /></button>{scope && <button disabled={busy} onClick={clearScope} title="جست‌وجو در کل تصویر"><X size={14} /></button>}</div>
    <label className="field-label">حداقل امتیاز شباهت <b>{percent(threshold)}</b><input aria-label="حداقل شباهت" type="range" min={.4} max={.99} step={.01} value={threshold} disabled={busy} onChange={event => setThreshold(Number(event.target.value))} /></label>
    <p className="hint">۵۵٪ برای دیدن موارد مشابه و قابل بررسی؛ ۸۵٪ برای تطابق‌های نزدیک‌تر. کم‌کردن آستانه ممکن است نتیجهٔ نامرتبط هم اضافه کند.</p>
    <label className="check-field"><input type="checkbox" checked={rotate} disabled={busy} onChange={event => setRotate(event.target.checked)} />جست‌وجو در جهت‌های مختلف، هر ۴۵ درجه</label>
    <label className="check-field"><input type="checkbox" checked={broad} disabled={busy} onChange={event => setBroad(event.target.checked)} />تغییر اندازهٔ گسترده‌تر: نصف تا دو برابر</label>
    <label className="check-field"><input type="checkbox" checked={intelligent} disabled={busy} onChange={event => setIntelligent(event.target.checked)} />جست‌وجوی عمیق ویژگی‌ها با DINOv2 و بازبینی CLIP</label>
    <p className="hint">جست‌وجوی عمیق، شکل و بافت مشابه را مستقل از تطابق پیکسلی بررسی می‌کند؛ روی CPU زمان بیشتری می‌خواهد.</p>
    {intelligent && !available && <p className="reference-error">مدل‌های جست‌وجوی عمیق در دسترس نیستند؛ پوشهٔ کامل نسخهٔ جدید برنامه را استخراج کن.</p>}
    {plan && <div className="similar-plan"><ScanLine size={16} /><span>{fa(plan.windows)} بخش برای بررسی{plan.completed > 0 && <> · {fa(plan.completed)} بخش آماده</>}<small>پیشرفت ذخیره می‌شود. تصویر بزرگ زمان بیشتری می‌خواهد؛ هیچ بخشی خودکار حذف نمی‌شود.</small></span></div>}
    {error && <p className="reference-error" role="alert">{error}</p>}
    <p className="hint">همهٔ نتایج پذیرفته‌شده ذخیره می‌شوند. شباهت بصری تضمین تشخیص بی‌خطا نیست.</p>
  </section>;
}
