import { useEffect, useRef, useState } from 'react';
import { ClipboardPaste, ImagePlus, LoaderCircle, ScanLine, SquareDashed, Target, X, Plus, Trash2, Check } from 'lucide-react';
import { api, fa, Job, percent, post, Rect, Reference, Scene } from './offline-api';
import PrecisionTools, { precisionDefaults, PrecisionOptions } from './PrecisionTools';
import ReferenceEditor from './ReferenceEditor';
import AssistantPanel from './AssistantPanel';
import type { Tool } from './ImageCanvas';

export type SimilarControl = { settings: unknown; ready: boolean; label: string };
type Props = { onSearchChange: (control: SimilarControl) => void;
  scene: Scene | null; reference: Reference | null; setReference: (item: Reference | null) => void;
  groupId: string; setGroupId: (id: string) => void; onTrainingChange: () => void; onMembersChange: (members: Reference[]) => void;
  revision: number; busy: boolean; available: boolean; aiAvailable: boolean; scope: Rect | null; clearScope: () => void;
  setTool: (tool: Tool) => void; start: (request: () => Promise<Job>) => void; job: Job | null;
};
type Group = { id: string; name: string; description?: string; excluded_description?: string; total?: number; enabled?: number; references?: Reference[] };
type Plan = { windows: number; completed: number };
export default function SimilarityPanel({ scene, reference, setReference, groupId, setGroupId, onTrainingChange, onMembersChange, revision, busy, available, aiAvailable, scope, clearScope, setTool, onSearchChange, start, job }: Props) {
  const [matchMode,setMatchMode] = useState<'instances'|'exact'>('instances');
  const [instruction,setInstruction] = useState(''),[excluded,setExcluded]=useState(''),[prepared,setPrepared]=useState('');
  const [quality,setQuality]=useState<'fast'|'balanced'|'detailed'>('balanced');
  const [deleted,setDeleted]=useState<{id:string;name:string}|null>(null);
  const [precision,setPrecision] = useState<PrecisionOptions>(precisionDefaults);
  const [editor,setEditor] = useState<Reference|null>(null);
  const [uploadLabel,setUploadLabel] = useState<0|1>(1),[partition,setPartition] = useState<'train'|'test'>('train');
  const [groups, setGroups] = useState<Group[]>([]), [members, setMembers] = useState<Reference[]>([]);
  const [name, setName] = useState(''), [newName, setNewName] = useState('');
  const [width, setWidth] = useState(128), [height, setHeight] = useState(128);
  const [threshold, setThreshold] = useState(.12), [broad, setBroad] = useState(false);
  const [rotate, setRotate] = useState(true), [intelligent, setIntelligent] = useState(true), [locked, setLocked] = useState(true);
  const [uploading, setUploading] = useState(false), [error, setError] = useState(''), [progress, setProgress] = useState('');
  const [plan, setPlan] = useState<Plan | null>(null);
  const input = useRef<HTMLInputElement>(null), loading = useRef(0);
  const load = async (preferred = groupId) => {
    const request = ++loading.current;
    const result = await api<{ groups: Group[]; active?: string }>('/workspace/reference-groups');
    const id = [preferred,result.active,result.groups[0]?.id].find(value=>result.groups.some(g=>g.id===value)) || '';
    const target = id ? await api<Group>(`/workspace/reference-groups/${id}`) : null;
    if (request !== loading.current) return;
    setGroups(result.groups); setMembers(target?.references ?? []); onMembersChange((target?.references ?? []).filter(r => r.enabled && r.label !== 0 && r.partition !== 'test')); setName(target?.name ?? ''); setInstruction(target?.description ?? ''); setExcluded(target?.excluded_description ?? ''); setPrepared('');
    if (id !== groupId) setGroupId(id);
    setReference(target?.references?.find(r => r.enabled && r.label !== 0 && r.partition !== 'test') ?? null);
  };
  useEffect(() => { load().catch(e => setError(e.message)); }, [revision, groupId]);
  useEffect(() => {
    if (reference) { setWidth(Math.min(8192, reference.width)); setHeight(Math.min(8192, reference.height)); }
  }, [reference?.id]);
  const upload = async (files: File[]) => {
    if (!files.length || busy || uploading) return;
    setUploading(true); setError('');
    let id = groupId; const errors: string[] = []; let added = false;
    try {
      for (let i=0; i<files.length; i++) {
        const file = files[i]; setProgress(`افزودن نمونه ${fa(i+1)} از ${fa(files.length)}`);
        if (file.size > 15*1024*1024) { errors.push(`${file.name}: حداکثر ۱۵ مگابایت برای هر نمونه`); continue; }
        try {
          const result = await api<Reference>(`/workspace/references?name=${encodeURIComponent(file.name)}${id ? `&group=${encodeURIComponent(id)}` : ''}&label=${uploadLabel}&partition=${partition}`, { method: 'POST', body: file });
          id = result.group!; added = true;
        } catch (e) { errors.push(`${file.name}: ${(e as Error).message}`); }
      }
      if (added) { setGroupId(id); await load(id); onTrainingChange(); }
      setError(errors.join('؛ '));
    } catch (e) { setError((e as Error).message); } finally { setUploading(false); setProgress(''); }
  };
  useEffect(() => {
    const paste = (event: ClipboardEvent) => {
      const files = Array.from(event.clipboardData?.files ?? []).filter(file => file.type.startsWith('image/'));
      if (files.length) { event.preventDefault(); upload(files); }
    };
    window.addEventListener('paste', paste);
    return () => window.removeEventListener('paste', paste);
  }, [busy, uploading, groupId, uploadLabel, partition]);
  const mutate = async (path: string, method: string, value?: unknown, changed = true) => {
    setUploading(true); setError('');
    try { await api(path, { method, ...(value === undefined ? {} : {headers: {'Content-Type':'application/json'},body:JSON.stringify(value)}) }); await load(); if (changed) onTrainingChange(); }
    catch (e) { await load().catch(() => {}); setError((e as Error).message); } finally { setUploading(false); }
  };
  const toggle = (id: string, enabled: boolean) => {
    setMembers(previous => previous.map(r => r.id === id ? {...r, enabled} : r));
    onTrainingChange();
    mutate(`/workspace/reference-groups/${groupId}/members/${id}`, 'PATCH', {enabled});
  };
  const create = async () => {
    if (!newName.trim()) return;
    setUploading(true); setError('');
    try { const group = await post<Group>('/workspace/reference-groups', {name:newName.trim()}); setGroupId(group.id); setNewName(''); await load(group.id); onTrainingChange(); }
    catch (e) { setError((e as Error).message); } finally { setUploading(false); }
  };
  const active = members.filter(r => r.enabled && r.label !== 0 && r.partition !== 'test');
  const signature = members.map(r => `${r.id}:${r.enabled}:${r.label}:${r.partition}`).join(',');
  const settings = { match_mode: matchMode, instruction, excluded_description: excluded, prepared: prepared||null, quality, scene: scene?.id, reference: reference?.id, group: groupId || null, width, height, threshold, scope, broad, rotate, intelligent, ...precision, mode: matchMode==='instances'?'complete':precision.mode };
  const precisionSignature=JSON.stringify(precision);
  useEffect(() => {
    setPlan(null);
    if (!scene || (matchMode==='exact' ? !reference || !active.length : !active.length&&!instruction.trim())) return;
    let stopped = false;
    const timer = setTimeout(() => post<Plan>('/workspace/similar/plan', settings).then(next => {
      if (!stopped) { setPlan(next); setError(''); }
    }).catch(e => { if (!stopped) setError(e.message); }), 250);
    return () => { stopped = true; clearTimeout(timer); };
  }, [scene?.id, reference?.id, groupId, signature, width, height, scope, broad, rotate, intelligent, precisionSignature, busy, uploading, matchMode, instruction, excluded, prepared, quality]);
  useEffect(() => {
    onSearchChange({ settings, ready: !uploading && !!scene && !!plan && (matchMode==='instances' ? aiAvailable&&!!(active.length||instruction.trim()) : !!reference&&!!active.length&&(!intelligent||available)), label: plan && plan.completed === plan.windows ? 'به‌روزرسانی نتایج مشابه' : plan?.completed ? 'ادامهٔ جست‌وجوی مشابه' : 'پیداکردن موارد مشابه' });
  }, [scene?.id, reference?.id, groupId, signature, width, height, threshold, scope, broad, rotate, intelligent, available, plan, uploading, precisionSignature, onSearchChange, matchMode, instruction, excluded, prepared, quality, aiAvailable]);
  const changeWidth = (value: number) => { setWidth(value); if (locked && reference) setHeight(Math.max(8, Math.min(8192, Math.round(value * reference.height / reference.width)))); };
  const changeHeight = (value: number) => { setHeight(value); if (locked && reference) setWidth(Math.max(8, Math.min(8192, Math.round(value * reference.width / reference.height)))); };
  return <section className="similarity-panel step-section" aria-label="آموزش هدف با یک یا چند نمونه">
    <div className="step-heading"><span className="step-number">۲</span><h2>جست‌وجو با هوش مصنوعی آفلاین</h2><Target size={17} /></div>
    <div className="mode-switch"><button aria-label="تشخیص هم‌نوع‌ها" className={matchMode==='instances'?'active':''} disabled={busy} onClick={()=>{setMatchMode('instances');setThreshold(.12);}}>تشخیص هم‌نوع‌ها</button><button aria-label="تطابق دقیق" className={matchMode==='exact'?'active':''} disabled={busy} onClick={()=>{setMatchMode('exact');setThreshold(.55);}}>تطابق دقیق</button></div>
    <p className="hint">عکس، متن، یا هر دو را بده. یک نمونه کافی است. برای آموزش بهتر، چند عکس واضح از همان هدف و زاویه‌های مختلف اضافه کن.</p>
    <div className="new-concept"><input aria-label="نام هدف تصویری جدید" value={newName} disabled={busy || uploading} onChange={e => setNewName(e.target.value)} placeholder="هدف جدید؛ مثلاً سوله یا زمین مشخص" onKeyDown={e => { if (e.key === 'Enter') create(); }} /><button title="ساخت هدف تصویری" disabled={busy || uploading || !newName.trim()} onClick={create}><Plus size={17} /></button></div>
    {groups.length > 0 && <label className="field-label">هدف تصویری<select aria-label="هدف تصویری" disabled={busy || uploading} value={groupId} onChange={e => { setGroupId(e.target.value); post(`/workspace/reference-groups/${e.target.value}/select`).catch(e => setError(e.message)); onTrainingChange(); }}>{groups.map(g => <option key={g.id} value={g.id}>{g.name} · {fa(g.enabled ?? members.filter(r => r.enabled && r.label !== 0 && r.partition !== 'test').length)} نمونهٔ فعال</option>)}</select></label>}
    {groupId && <div className="target-name"><input aria-label="نام هدف تصویری" value={name} disabled={busy || uploading} onChange={e => setName(e.target.value)} /><button title="ذخیره نام هدف" disabled={busy || uploading || !name.trim()} onClick={() => mutate(`/workspace/reference-groups/${groupId}`, 'PATCH', {name}, false)}><Check size={15} /></button><button title="حذف هدف تصویری" disabled={busy||uploading} onClick={async()=>{setUploading(true);try{await api(`/workspace/reference-groups/${groupId}`,{method:'DELETE'});setDeleted({id:groupId,name});setReference(null);setGroupId('');await load('');onTrainingChange();}catch(e){setError((e as Error).message);}finally{setUploading(false);}}}><Trash2 size={15}/></button></div>}
    {deleted&&<p className="hint">«{deleted.name}» حذف شد. <button disabled={busy||uploading} onClick={async()=>{try{await post(`/workspace/reference-groups/${deleted.id}/restore`);await load(deleted.id);setDeleted(null);onTrainingChange();}catch(e){setError((e as Error).message);}}}>بازگرداندن هدف</button></p>}
    {matchMode==='instances'&&<>
      <label className="field-label">خواستهٔ جست‌وجو<textarea aria-label="خواستهٔ جست‌وجو" value={instruction} maxLength={8000} disabled={busy||uploading} placeholder="آزادانه بنویس چه چیزی پیدا کنم؛ عکس نمونه اختیاری است." onChange={e=>{setInstruction(e.target.value);setPrepared('');}}/></label>
      <details className="assistant-details"><summary>توضیح موارد نامطلوب</summary><label className="field-label">مواردی که نمی‌خواهم<textarea aria-label="موارد نامطلوب" value={excluded} maxLength={8000} disabled={busy||uploading} onChange={e=>{setExcluded(e.target.value);setPrepared('');}}/></label></details>
      {groupId&&<div className="assistant-actions"><button disabled={busy||uploading} onClick={()=>mutate(`/workspace/reference-groups/${groupId}`,'PATCH',{description:instruction,excluded_description:excluded})}>ذخیرهٔ توضیحات هدف</button><button disabled={busy||uploading} onClick={()=>{setInstruction('');setExcluded('');setPrepared('');mutate(`/workspace/reference-groups/${groupId}`,'PATCH',{description:'',excluded_description:''});}}>پاک‌کردن توضیحات</button></div>}
      <details className="assistant-details"><summary>گفت‌وگو با هوش مصنوعی آفلاین</summary><AssistantPanel group={groupId} scene={scene} busy={busy||uploading} available={aiAvailable} job={job} start={start} usePlan={(message,key)=>{setInstruction(message);setExcluded('');setPrepared(key);}}/></details>
      <label className="field-label">دقت بررسی<select aria-label="دقت تشخیص هم‌نوع‌ها" value={quality} disabled={busy} onChange={e=>setQuality(e.target.value as typeof quality)}><option value="fast">سریع · یک مقیاس</option><option value="balanced">متعادل · دو مقیاس</option><option value="detailed">ریزتر · سه مقیاس</option></select></label>
      <p className="hint">در هر حالت تمام محدوده بررسی می‌شود. حالت ریزتر برای هدف‌های کوچک‌تر زمان بیشتری می‌خواهد. امتیاز مدل احتمال قطعی درست‌بودن نیست.</p>
    </>}
    <div className="training-upload-options"><label>نوع عکس<select aria-label="نوع عکس آموزشی" value={uploadLabel} disabled={busy||uploading} onChange={e=>setUploadLabel(Number(e.target.value) as 0|1)}><option value={1}>هدف</option><option value={0}>غیرهدف</option></select></label><label>کاربرد<select aria-label="کاربرد عکس آموزشی" value={partition} disabled={busy||uploading} onChange={e=>setPartition(e.target.value as 'train'|'test')}><option value="train">آموزش</option><option value="test">آزمون مستقل</option></select></label></div>
    <button className="reference-drop" disabled={busy || uploading} onClick={() => input.current?.click()}
      onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); upload(Array.from(event.dataTransfer.files)); }}>
      {uploading ? <LoaderCircle size={26} className="spin" /> : <ImagePlus size={26} />}<strong>{progress || (members.length ? 'افزودن عکس و زاویهٔ دیگر' : 'اسکرین‌شات یا عکس نمونه را بده')}</strong>
      <small>یک یا چند فایل، کشیدن و رهاکردن، یا Ctrl + V</small>
    </button>
    <input ref={input} type="file" multiple hidden accept="image/*,.ecw,.jp2,.heic,.heif" data-testid="reference-input" onChange={event => { upload(Array.from(event.target.files ?? [])); event.target.value = ''; }} />
    <button className="reference-draw" disabled={!scene || busy || uploading} onClick={() => setTool('reference')}><Target size={16} />انتخاب نمونه از روی تصویر</button><button className="reference-draw negative" disabled={!scene || busy || uploading || !groupId} onClick={() => setTool('negative-reference')}>نمونهٔ غیرهدف از روی تصویر</button>
    <p className="hint">هدف در هر عکس واضح و نزدیک کادر باشد؛ نماهای مختلفِ همان هدف را اضافه کن.</p>
    <div className="teaching-summary"><b>{fa(active.length)} نمونهٔ فعال</b><span>شروع با یک نمونه · زاویه‌های بیشتر اختیاری</span></div>
    {members.length > 0 && <div className="teaching-gallery" aria-label="نمونه‌های آموزشی این هدف">{members.map((r,i) => <article key={r.id} className={`${r.enabled ? 'enabled' : 'disabled'} ${r.label===0 ? 'negative' : ''}`}><img src={r.image_url} alt={`نمونهٔ آموزشی ${r.name}`} loading="lazy" /><div><label><input type="checkbox" checked={!!r.enabled} disabled={busy || uploading} aria-label={`استفاده از نمونه ${r.name}`} onChange={e => toggle(r.id, e.target.checked)} /><span>{r.name}</span></label><small>{r.label===0?'غیرهدف':reference?.id === r.id?'نمونهٔ پایه':`نمای ${fa(i+1)}`}{r.partition==='test'?' · آزمون':''}</small><button title={`جداکردن هدف در ${r.name}`} disabled={busy||uploading} onClick={()=>setEditor(r)}><Target size={13}/></button><select aria-label={`برچسب ${r.name}`} value={r.label??1} disabled={busy||uploading} onChange={e=>mutate(`/workspace/reference-groups/${groupId}/members/${r.id}`,'PATCH',{label:Number(e.target.value)})}><option value={1}>هدف</option><option value={0}>غیرهدف</option></select><select aria-label={`کاربرد ${r.name}`} value={r.partition??'train'} disabled={busy||uploading} onChange={e=>mutate(`/workspace/reference-groups/${groupId}/members/${r.id}`,'PATCH',{partition:e.target.value})}><option value="train">آموزش</option><option value="test">آزمون</option></select><button title={`حذف نمونه ${r.name} از هدف`} disabled={busy || uploading} onClick={() => mutate(`/workspace/reference-groups/${groupId}/members/${r.id}`, 'DELETE')}><Trash2 size={13} /></button></div></article>)}</div>}
    {reference && <div className="reference-preview"><img src={reference.image_url} alt="نمونهٔ مرجع جست‌وجو" /><div><b>{reference.name}</b><small>{fa(reference.width)} × {fa(reference.height)} پیکسل</small><span><ClipboardPaste size={12} />جست‌وجو با همهٔ نمونه‌های فعال همین هدف</span></div></div>}
    <div className="shape-teaching-note"><Target size={16} /><span>یادگیری شکل و بافت<small>رنگ از ورودی مقایسه حذف می‌شود؛ هم‌رنگ‌بودن به‌تنهایی امتیاز تطابق نمی‌دهد.</small></span></div>
    {matchMode==='exact'&&<><label className="field-label">اندازهٔ تقریبی نمونه روی تصویر اصلی</label>
    <div className="reference-size" dir="ltr"><label>عرض<input aria-label="عرض نمونه روی تصویر" type="number" min={8} max={8192} value={width} disabled={busy} onChange={event => changeWidth(Number(event.target.value))} /></label><span>×</span><label>ارتفاع<input aria-label="ارتفاع نمونه روی تصویر" type="number" min={8} max={8192} value={height} disabled={busy} onChange={event => changeHeight(Number(event.target.value))} /></label></div>
    <label className="check-field"><input type="checkbox" checked={locked} onChange={event => setLocked(event.target.checked)} />حفظ نسبت عرض و ارتفاع</label>
    <p className="hint">اندازهٔ هدف روی تصویر اصلی را تنظیم کن. عکس‌های آموزشی با حفظ نسبت ابعاد خود مقیاس می‌شوند؛ یکسان‌بودن اندازهٔ فایل‌ها لازم نیست.</p>
    </>}
    <div className="similar-scope"><span>{scope ? 'داخل محدودهٔ انتخابی' : 'کل تصویر'}</span><button disabled={!scene || busy} onClick={() => setTool('scope')} title="انتخاب محدوده"><SquareDashed size={16} /></button>{scope && <button disabled={busy} onClick={clearScope} title="جست‌وجو در کل تصویر"><X size={14} /></button>}</div>
    <label className="field-label">حداقل امتیاز شباهت <b>{percent(threshold)}</b><input aria-label="حداقل شباهت" type="range" min={matchMode==='instances'?.01:.4} max={.99} step={.01} value={threshold} disabled={busy} onChange={event => setThreshold(Number(event.target.value))} /></label>
    {matchMode==='exact'&&<><p className="hint">۵۵٪ برای دیدن موارد مشابه و قابل بررسی؛ ۸۵٪ برای تطابق‌های نزدیک‌تر. کم‌کردن آستانه ممکن است نتیجهٔ نامرتبط هم اضافه کند.</p>
    <label className="check-field"><input type="checkbox" checked={rotate} disabled={busy} onChange={event => setRotate(event.target.checked)} />جست‌وجو در جهت‌های مختلف، هر ۴۵ درجه</label>
    <label className="check-field"><input type="checkbox" checked={broad} disabled={busy} onChange={event => setBroad(event.target.checked)} />تغییر اندازهٔ گسترده‌تر: نصف تا دو برابر</label>
    <label className="check-field"><input type="checkbox" checked={intelligent} disabled={busy} onChange={event => setIntelligent(event.target.checked)} />جست‌وجوی عمیق ویژگی‌ها با DINOv2 و بازبینی CLIP</label>
    <p className="hint">جست‌وجوی عمیق، شکل و بافت مشابه را مستقل از تطابق پیکسلی بررسی می‌کند؛ روی CPU زمان بیشتری می‌خواهد.</p>
    {intelligent && !available && <p className="reference-error">مدل‌های جست‌وجوی عمیق در دسترس نیستند؛ پوشهٔ کامل نسخهٔ جدید برنامه را استخراج کن.</p>}
    </>}
    {plan && <div className="similar-plan"><ScanLine size={16} /><span>{fa(plan.windows)} بخش برای بررسی{plan.completed > 0 && <> · {fa(plan.completed)} بخش آماده</>}<small>پیشرفت ذخیره می‌شود. تصویر بزرگ زمان بیشتری می‌خواهد؛ {matchMode==='instances'||precision.mode==='complete'?'همهٔ بخش‌ها بررسی می‌شوند.':'حالت سریع فقط ناحیه‌های برگزیده را بررسی می‌کند.'}</small></span></div>}
    {matchMode==='exact'&&<PrecisionTools group={groupId} scene={scene} busy={busy||uploading} revision={revision} options={precision} onChange={setPrecision} start={start}/>}
    {editor && <ReferenceEditor reference={editor} group={groupId} busy={busy} job={job} start={start} close={()=>setEditor(null)} updated={async()=>{await load();onTrainingChange();}}/>}
    {error && <p className="reference-error" role="alert">{error}</p>}
    <p className="hint">همهٔ نتایج پذیرفته‌شده ذخیره می‌شوند. شباهت بصری تضمین تشخیص بی‌خطا نیست.</p>
  </section>;
}
