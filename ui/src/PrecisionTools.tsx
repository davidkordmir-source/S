import { useEffect, useState } from 'react';
import { BrainCircuit, Database, Gauge, ScanLine } from 'lucide-react';
import { api, fa, Job, post, Scene } from './offline-api';

export type PrecisionOptions = { aerial: boolean; geometry: boolean; strict_geometry: boolean; fine_pose: boolean; adapt: boolean; mode: 'complete' | 'fast'; index_block: 256 | 512 | 1024; fast_regions: number };
export const precisionDefaults: PrecisionOptions = { aerial: true, geometry: true, strict_geometry: false, fine_pose: true, adapt: true, mode: 'complete', index_block: 512, fast_regions: 1000 };
type Index = { windows: number; completed: number; estimated_bytes: number; settings: { block: number } };
type Learning = { positive: number; negative: number; test: number; ready: boolean; report: null | { evaluation_kind: string; evaluation_references: number; evaluation_note?: string; metrics: null | {precision:number;recall:number;f1:number;true_positive:number;false_positive:number;false_negative:number;true_negative:number} } };
type Props = { group: string; scene: Scene | null; busy: boolean; revision: number; options: PrecisionOptions; onChange: (options: PrecisionOptions) => void; start: (request: () => Promise<Job>) => void };
export default function PrecisionTools({group,scene,busy,revision,options,onChange,start}: Props) {
  const [indexes,setIndexes] = useState<Index[]>([]), [learning,setLearning] = useState<Learning | null>(null), [error,setError] = useState('');
  const [budget,setBudget] = useState(20);
  useEffect(() => {
    if (!scene) {setIndexes([]);return;}
    let stopped=false;
    api<{indexes:Index[]}>(`/workspace/scenes/${scene.id}/indexes`).then(r=>{if(!stopped)setIndexes(r.indexes);}).catch(e=>{if(!stopped)setError(e.message);});
    return ()=>{stopped=true;};
  },[scene?.id,busy]);
  useEffect(() => {
    if (!group) {setLearning(null);return;}
    let stopped=false;
    api<Learning>(`/workspace/reference-groups/${group}/learning`).then(r=>{if(!stopped)setLearning(r);}).catch(e=>{if(!stopped)setError(e.message);});
    return ()=>{stopped=true;};
  },[group,busy,revision]);
  const change = <K extends keyof PrecisionOptions>(key: K,value: PrecisionOptions[K]) => onChange({...options,[key]:value,...(key==='geometry'&&!value?{strict_geometry:false}:{})});
  const index=indexes.find(i=>i.settings.block===options.index_block);
  return <details className="precision-tools"><summary><Gauge size={16}/>دقت، آموزش اختصاصی و کارایی</summary>
    <div className="precision-section"><b><ScanLine size={15}/>بررسی دقیق‌تر نتیجه‌ها</b>
      <label className="check-field"><input type="checkbox" checked={options.aerial} disabled={busy} onChange={e=>change('aerial',e.target.checked)}/>مدل مخصوص تصاویر هوایی RemoteCLIP</label>
      <label className="check-field"><input type="checkbox" checked={options.geometry} disabled={busy} onChange={e=>change('geometry',e.target.checked)}/>تأیید شکل، تناسب و چیدمان با بررسی هندسی</label>
      <label className="check-field"><input type="checkbox" checked={options.fine_pose} disabled={busy} onChange={e=>change('fine_pose',e.target.checked)}/>بررسی چرخش پیوسته، اندازه و پرسپکتیو</label>
      <label className="check-field"><input type="checkbox" checked={options.strict_geometry} disabled={busy || !options.geometry} onChange={e=>change('strict_geometry',e.target.checked)}/>فقط نتیجه‌های تأییدشدهٔ هندسی</label>
      <p className="hint">حالت سخت‌گیرانه می‌تواند هدف‌های کم‌جزئیات یا کم‌وضوح را کنار بگذارد.</p>
    </div>
    <div className="precision-section"><b><BrainCircuit size={15}/>یادگیری از نمونه‌ها و اشتباهات</b>
      <label className="check-field"><input type="checkbox" checked={options.adapt} disabled={busy} onChange={e=>change('adapt',e.target.checked)}/>استفاده از آموزش اختصاصی همین هدف</label>
      <p className="hint">دکمه‌های «درست» و «غلط» زیر نتایج، همان هدف را آموزش می‌دهند. برای شروع جست‌وجو همچنان یک عکس کافی است.</p>
      {learning && <><div className="learning-counts"><span>{fa(learning.positive)} هدف</span><span>{fa(learning.negative)} غیرهدف</span><span>{fa(learning.test)} آزمون</span></div><button className="reference-draw" disabled={busy || !learning.positive || !learning.negative} onClick={()=>start(()=>post<Job>(`/workspace/reference-groups/${group}/train`))}>{learning.ready?'آموزش اختصاصی آماده است · آموزش مجدد':'آموزش اختصاصی و ارزیابی'}</button>
      {learning.report?.metrics && learning.ready ? <div className="evaluation-report"><b>{learning.report.evaluation_kind==='independent'?'آزمون کنارگذاشته‌شده':'اعتبارسنجی گروهی نمونه‌ها'}</b><span>دقت نتیجه‌های مثبت {fa(learning.report.metrics.precision*100,1)}٪</span><span>پوشش هدف‌ها {fa(learning.report.metrics.recall*100,1)}٪</span><span>F1 {fa(learning.report.metrics.f1*100,1)}٪ · {fa(learning.report.evaluation_references)} نمونه</span><small>هشدار اشتباه: {fa(learning.report.metrics.false_positive)} · هدف ازدست‌رفته: {fa(learning.report.metrics.false_negative)}</small></div> : <p className="hint">برای سنجش کیفیت، نمونه‌های هدف و غیرهدفِ جداگانه را «آزمون» علامت بزن. بدون دادهٔ کافی، عدد دقت نمایش داده نمی‌شود.</p>}{learning.ready && learning.report?.evaluation_note && <p className="hint">{learning.report.evaluation_note}</p>}</>}
    </div>
    <div className="precision-section"><b><Database size={15}/>نمایهٔ ذخیره‌شدهٔ تصویر اصلی</b>
      <label className="field-label">روش جست‌وجو<select aria-label="حالت جست‌وجوی صحنه" value={options.mode} disabled={busy} onChange={e=>change('mode',e.target.value as PrecisionOptions['mode'])}><option value="complete">کامل · همهٔ بخش‌ها</option><option value="fast">سریع · ناحیه‌های برگزیدهٔ نمایه</option></select></label>
      {options.mode==='fast' && <p className="precision-warning">حالت سریع تقریبی است و ممکن است بعضی موارد را پیدا نکند. نمایه باید کامل شده باشد.</p>}
      <label className="field-label">جزئیات نمایه<select aria-label="جزئیات نمایه" value={options.index_block} disabled={busy} onChange={e=>change('index_block',Number(e.target.value) as PrecisionOptions['index_block'])}><option value={256}>جزئیات بیشتر · فضای بیشتر</option><option value={512}>متعادل</option><option value={1024}>کم‌حجم‌تر · بررسی کلی‌تر</option></select></label>
      {index && <p className="hint">{fa(index.completed)} از {fa(index.windows)} بخش آماده · فضای برآوردی {fa(index.estimated_bytes/1024**3,2)} گیگابایت</p>}
      <label className="field-label">بودجهٔ دیسک نمایه، گیگابایت<input aria-label="بودجه دیسک نمایه" type="number" min={1} max={4096} value={budget} disabled={busy} onChange={e=>setBudget(Number(e.target.value))}/></label>
      <button className="reference-draw" disabled={busy || !scene || !index || index.completed===index.windows} onClick={()=>start(()=>post<Job>('/workspace/scene-index',{scene:scene!.id,block:options.index_block,budget_gb:budget}))}>{index?.completed?'ادامهٔ نمایه‌سازی':'ساخت نمایهٔ آفلاین'}</button>
      {options.mode==='fast' && <label className="field-label">تعداد ناحیه‌های برگزیده<input aria-label="تعداد ناحیه سریع" type="number" min={10} max={100000} value={options.fast_regions} disabled={busy} onChange={e=>change('fast_regions',Number(e.target.value))}/></label>}
      <p className="hint">ویژگی‌های صحنه روی دیسک ذخیره می‌شوند و در جست‌وجوهای بعدی استفاده می‌شوند. بررسی کامل به نمایه‌سازی اولیه نیاز ندارد.</p>
    </div>{error && <p className="reference-error">{error}</p>}
  </details>;
}
