import { useEffect, useState } from 'react';
import { Cpu, Gauge } from 'lucide-react';
import { fa, post, State } from './offline-api';

type Props={state:State|null;busy:boolean;updated:()=>Promise<unknown>};
export default function HardwarePanel({state,busy,updated}:Props){
  const c=state?.models.compute;const [error,setError]=useState(''),[saving,setSaving]=useState(false),[renderer,setRenderer]=useState('');
  useEffect(()=>{const canvas=document.createElement('canvas');const gl=canvas.getContext('webgl2')??canvas.getContext('webgl');if(!gl){setRenderer('رندر نرم‌افزاری یا نامشخص');return;}const info=gl.getExtension('WEBGL_debug_renderer_info');const name=String(gl.getParameter(info?info.UNMASKED_RENDERER_WEBGL:gl.RENDERER));setRenderer(name);gl.getExtension('WEBGL_lose_context')?.loseContext();},[]);
  const change=async(device:string,performance:string)=>{setSaving(true);setError('');try{await post('/compute',{device,performance});await updated();}catch(e){setError((e as Error).message);}finally{setSaving(false);}};
  return <div className="hardware-panel"><b><Cpu size={17}/>پردازش روی سخت‌افزار همین کامپیوتر</b><p className="hint">حالت خودکار از GPU سازگار برای ویژگی‌های تصویری استفاده می‌کند. رندر رابط از شتاب گرافیکی سیستم بهره می‌برد؛ همهٔ داده‌ها محلی می‌مانند.</p>
    <label className="field-label">پردازش هوش مصنوعی<select aria-label="پردازش سخت افزاری" value={c?.preference??'auto'} disabled={busy||saving||!c} onChange={e=>change(e.target.value,c?.performance??'maximum')}><option value="auto">خودکار · GPU در صورت پشتیبانی</option><option value="cpu">CPU</option><option value="gpu" disabled={!c?.gpu_available}>GPU</option></select></label>
    <label className="field-label">کارایی<select aria-label="حالت کارایی سیستم" value={c?.performance??'maximum'} disabled={busy||saving||!c} onChange={e=>change(c?.preference??'auto',e.target.value)}><option value="maximum">حداکثر کارایی · منابع بیشتر</option><option value="balanced">متعادل</option></select></label>
    <div className="hardware-status"><span><Gauge size={14}/>پردازش فعال: {c?.device??state?.models.device??'CPU'}</span>{c && <span>{fa(c.cpu_threads)} رشتهٔ CPU برای محاسبه</span>}{c?.adapters.map(a=><small key={a.id}>{a.name}</small>)}<small>رندر رابط: {renderer}</small>{/swiftshader|llvmpipe|software|basic render/i.test(renderer)&&<p className="hint">درایور فعلی رندر نرم‌افزاری ارائه می‌کند؛ برای شتاب واقعی، درایور GPU سیستم باید فعال باشد.</p>}{c?.fallback_reason&&<p>{c.fallback_reason}</p>}</div>{error&&<p className="reference-error">{error}</p>}
  </div>;
}
