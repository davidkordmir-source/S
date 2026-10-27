import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, Eraser, MousePointer2, ScanLine, SquareDashed, X } from 'lucide-react';
import { Job, post, Reference, Rect } from './offline-api';

type Point={x:number;y:number;label:0|1};
type Props={reference:Reference;group:string;busy:boolean;job:Job|null;start:(request:()=>Promise<Job>)=>void;close:()=>void;updated:()=>Promise<void>};
export default function ReferenceEditor({reference,group,busy,job,start,close,updated}:Props) {
  const [tool,setTool]=useState<'point'|'background'|'box'|'polygon'>('point');
  const [points,setPoints]=useState<Point[]>([]),[polygon,setPolygon]=useState<Point[]>([]),[box,setBox]=useState<Rect|null>(null);
  const [draft,setDraft]=useState<{draft:string;mask_url:string}|null>(null),[error,setError]=useState(''),[saving,setSaving]=useState(false);
  const [size,setSize]=useState({width:1024,height:1024});const surface=useRef<HTMLDivElement>(null),origin=useRef<Point|null>(null);
  const source=reference.source_image_url??reference.image_url;
  const point=(event:React.PointerEvent):Point=>{const b=surface.current!.getBoundingClientRect();return {x:Math.max(0,Math.min(size.width-1,(event.clientX-b.left)*size.width/b.width)),y:Math.max(0,Math.min(size.height-1,(event.clientY-b.top)*size.height/b.height)),label:tool==='background'?0:1};};
  useEffect(()=>{if(job?.kind==='segment-reference' && job.status==='done' && job.result?.draft && job.result.reference===(reference.source_reference??reference.id))setDraft({draft:job.result.draft,mask_url:job.result.mask_url!});},[job]);
  const submit=(manual=false)=>{setDraft(null);setError('');start(()=>post<Job>(`/workspace/references/${reference.id}/segment`,{points,box,polygon:manual?polygon:[]}));};
  const apply=async()=>{if(!draft)return;setSaving(true);try{await post(`/workspace/reference-groups/${group}/apply-mask`,{reference:reference.id,draft:draft.draft});await updated();close();}catch(e){setError((e as Error).message);}finally{setSaving(false);}};
  const reset=async()=>{setSaving(true);try{await post(`/workspace/reference-groups/${group}/reset-mask`,{reference:reference.id});await updated();close();}catch(e){setError((e as Error).message);}finally{setSaving(false);}};
  return createPortal(<div className="modal-backdrop"><section className="modal reference-editor" role="dialog" aria-modal="true" aria-label="جداکردن خود هدف"><div className="modal-heading"><span><ScanLine size={23}/></span><h2>خودِ هدف را مشخص کن</h2><button aria-label="بستن ویرایش هدف" onClick={close}><X size={20}/></button></div>
    <p className="hint">داخل هدف کلیک کن یا دور آن کادر بکش. برای اصلاح، روی پس‌زمینه نقطهٔ قرمز بگذار؛ ماسک پیشنهادی را پیش از اعمال بررسی کن.</p>
    <div className="mask-toolbar"><button className={tool==='point'?'active':''} disabled={busy} onClick={()=>setTool('point')}><MousePointer2 size={15}/>داخل هدف</button><button className={tool==='background'?'active':''} disabled={busy} onClick={()=>setTool('background')}>پس‌زمینه</button><button className={tool==='box'?'active':''} disabled={busy} onClick={()=>setTool('box')}><SquareDashed size={15}/>کادر</button><button className={tool==='polygon'?'active':''} disabled={busy} onClick={()=>setTool('polygon')}>چندضلعی دستی</button><button disabled={busy} onClick={()=>{setPoints([]);setPolygon([]);setBox(null);setDraft(null);}}><Eraser size={15}/>پاک‌کردن نشانه‌ها</button></div>
    <div ref={surface} className="mask-surface" style={{aspectRatio:`${size.width}/${size.height}`,width:`min(100%, ${55*size.width/size.height}vh)`}} onPointerDown={e=>{if(busy||saving)return;e.preventDefault();const p=point(e);if(tool==='box'){origin.current=p;e.currentTarget.setPointerCapture(e.pointerId);setBox({x:p.x,y:p.y,width:0,height:0});}else if(tool==='polygon')setPolygon(v=>[...v,p]);else setPoints(v=>[...v,p]);}} onPointerMove={e=>{if(origin.current){const p=point(e),o=origin.current;setBox({x:Math.min(p.x,o.x),y:Math.min(p.y,o.y),width:Math.abs(p.x-o.x),height:Math.abs(p.y-o.y)});}}} onPointerUp={e=>{origin.current=null;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);}}>
      <img src={source} alt="عکس اصلی برای جداسازی هدف" draggable={false} onLoad={e=>setSize({width:e.currentTarget.naturalWidth,height:e.currentTarget.naturalHeight})}/>
      {draft && <img className="mask-preview" src={draft.mask_url} alt="پیش‌نمایش ماسک هدف"/>}
      <svg viewBox={`0 0 ${size.width} ${size.height}`}><g>{points.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r={Math.max(size.width,size.height)/90} fill={p.label?'#2bd080':'#ef6057'} stroke="white" strokeWidth={Math.max(size.width,size.height)/500}/>)}</g>{box && <rect {...box} fill="#55b8ff22" stroke="#45b3fa" strokeWidth={Math.max(size.width,size.height)/300}/>}{polygon.length>0 && <polygon points={polygon.map(p=>`${p.x},${p.y}`).join(' ')} fill="#f4d48733" stroke="#f4d487" strokeWidth={Math.max(size.width,size.height)/300}/>}</svg>
    </div>
    {busy && <p className="hint" role="status">{job?.message??'در حال جداسازی هدف…'}</p>}
    {job?.kind==='segment-reference' && job.status==='failed' && <p className="reference-error">{job.error}</p>}{error&&<p className="reference-error">{error}</p>}
    <div className="mask-actions"><button className="primary-button" disabled={busy||saving||(!points.length&&!box)} onClick={()=>submit()}><ScanLine size={16}/>جداکردن با هوش مصنوعی</button><button className="secondary-button" disabled={busy||saving||polygon.length<3} onClick={()=>submit(true)}>ساخت ماسک دستی</button><button className="primary-button" disabled={busy||saving||!draft} onClick={apply}><Check size={16}/>اعمال همین ماسک</button>{reference.has_mask && <button disabled={busy||saving} onClick={reset}>بازگشت به عکس کامل</button>}</div>
  </section></div>, document.body);
}
