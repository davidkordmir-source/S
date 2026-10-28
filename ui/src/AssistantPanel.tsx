import { useEffect, useState } from 'react';
import { BrainCircuit, Check, Pencil, Send, Trash2, X } from 'lucide-react';
import { api, AssistantPlan, Job, post, Scene } from './offline-api';

type Message = { id: string; role: 'user'|'assistant'; content: string; model?: string };
type Props = { group: string; scene: Scene|null; busy: boolean; available: boolean; job: Job|null; start: (request:()=>Promise<Job>)=>void; usePlan:(message:string,key:string)=>void };
export default function AssistantPanel({group,scene,busy,available,job,start,usePlan}:Props) {
  const [messages,setMessages]=useState<Message[]>([]),[text,setText]=useState(''),[error,setError]=useState('');
  const [editing,setEditing]=useState(''),[editText,setEditText]=useState(''),[saving,setSaving]=useState(false);
  const [reply,setReply]=useState<{message:string;plan:AssistantPlan}|null>(null);
  const load=()=>api<{messages:Message[]}>(`/workspace/assistant/messages${group?'?group='+encodeURIComponent(group):''}`).then(r=>setMessages(r.messages));
  useEffect(()=>{let stopped=false;setReply(null);api<{messages:Message[]}>(`/workspace/assistant/messages${group?'?group='+encodeURIComponent(group):''}`).then(r=>{if(!stopped)setMessages(r.messages);}).catch(e=>{if(!stopped)setError(e.message);});return()=>{stopped=true;};},[group]);
  useEffect(()=>{if(job?.kind==='assistant'&&job.status==='done'){load().catch(e=>setError(e.message));const plan=job.result?.assistant;if(plan)setReply({message:String((job.result as {instruction?:string}).instruction??''),plan});}},[job?.id,job?.status]);
  const send=()=>{if(!text.trim()||busy||saving)return;setError('');setReply(null);const message=text.trim();start(()=>post<Job>('/workspace/assistant',{message,group:group||null,scene:scene?.id||null}));setText('');};
  const change=async(path:string,method:string,value?:unknown)=>{setSaving(true);setError('');try{await api(path,{method,...(value===undefined?{}:{headers:{'Content-Type':'application/json'},body:JSON.stringify(value)})});setEditing('');setReply(null);await load();}catch(e){setError((e as Error).message);}finally{setSaving(false);}};
  return <section className="assistant-panel" aria-label="هوش مصنوعی آفلاین">
    <h3><BrainCircuit size={18}/>هوش مصنوعی آفلاین · تصویر و متن</h3>
    <p className="hint">به فارسی خواسته‌ات را توضیح بده و دربارهٔ نمونه‌ها گفتگو کن. پاسخ با مدل محلی ساخته می‌شود؛ تصویر و متن روی همین رایانه می‌مانند.</p>
    {!available&&<p className="reference-error">مدل زبان و تصویر نصب نیست. برای استفاده، بستهٔ کامل مدل‌های این نسخه لازم است.</p>}
    <div className="assistant-history" aria-live="polite">{messages.map(m=><article key={m.id} className={`assistant-message ${m.role}`}>
      <div><b>{m.role==='user'?'شما':'دستیار محلی'}</b><span>{m.role==='user'&&<button title="ویرایش پیام" disabled={busy||saving} onClick={()=>{setEditing(m.id);setEditText(m.content);}}><Pencil size={12}/></button>}<button title="حذف پیام" disabled={busy||saving} onClick={()=>change(`/workspace/assistant/messages/${m.id}`,'DELETE')}><Trash2 size={12}/></button></span></div>
      {editing===m.id?<><textarea aria-label="ویرایش پیام" value={editText} maxLength={8000} onChange={e=>setEditText(e.target.value)}/><button disabled={busy||saving||!editText.trim()} onClick={()=>change(`/workspace/assistant/messages/${m.id}`,'PATCH',{message:editText.trim()})}><Check size={14}/>ذخیره</button><button onClick={()=>setEditing('')}><X size={14}/></button></>:<p>{m.content}</p>}
    </article>)}</div>
    <textarea aria-label="پیام به هوش مصنوعی آفلاین" placeholder="چه چیزی پیدا کنم؟ مشخصات، تفاوت‌های مجاز و چیزهایی که نمی‌خواهی را توضیح بده…" value={text} maxLength={8000} disabled={busy||saving||!available} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();send();}}}/>
    <div className="assistant-actions"><button disabled={busy||saving||!available||!text.trim()} onClick={send}><Send size={14}/>بررسی خواسته و نمونه‌ها</button>{messages.length>0&&<button disabled={busy||saving} onClick={()=>change(`/workspace/assistant/messages${group?'?group='+encodeURIComponent(group):''}`,'DELETE')}><Trash2 size={13}/>پاک‌کردن گفتگو</button>}</div>
    {reply&&reply.plan.intent==='search'&&<button className="train-button" disabled={busy} onClick={()=>usePlan(reply.message,reply.plan.prepared_key)}><Check size={16}/>استفاده از این درخواست برای جست‌وجو</button>}
    {error&&<p className="reference-error" role="alert">{error}</p>}
  </section>;
}
