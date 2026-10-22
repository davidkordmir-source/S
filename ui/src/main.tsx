import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowDownToLine, ArrowLeft, ArrowUpLeft, Bookmark, Check, ChevronLeft, CircleHelp, Copy, Database, Download, FileImage, Focus, ImagePlus, Layers, LoaderCircle, MapPin, ScanLine, Search, Settings2, ShieldCheck, Sparkles, Square, Type, Upload, X } from 'lucide-react';
import '@fontsource/vazirmatn/400.css';
import '@fontsource/vazirmatn/500.css';
import '@fontsource/vazirmatn/600.css';
import '@fontsource/vazirmatn/700.css';
import ScoutMap from './Map';
import { api, Building, fa, Job, post, State } from './api';
import './style.css';

const EXAMPLES = ['ساختمان با سقف قرمز و حیاط بزرگ', 'بنای مستطیلی کنار آب', 'خانه با استخر در حیاط'];

function App() {
  const [state, setState] = useState<State | null>(null);
  const [tab, setTab] = useState<'search' | 'library'>('search');
  const [mode, setMode] = useState<'text' | 'image'>('text');
  const [query, setQuery] = useState('');
  const [photo, setPhoto] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [center, setCenter] = useState<[number, number]>([52.3728, 4.8936]);
  const [radius, setRadius] = useState(250);
  const [areaOnly, setAreaOnly] = useState(true);
  const [library, setLibrary] = useState<Building[]>([]);
  const [results, setResults] = useState<Building[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<{ text: string; error?: boolean } | null>(null);
  const [dialog, setDialog] = useState<'settings' | 'help' | null>(null);
  const [dragging, setDragging] = useState(false);
  const imageInput = useRef<HTMLInputElement>(null);
  const archiveInput = useRef<HTMLInputElement>(null);
  const busy = submitting || job?.status === 'queued' || job?.status === 'running';
  const visible = tab === 'library' ? library : results ?? library;
  const chosen = visible.find(building => building.id === selected);
  const indexed = state?.library.indexed ?? 0;
  const total = state?.library.total ?? 0;

  const refresh = useCallback(async () => {
    const [nextState, nextLibrary] = await Promise.all([api<State>('/state'), api<{ buildings: Building[] }>('/buildings')]);
    setState(nextState); setLibrary(nextLibrary.buildings);
  }, []);
  useEffect(() => { refresh().catch(error => setNotice({ text: error.message, error: true })); }, [refresh]);
  useEffect(() => {
    if (!photo) { setPreview(null); return; }
    const url = URL.createObjectURL(photo); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api<Job>(`/jobs/${job.id}`);
        if (stopped) return;
        setJob(next);
        if (next.status === 'done') {
          if (next.result?.center) setCenter([next.result.center.lat, next.result.center.lon]);
          if (next.result?.buildings) { setResults(next.result.buildings); setSelected(null); setTab('search'); }
          else if (next.result?.message) setNotice({ text: next.result.message + (next.result.provider_capped ? ' سرویس حداکثر هزار ردپای ساختمان را بررسی کرده است.' : '') });
          await refresh();
        } else if (next.status === 'failed') setNotice({ text: next.error || 'عملیات کامل نشد.', error: true });
        else if (next.status === 'cancelled') { setNotice({ text: 'عملیات لغو شد؛ مراحل تکمیل‌شده در کتابخانه باقی می‌مانند.' }); await refresh(); }
        else timer = setTimeout(poll, 900);
      } catch (error) {
        if (!stopped) { setNotice({ text: (error as Error).message, error: true }); setJob(current => current ? { ...current, status: 'failed' } : null); }
      }
    };
    timer = setTimeout(poll, 400);
    return () => { stopped = true; clearTimeout(timer); };
  }, [job?.id, job?.status === 'queued' || job?.status === 'running', refresh]);

  useEffect(() => {
    if (!dialog) return;
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setDialog(null); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [dialog]);

  const start = async (request: () => Promise<Job>) => {
    if (busy) return;
    setSubmitting(true); setNotice(null);
    try { setJob(await request()); } catch (error) { setNotice({ text: (error as Error).message, error: true }); }
    finally { setSubmitting(false); }
  };
  const pickPhoto = (file?: File) => {
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp', 'image/bmp'].includes(file.type) || file.size > 15 * 1024 * 1024) {
      setNotice({ text: 'یک تصویر JPG، PNG یا WEBP با حجم کمتر از ۱۵ مگابایت انتخاب کنید.', error: true }); return;
    }
    setPhoto(file); setMode('image'); setResults(null);
  };
  const search = () => start(() => mode === 'text' ? post<Job>('/search/text', { text: query.trim(), limit: 12, area: areaOnly ? { lat: center[0], lon: center[1], radius } : null }) : api<Job>(`/search/image${areaOnly ? `?lat=${center[0]}&lon=${center[1]}&radius=${radius}` : ''}`, { method: 'POST', headers: { 'Content-Type': photo!.type }, body: photo }));
  const importZip = (file?: File) => {
    if (!file) return;
    if (file.size > 80 * 1024 * 1024) { setNotice({ text: 'حجم فایل ZIP باید کمتر از ۸۰ مگابایت باشد.', error: true }); return; }
    start(() => api<Job>('/import', { method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file }));
  };
  const downloadResults = () => {
    const file = { type: 'FeatureCollection', features: visible.map(item => ({ type: 'Feature', geometry: item.geometry ?? { type: 'Point', coordinates: [item.lon, item.lat] }, properties: { id: item.id, title: item.title, source: item.source, cosine_similarity: item.score ?? null } })) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(file, null, 2)], { type: 'application/geo+json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'atlas-results.geojson'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 10_000);
  };
  const selectBuilding = useCallback((id: string) => setSelected(id), []);

  return <div className="app-shell">
    <header className="header">
      <a href="#" className="brand" onClick={event => { event.preventDefault(); setTab('search'); }}><span className="brand-symbol"><ScanLine size={25} strokeWidth={1.5} /></span><div><strong>اطلس<span>SCOUT</span></strong><small>ردِ ساختمان‌ها را پیدا کن</small></div></a>
      <nav className="navigation" aria-label="بخش‌های برنامه"><button className={tab === 'search' ? 'active' : ''} onClick={() => setTab('search')}><Search size={17} />جست‌وجو</button><button className={tab === 'library' ? 'active' : ''} onClick={() => setTab('library')}><Layers size={17} />کتابخانه<span className="nav-count">{fa(total)}</span></button></nav>
      <div className="header-actions"><span className="desktop-label"><span />نسخهٔ دسکتاپ</span><button className="icon-button" title="راهنما" aria-label="راهنما" onClick={() => setDialog('help')}><CircleHelp size={19} /></button><button className="icon-button" title="تنظیمات" aria-label="تنظیمات" onClick={() => setDialog('settings')}><Settings2 size={19} /></button></div>
    </header>

    <div className="workspace">
      <aside className="sidebar">
        <div className="sidebar-scroll">
          <div className="eyebrow"><span />EXPLORER / 01</div>
          <h1>هر ساختمان،<br />یک نشانه دارد<span>.</span></h1>
          <p className="intro">عکس بده یا توصیف کن؛ ساختمان‌های مشابه را روی نقشه پیدا کن.</p>
          <div className="mode-switch" role="tablist" aria-label="روش جست‌وجو"><button role="tab" aria-selected={mode === 'text'} className={mode === 'text' ? 'active' : ''} onClick={() => setMode('text')}><Type size={17} />با توضیح</button><button role="tab" aria-selected={mode === 'image'} className={mode === 'image' ? 'active' : ''} onClick={() => setMode('image')}><FileImage size={17} />با تصویر</button></div>

          {mode === 'text' ? <div className="query-section"><label htmlFor="building-description">دنبال چه ساختمانی می‌گردی؟<span>فارسی یا انگلیسی</span></label><div className="textarea-wrap"><textarea id="building-description" value={query} maxLength={500} onChange={event => setQuery(event.target.value)} placeholder="مثلاً یک ساختمان با سقف قرمز، حیاط بزرگ و استخر مستطیلی..." /><span className="character-count">{fa(query.length)} / ۵۰۰</span><Sparkles size={15} /></div><p className="field-hint">رنگ و شکل سقف، محوطه و ساختمان‌های اطراف را توصیف کن.</p><div className="examples"><span>برای شروع</span>{EXAMPLES.map(example => <button key={example} onClick={() => setQuery(example)}>{example}<ArrowUpLeft size={13} /></button>)}</div></div>
          : <div className="query-section"><label>تصویر مرجع<span>حداکثر ۱۵ مگابایت</span></label><div className={`image-drop ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); pickPhoto(event.dataTransfer.files[0]); }}>
              {preview ? <><img src={preview} alt="تصویر مرجع انتخاب‌شده" /><button className="remove-photo" aria-label="حذف تصویر" onClick={() => setPhoto(null)}><X size={16} /></button><button className="replace-photo" onClick={() => imageInput.current?.click()}>تغییر تصویر</button></> : <button className="drop-trigger" onClick={() => imageInput.current?.click()}><span><ImagePlus size={27} strokeWidth={1.4} /></span><strong>عکس را اینجا رها کن</strong><small>یا برای انتخاب فایل کلیک کن</small><b>JPG · PNG · WEBP</b></button>}
            </div><p className="field-hint">تصویر از بالا بهترین نتیجه را می‌دهد. عکسِ نما ممکن است با سقف ساختمان تطبیق نداشته باشد.</p></div>}
          <input hidden type="file" accept="image/jpeg,image/png,image/webp,image/bmp" ref={imageInput} onChange={event => { pickPhoto(event.target.files?.[0]); event.target.value = ''; }} />

          <div className="section-divider" />
          <section className="area-section"><div className="section-title"><span><Focus size={17} />محدودهٔ جست‌وجو</span><span className="small-badge">نسخهٔ اول · هلند</span></div><div className="area-name"><span className="location-icon"><MapPin size={18} /></span><div><strong>محدودهٔ انتخابی روی نقشه</strong><small dir="ltr">{center[0].toFixed(4)}, {center[1].toFixed(4)}</small></div><button title="انتخاب آمستردام" aria-label="انتخاب آمستردام" onClick={() => { setCenter([52.3728, 4.8936]); setSelected(null); }}><Focus size={17} /></button></div><label className="radius-label" htmlFor="radius"><span>شعاع بررسی</span><strong>{fa(radius)} <small>متر</small></strong></label><input id="radius" aria-label="شعاع بررسی" type="range" min="100" max="600" step="50" value={radius} onChange={event => setRadius(Number(event.target.value))} /><div className="range-labels"><span>۱۰۰ متر</span><span>۶۰۰ متر</span></div><button className="secondary-button prepare-button" disabled={busy || !state} onClick={() => start(() => post<Job>('/area', { lat: center[0], lon: center[1], radius, limit: 30 }))}><Download size={15} />دریافت حداکثر ۳۰ تصویر از محدوده<ChevronLeft size={15} /></button></section>
          <label className="scope-checkbox"><input type="checkbox" checked={areaOnly} onChange={event => setAreaOnly(event.target.checked)} />جست‌وجو فقط در محدودهٔ انتخابی</label><div className="index-line"><Database size={15} /><span>{fa(indexed)} از {fa(total)} تصویر نمایه شده</span><button disabled={busy || !total || indexed === total} onClick={() => start(() => post<Job>('/index'))}>ساخت نمایه<ArrowLeft size={13} /></button></div>
        </div>
        <div className="search-footer"><button className="search-button" disabled={busy || !indexed || (mode === 'text' ? query.trim().length < 2 : !photo)} onClick={search}>{busy ? <LoaderCircle className="spin" size={20} /> : <Search size={20} />}<span>{busy ? 'در حال پردازش' : 'پیدا کردن ساختمان‌ها'}</span><ArrowLeft size={20} /></button><p><ShieldCheck size={13} />پردازش هوش مصنوعی روی کامپیوتر شما</p></div>
      </aside>

      <main className="main-content">
        <div className="page-heading"><div><div className="breadcrumb">فضای جست‌وجو <ChevronLeft size={12} /> {tab === 'library' ? 'کتابخانهٔ محلی' : 'کاوش تصویری'}</div><h2>{tab === 'library' ? 'کتابخانهٔ ساختمان‌ها' : 'از یک نشانه، به یک مکان'}<span className="beta-tag">نسخهٔ اولیه</span></h2></div><div className="library-status"><span className={indexed ? 'ready' : ''} />{indexed ? `${fa(indexed)} تصویر آمادهٔ جست‌وجو` : 'کتابخانه هنوز آماده نیست'}</div></div>
        {notice && <div className={`notice ${notice.error ? 'error' : ''}`} role={notice.error ? 'alert' : 'status'}><span>{notice.error ? <CircleHelp size={17} /> : <Check size={17} />}{notice.text}</span><button aria-label="بستن پیام" onClick={() => setNotice(null)}><X size={16} /></button></div>}
        {busy && job && <div className="job-panel" role="status"><LoaderCircle className="spin" size={18} /><div><span>{job.message}</span><div className="progress-track"><span style={{ width: `${job.progress}%` }} /></div></div><span>{fa(job.progress)}٪</span><button title="لغو عملیات" aria-label="لغو عملیات" onClick={() => post(`/jobs/${job.id}/cancel`).then(() => setNotice({ text: 'درخواست لغو ثبت شد. مرحلهٔ جاری ممکن است کمی زمان ببرد.' })).catch(error => setNotice({ text: error.message, error: true }))}><Square size={14} /></button></div>}

        <div className="map-container"><ScoutMap center={center} radius={radius} buildings={visible} selected={selected} onCenter={setCenter} onSelect={selectBuilding} />{chosen && <div className="selected-building"><img src={chosen.image_url} alt="تصویر ساختمان انتخاب‌شده" /><div><strong>{chosen.title}</strong><span dir="ltr">{chosen.lat.toFixed(6)}, {chosen.lon.toFixed(6)}</span></div><button title="کپی مختصات" aria-label="کپی مختصات" onClick={() => navigator.clipboard.writeText(`${chosen.lat}, ${chosen.lon}`).then(() => setNotice({ text: 'مختصات کپی شد.' })).catch(() => setNotice({ text: 'کپی مختصات ممکن نشد.', error: true }))}><Copy size={16} /></button><button title="بستن جزئیات" aria-label="بستن جزئیات" onClick={() => setSelected(null)}><X size={16} /></button></div>}</div>

        <section className="results-section"><div className="results-heading"><div><span className="results-icon">{tab === 'library' ? <Layers size={18} /> : <ScanLine size={18} />}</span><h3>{tab === 'library' ? 'تصاویر ذخیره‌شده' : results ? 'ساختمان‌های مشابه' : 'نتایج جست‌وجو'}</h3><span className="count-tag">{fa(tab === 'library' || results ? visible.length : 0)}</span></div><div className="results-actions">{results && tab === 'search' && <button onClick={() => { setResults(null); setSelected(null); }}>پاک‌کردن نتایج</button>}<button disabled={!visible.length} onClick={downloadResults}><ArrowDownToLine size={15} />خروجی GeoJSON</button></div></div>
          {(tab === 'library' ? library.length > 0 : results !== null && results.length > 0) ? <><div className="results-grid">{visible.map((building, index) => <button className={`result-card ${selected === building.id ? 'selected' : ''}`} key={building.id} onClick={() => setSelected(building.id)}><div className="result-picture"><img src={building.image_url} alt={building.title} loading="lazy" /><span className="result-number">{fa(index + 1)}</span>{building.score !== undefined && <span className="score"><Sparkles size={11} />{fa(building.score, 3)} <small>شباهت</small></span>}<span className="view-location"><MapPin size={14} />نمایش روی نقشه</span></div><div className="result-info"><strong>{building.title}</strong><span><MapPin size={11} /><b dir="ltr">{building.lat.toFixed(4)}, {building.lon.toFixed(4)}</b><ArrowUpLeft size={14} /></span></div></button>)}</div>{results && tab === 'search' && <p className="results-disclaimer">امتیاز، شباهت برداری است؛ احتمال صحت یا تأیید هویت ساختمان نیست. نتایج فقط در تصاویر نمایه‌شده جست‌وجو شده‌اند.</p>}</>
          : <div className="empty-state"><div className="empty-art"><div className="scan-corner top-right" /><div className="scan-corner bottom-left" /><svg width="90" height="76" viewBox="0 0 90 76" fill="none" aria-hidden="true"><path d="M15 57 44 73 77 55 47 39 15 57Z" stroke="currentColor" strokeDasharray="3 3" opacity=".4"/><path d="m26 48 19 11 23-13V24L46 12 26 24v24Z" fill="#25372b" stroke="currentColor" strokeWidth="1.2"/><path d="m26 24 19 11 23-11M45 35v24M35 20l21 11M35 30v22M56 29v24" stroke="currentColor" strokeWidth="1.2"/><path d="m18 21 28-16 30 17" stroke="currentColor" opacity=".5"/><path d="M17 35v12M74 32v12" stroke="currentColor" strokeDasharray="3 3" opacity=".5"/></svg><span className="scan-line" /></div><div><h4>{tab === 'library' ? 'کتابخانه‌ات را بساز' : !total ? 'اولین کشف از اینجا شروع می‌شود' : !indexed ? 'یک قدم تا اولین جست‌وجو' : 'حالا نشانه‌ات را وارد کن'}</h4><p>{tab === 'library' || !total ? 'تصاویر محدوده را دریافت کن یا یک مجموعهٔ دارای مختصات وارد کن.' : !indexed ? 'برای مقایسهٔ تصاویر، دکمهٔ «ساخت نمایه» را بزن.' : 'ساختمان را توصیف کن یا عکسش را بده و جست‌وجو را شروع کن.'}</p><button className="import-link" disabled={busy} onClick={() => archiveInput.current?.click()}><Upload size={14} />واردکردن مجموعه تصاویر<ArrowLeft size={14} /></button></div></div>}
        </section>
        <footer className="workspace-footer"><span><span />ATLAS SCOUT <i>۰.۱</i></span><span>دادهٔ آنلاین: PDOK هلند · جست‌وجوی محلی با CLIP</span><button onClick={() => setDialog('help')}>راهنمای شروع<ArrowUpLeft size={12} /></button></footer>
      </main>
    </div>
    <input hidden ref={archiveInput} type="file" accept=".zip,application/zip" onChange={event => { importZip(event.target.files?.[0]); event.target.value = ''; }} />

    {dialog && <div className="modal-backdrop" onClick={() => setDialog(null)}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="dialog-title" onClick={event => event.stopPropagation()}><div className="modal-heading"><span className="modal-icon">{dialog === 'settings' ? <Settings2 size={21} /> : <CircleHelp size={21} />}</span><div><span>ATLAS SCOUT</span><h2 id="dialog-title">{dialog === 'settings' ? 'تنظیمات و وضعیت برنامه' : 'از کجا شروع کنم؟'}</h2></div><button autoFocus className="icon-button" aria-label="بستن پنجره" onClick={() => setDialog(null)}><X size={20} /></button></div>
      {dialog === 'settings' ? <><p className="modal-intro">مدل‌ها فقط بار اول دانلود می‌شوند. پس از آن، جست‌وجوی کتابخانهٔ آماده به اینترنت نیاز ندارد.</p><div className="settings-row"><span>مدل تصویر</span><strong>{state?.models.image_loaded ? 'بارگذاری شده' : 'هنوز بارگذاری نشده'}</strong></div><div className="settings-row"><span>مدل متن فارسی و انگلیسی</span><strong>{state?.models.text_loaded ? 'بارگذاری شده' : 'هنوز بارگذاری نشده'}</strong></div><div className="settings-row"><span>پردازش</span><strong>محلی · CPU</strong></div><button className="secondary-button full-width" disabled={busy} onClick={() => { setDialog(null); start(() => post<Job>('/models')); }}><Download size={16} />دانلود و آماده‌سازی مدل‌ها</button><div className="settings-path"><span>محل ذخیرهٔ تصاویر، مدل‌ها و گزارش خطا</span><code dir="ltr">{state?.data_directory ?? '—'}</code></div><div className="modal-button-row"><button className="secondary-button" disabled={busy} onClick={() => { setDialog(null); archiveInput.current?.click(); }}><Upload size={15} />واردکردن مجموعه</button><button className="secondary-button" disabled={!total || busy || total === indexed} onClick={() => { setDialog(null); start(() => post<Job>('/index')); }}><Database size={15} />تکمیل نمایه</button></div></>
      : <><ol className="help-steps"><li><span>۱</span><div><strong>یک محدوده انتخاب کن</strong><p>مرکز پیش‌فرض آمستردام است. روی نقشهٔ هلند کلیک کن، شعاع را تنظیم کن و تصاویر را دریافت کن.</p></div></li><li><span>۲</span><div><strong>نمایهٔ هوش مصنوعی را بساز</strong><p>دکمهٔ «ساخت نمایه» را بزن. اولین بار دانلود مدل و پردازش تصاویر زمان می‌برد.</p></div></li><li><span>۳</span><div><strong>نشانه بده و مقایسه کن</strong><p>عکس هوایی یا توضیح فارسی و انگلیسی وارد کن. روی هر نتیجه کلیک کن تا محلش را ببینی.</p></div></li></ol><div className="help-note"><Bookmark size={19} /><p>مجموعهٔ شخصی می‌تواند متعلق به هر نقطهٔ دنیا باشد: یک ZIP شامل <code>buildings.json</code> و تصاویر. قالب دقیق در راهنمای پروژه آمده است.</p></div><p className="modal-intro">این نسخه روی ظاهر سقف و محوطه تمرکز دارد؛ یافتن قطعی یک ساختمان از عکس خیابانی یا جست‌وجوی تمام دنیا را تضمین نمی‌کند.</p><button className="search-button" onClick={() => setDialog(null)}>شروع کاوش<ArrowLeft size={18} /></button></>}
    </section></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
