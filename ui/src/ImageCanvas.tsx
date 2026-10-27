import { useEffect, useMemo, useRef, useState } from 'react';
import { Focus, Hand, ImagePlus, Minus, Plus, ScanLine, SquareDashed, SquareMinus, SquarePlus } from 'lucide-react';
import { Detection, Example, fa, grade, Level, levels, percent, post, Rect, Scene } from './offline-api';
import TileRenderer from './TileRenderer';

export type Tool = 'pan' | 'positive' | 'negative' | 'scope' | 'reference' | 'negative-reference';
type Transform = { scale: number; x: number; y: number };
type Props = { advanced?: boolean; reference?: Rect | null; referenceBoxes?: Rect[]; similarRun?: string | null; level?: Level; scene: Scene | null; tool: Tool; setTool: (tool: Tool) => void; examples: Example[]; detections: Detection[]; scope: Rect | null; focus: Rect | null; selected: number | null; busy: boolean; canAnnotate: boolean; onBox: (box: Rect) => void; onImport: () => void; onSelect: (index: number) => void };

export default function ImageCanvas({ advanced = true, reference, referenceBoxes = [], similarRun, level = 'all', scene, tool, setTool, examples, detections, scope, focus, selected, busy, canAnnotate, onBox, onImport, onSelect }: Props) {
  const viewport = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState<Transform>({ scale: 1, x: 0, y: 0 });
  const current = useRef(transform); current.current = transform;
  const [draft, setDraft] = useState<Rect | null>(null);
  const [visible, setVisible] = useState<Detection[]>([]);
  const [truncated, setTruncated] = useState(false);
  const [fullLoaded, setFullLoaded] = useState(false);
  const [gpuReady, setGpuReady] = useState(false);
  const [bounds, setBounds] = useState({ width: 1000, height: 700 });
  const [tileView, setTileView] = useState(transform);
  const minimumZoom = useRef(.000001);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const drag = useRef<{ startX: number; startY: number; transform: Transform; pan: boolean } | null>(null);
  const fit = () => {
    if (!scene || !viewport.current) return;
    const { width, height } = viewport.current.getBoundingClientRect();
    const scale = Math.min((width-60)/scene.width, (height-60)/scene.height, 2);
    minimumZoom.current = scale / 8;
    setBounds({ width, height });
    setTransform({ scale, x: (width-scene.width*scale)/2, y: (height-scene.height*scale)/2 });
  };
  useEffect(() => {
    fit(); setFullLoaded(false); setDraft(null);
  }, [scene?.id]);
  useEffect(() => {
    if (!viewport.current) return;
    const observer = new ResizeObserver(() => { if (!drag.current) fit(); });
    observer.observe(viewport.current); return () => observer.disconnect();
  }, [scene?.id]);
  const zoom = (factor: number, x?: number, y?: number) => {
    const bounds = viewport.current?.getBoundingClientRect(); if (!bounds) return;
    const anchorX = x ?? bounds.width/2, anchorY = y ?? bounds.height/2;
    setTransform(previous => { const scale = Math.max(minimumZoom.current, Math.min(16, previous.scale*factor)); const ratio = scale/previous.scale;
      return { scale, x: anchorX-(anchorX-previous.x)*ratio, y: anchorY-(anchorY-previous.y)*ratio }; });
  };
  useEffect(() => {
    const element = viewport.current; if (!element) return;
    const wheel = (event: WheelEvent) => { event.preventDefault(); const bounds = element.getBoundingClientRect(); zoom(event.deltaY < 0 ? 1.15 : 1/1.15, event.clientX-bounds.left, event.clientY-bounds.top); };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => element.removeEventListener('wheel', wheel);
  }, []);
  useEffect(() => {
    if (!focus || !viewport.current || !scene) return;
    const bounds = viewport.current.getBoundingClientRect();
    const scale = Math.min(6, (bounds.width-140)/focus.width, (bounds.height-140)/focus.height);
    setTransform({ scale, x: bounds.width/2-(focus.x+focus.width/2)*scale, y: bounds.height/2-(focus.y+focus.height/2)*scale });
  }, [focus]);
  useEffect(() => {
    const timer = setTimeout(() => setTileView(transform), 140);
    return () => clearTimeout(timer);
  }, [transform]);
  const tiles = useMemo(() => {
    if (!scene) return [];
    const maxLevel = Math.max(0, Math.ceil(Math.log2(Math.max(scene.width, scene.height) / 512)));
    const level = Math.max(0, Math.min(maxLevel, Math.floor(Math.log2(1 / tileView.scale))));
    const span = 512 * 2 ** level;
    const left = Math.max(0, Math.floor(-tileView.x / tileView.scale / span));
    const top = Math.max(0, Math.floor(-tileView.y / tileView.scale / span));
    const right = Math.min(Math.ceil(scene.width / span), Math.ceil((bounds.width - tileView.x) / tileView.scale / span));
    const bottom = Math.min(Math.ceil(scene.height / span), Math.ceil((bounds.height - tileView.y) / tileView.scale / span));
    const result = [];
    for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) {
      if (result.length >= 64) return result;
      result.push({ key: `${scene.id}-${level}-${x}-${y}`, url: `/api/workspace/scenes/${scene.id}/tiles/${level}/${x}/${y}`, x: x * span, y: y * span, width: Math.min(span, scene.width - x * span), height: Math.min(span, scene.height - y * span) });
    }
    return result;
  }, [scene?.id, tileView, bounds]);
  useEffect(() => {
    setVisible([]); setTruncated(false);
    if (!similarRun || !scene) return;
    const x = Math.max(0, -tileView.x / tileView.scale), y = Math.max(0, -tileView.y / tileView.scale);
    const width = Math.min(scene.width, (bounds.width - tileView.x) / tileView.scale) - x;
    const height = Math.min(scene.height, (bounds.height - tileView.y) / tileView.scale) - y;
    if (width < 8 || height < 8) return;
    let stopped = false;
    post<{ detections: Detection[]; truncated: boolean }>(`/workspace/runs/${similarRun}/visible?level=${level}`, { x, y, width, height }).then(result => {
      if (!stopped) { setVisible(result.detections); setTruncated(result.truncated); }
    }).catch(() => { if (!stopped) setVisible(detections); });
    return () => { stopped = true; };
  }, [similarRun, level, scene?.id, tileView, bounds]);
  const displayed = similarRun ? visible : detections;
  const pixel = (clientX: number, clientY: number, value = current.current) => {
    const bounds = viewport.current!.getBoundingClientRect();
    return [Math.max(0, Math.min(scene!.width, (clientX-bounds.left-value.x)/value.scale)), Math.max(0, Math.min(scene!.height, (clientY-bounds.top-value.y)/value.scale))];
  };
  const finish = (event: React.PointerEvent) => {
    const gesture = drag.current; if (!gesture) return;
    drag.current = null;
    if (!gesture.pan && draft && draft.width >= 8 && draft.height >= 8 && !busy) onBox({ x: Math.round(draft.x), y: Math.round(draft.y), width: Math.round(draft.width), height: Math.round(draft.height) });
    setDraft(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return <section className="canvas-panel">
    <div className="canvas-toolbar"><div className="drawing-tools" role="toolbar" aria-label="ابزارهای تصویر">
      {([{ id: 'pan', icon: Hand, title: 'جابجایی تصویر' }, { id: 'reference', icon: ScanLine, title: 'نمونهٔ مرجع' }, { id: 'positive', icon: SquarePlus, title: 'نمونه درست' }, { id: 'negative', icon: SquareMinus, title: 'نمونه غلط' }, { id: 'scope', icon: SquareDashed, title: 'محدوده جست‌وجو' }] as const).filter(item => advanced || (item.id !== 'positive' && item.id !== 'negative')).map(item => <button key={item.id} className={`${tool === item.id ? 'active' : ''} ${item.id}`} title={item.title} aria-label={item.title} aria-pressed={tool === item.id} disabled={!scene || busy || ((item.id === 'positive' || item.id === 'negative') && !canAnnotate)} onClick={() => setTool(item.id)}><item.icon size={17} /><span>{item.title}</span></button>)}
    </div><div className="view-tools"><button title="کوچک‌نمایی" aria-label="کوچک‌نمایی" disabled={!scene} onClick={() => zoom(1/1.25)}><Minus size={16} /></button><span dir="ltr">{scene ? `${fa(transform.scale*100)}٪` : '—'}</span><button title="بزرگ‌نمایی" aria-label="بزرگ‌نمایی" disabled={!scene} onClick={() => zoom(1.25)}><Plus size={16} /></button><i /><button title="نمایش کل تصویر" aria-label="نمایش کل تصویر" disabled={!scene} onClick={fit}><Focus size={17} /></button></div></div>
    <div ref={viewport} className={`scene-viewport tool-${tool}`} data-testid="scene-viewport" role="region" aria-label="تصویر ماهواره‌ای" onDoubleClick={fit}
      onPointerDown={event => { if (!scene || (event.button !== 0 && event.button !== 1) || (busy && tool !== 'pan')) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); drag.current = { startX: event.clientX, startY: event.clientY, transform: { ...current.current }, pan: tool === 'pan' || event.altKey || event.button === 1 }; }}
      onPointerMove={event => { if (!scene) return; const value = pixel(event.clientX, event.clientY); setCursor([Math.round(value[0]), Math.round(value[1])]); const gesture = drag.current; if (!gesture) return;
        if (gesture.pan) setTransform({ ...gesture.transform, x: gesture.transform.x+event.clientX-gesture.startX, y: gesture.transform.y+event.clientY-gesture.startY });
        else { const start = pixel(gesture.startX, gesture.startY, gesture.transform), end = pixel(event.clientX, event.clientY, gesture.transform); setDraft({ x: Math.min(start[0],end[0]), y: Math.min(start[1],end[1]), width: Math.abs(end[0]-start[0]), height: Math.abs(end[1]-start[1]) }); } }}
      onPointerUp={finish} onPointerCancel={() => { drag.current = null; setDraft(null); }} onPointerLeave={() => { if (!drag.current) setCursor(null); }}>
      {!scene ? <div className="canvas-empty"><div className="empty-orbit"><ScanLine size={50} strokeWidth={1} /><span /><i /></div><h2>تصویر خودت، جست‌وجوی خودت</h2><p>تصویر اصلی را باز کن؛ با یک اسکرین‌شات یا انتخاب یک قسمت، شبیه‌هایش را پیدا کن.</p><button className="primary-button" onPointerDown={event => event.stopPropagation()} onClick={onImport}><ImagePlus size={18} />انتخاب تصویر ماهواره‌ای</button><small>ECW · GeoTIFF · JP2 · HEIC · PNG <b>کاملاً روی همین کامپیوتر</b></small></div>
      : <><TileRenderer scene={scene.id} preview={scene.preview_url} width={scene.width} height={scene.height} tiles={tiles} transform={transform} ready={setGpuReady}/><div className={`scene-layer ${gpuReady ? 'gpu-active' : ''}`} style={{ width: scene.width, height: scene.height, transform: `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})` }}>
        <img className="scene-preview" src={scene.preview_url} alt="پیش‌نمایش تصویر ماهواره‌ای" draggable={false} width={scene.width} height={scene.height} />
        <img className={`scene-full ${fullLoaded ? 'loaded' : ''}`} src={scene.preview_url} alt={scene.name} draggable={false} width={scene.width} height={scene.height} onLoad={() => setFullLoaded(true)} />
        <div className="scene-tiles">{!gpuReady && tiles.map(tile => <img key={tile.key} src={tile.url} alt="" draggable={false} style={{ left: tile.x, top: tile.y, width: tile.width, height: tile.height }} />)}</div>
        <svg className="annotation-layer" width={scene.width} height={scene.height} viewBox={`0 0 ${scene.width} ${scene.height}`}>
          {examples.filter(item => item.scene === scene.id && item.bbox).map(item => <rect key={item.id} {...item.bbox!} className={item.label ? 'example-positive' : 'example-negative'} strokeWidth={1.5/transform.scale}><title>{item.label ? 'نمونهٔ درست' : 'نمونهٔ غلط'}</title></rect>)}
          {displayed.map(item => <g key={item.index} className={`detection-box ${similarRun ? `level-${(item.level ?? grade(item.score))}` : ''} ${selected === item.index ? 'selected' : ''}`} onPointerDown={event => { if (tool === 'pan') event.stopPropagation(); }} onClick={event => { if (tool === 'pan') { event.stopPropagation(); onSelect(item.index); } }}>{item.quad && <polygon className="geometry-quad" points={item.quad.map(p=>p.join(',')).join(' ')} strokeWidth={2/transform.scale}/>}<title>{`نتیجه ${fa(item.index+1)} · ${percent(item.score)} شباهت · ${levels.find(l => l.id === (item.level ?? grade(item.score)))?.label}`}</title><rect {...item.bbox} strokeWidth={(selected === item.index ? 3 : 1.5)/transform.scale} /><rect x={item.bbox.x} y={Math.max(0,item.bbox.y-21/transform.scale)} width={92/transform.scale} height={21/transform.scale} className="detection-tag" /><text x={item.bbox.x+5/transform.scale} y={Math.max(14/transform.scale,item.bbox.y-6/transform.scale)} fontSize={11/transform.scale} direction="ltr">{fa(item.index+1)} · {similarRun ? percent(item.score) : fa(item.score,2)}</text></g>)}
          {referenceBoxes.map((box,i) => <rect key={`ref-${i}`} {...box} className="reference-box" strokeWidth={2/transform.scale} />)}
          {reference && <rect {...reference} className="reference-box" strokeWidth={2/transform.scale} />}
          {scope && <rect {...scope} className="scope-box" strokeWidth={2/transform.scale} strokeDasharray={`${8/transform.scale} ${5/transform.scale}`} />}
          {draft && <rect {...draft} className={`draft-box ${tool}`} strokeWidth={2/transform.scale} />}
        </svg>
      </div></>}
      {similarRun && <div className="map-level-legend" aria-label="راهنمای رنگ شباهت">{levels.filter(l => level === 'all' || level === l.id).map(l => <span className={`level-${l.id}`} key={l.id}><i />{l.label} <small>{l.range}</small></span>)}</div>}
      {truncated && <div className="density-note">تعداد نتایج این نما زیاد است؛ برای دیدن جزئیات بیشتر بزرگ‌نمایی کن.</div>}
      {scene && <div className={`canvas-instruction ${tool}`}><span />{tool === 'negative-reference' ? 'دور چیزی که نمی‌خواهی پیدا شود کادر بکش' : tool === 'reference' ? 'دور قسمت مرجع کادر بکش؛ موارد مشابه همین قسمت جست‌وجو می‌شوند' : tool === 'positive' ? 'دور چیزی که می‌خواهی پیدا شود کادر بکش' : tool === 'negative' ? 'دور چیزی که هدف تو نیست کادر بکش' : tool === 'scope' ? 'بخشی از تصویر را برای جست‌وجو انتخاب کن' : 'برای جابجایی بکش؛ برای بزرگ‌نمایی از چرخ ماوس استفاده کن'}</div>}
    </div>
    <div className="canvas-status"><span>{scene ? <><span className="status-dot" />{scene.name}<i />{fa(scene.width)} × {fa(scene.height)} پیکسل</> : 'هنوز تصویری وارد نشده'}</span><span dir="ltr">{cursor ? `X ${fa(cursor[0])} · Y ${fa(cursor[1])}` : scene ? 'LOCAL IMAGE' : 'OFFLINE WORKSPACE'}</span></div>
  </section>;
}
