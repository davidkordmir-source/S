import { useEffect, useRef, useState } from 'react';
import { Focus, Hand, ImagePlus, Minus, Plus, ScanLine, SquareDashed, SquareMinus, SquarePlus } from 'lucide-react';
import { Detection, Example, fa, Rect, Scene } from './offline-api';

export type Tool = 'pan' | 'positive' | 'negative' | 'scope';
type Transform = { scale: number; x: number; y: number };
type Props = { scene: Scene | null; tool: Tool; setTool: (tool: Tool) => void; examples: Example[]; detections: Detection[]; scope: Rect | null; focus: Rect | null; selected: number | null; busy: boolean; canAnnotate: boolean; onBox: (box: Rect) => void; onImport: () => void; onSelect: (index: number) => void };

export default function ImageCanvas({ scene, tool, setTool, examples, detections, scope, focus, selected, busy, canAnnotate, onBox, onImport, onSelect }: Props) {
  const viewport = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState<Transform>({ scale: 1, x: 0, y: 0 });
  const current = useRef(transform); current.current = transform;
  const [draft, setDraft] = useState<Rect | null>(null);
  const [fullLoaded, setFullLoaded] = useState(false);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const drag = useRef<{ startX: number; startY: number; transform: Transform; pan: boolean } | null>(null);
  const fit = () => {
    if (!scene || !viewport.current) return;
    const { width, height } = viewport.current.getBoundingClientRect();
    const scale = Math.min((width-60)/scene.width, (height-60)/scene.height, 2);
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
    setTransform(previous => { const scale = Math.max(.005, Math.min(16, previous.scale*factor)); const ratio = scale/previous.scale;
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
      {([{ id: 'pan', icon: Hand, title: 'جابجایی تصویر' }, { id: 'positive', icon: SquarePlus, title: 'نمونه درست' }, { id: 'negative', icon: SquareMinus, title: 'نمونه غلط' }, { id: 'scope', icon: SquareDashed, title: 'محدوده جست‌وجو' }] as const).map(item => <button key={item.id} className={`${tool === item.id ? 'active' : ''} ${item.id}`} title={item.title} aria-label={item.title} aria-pressed={tool === item.id} disabled={!scene || busy || ((item.id === 'positive' || item.id === 'negative') && !canAnnotate)} onClick={() => setTool(item.id)}><item.icon size={17} /><span>{item.title}</span></button>)}
    </div><div className="view-tools"><button title="کوچک‌نمایی" aria-label="کوچک‌نمایی" disabled={!scene} onClick={() => zoom(1/1.25)}><Minus size={16} /></button><span dir="ltr">{scene ? `${fa(transform.scale*100)}٪` : '—'}</span><button title="بزرگ‌نمایی" aria-label="بزرگ‌نمایی" disabled={!scene} onClick={() => zoom(1.25)}><Plus size={16} /></button><i /><button title="نمایش کل تصویر" aria-label="نمایش کل تصویر" disabled={!scene} onClick={fit}><Focus size={17} /></button></div></div>
    <div ref={viewport} className={`scene-viewport tool-${tool}`} data-testid="scene-viewport" role="region" aria-label="تصویر ماهواره‌ای" onDoubleClick={fit}
      onPointerDown={event => { if (!scene || (event.button !== 0 && event.button !== 1) || (busy && tool !== 'pan')) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); drag.current = { startX: event.clientX, startY: event.clientY, transform: { ...current.current }, pan: tool === 'pan' || event.altKey || event.button === 1 }; }}
      onPointerMove={event => { if (!scene) return; const value = pixel(event.clientX, event.clientY); setCursor([Math.round(value[0]), Math.round(value[1])]); const gesture = drag.current; if (!gesture) return;
        if (gesture.pan) setTransform({ ...gesture.transform, x: gesture.transform.x+event.clientX-gesture.startX, y: gesture.transform.y+event.clientY-gesture.startY });
        else { const start = pixel(gesture.startX, gesture.startY, gesture.transform), end = pixel(event.clientX, event.clientY, gesture.transform); setDraft({ x: Math.min(start[0],end[0]), y: Math.min(start[1],end[1]), width: Math.abs(end[0]-start[0]), height: Math.abs(end[1]-start[1]) }); } }}
      onPointerUp={finish} onPointerCancel={() => { drag.current = null; setDraft(null); }} onPointerLeave={() => { if (!drag.current) setCursor(null); }}>
      {!scene ? <div className="canvas-empty"><div className="empty-orbit"><ScanLine size={50} strokeWidth={1} /><span /><i /></div><h2>تصویر خودت، جست‌وجوی خودت</h2><p>یک تصویر ماهواره‌ای وارد کن و با چند کادر، هدف را به برنامه یاد بده.</p><button className="primary-button" onPointerDown={event => event.stopPropagation()} onClick={onImport}><ImagePlus size={18} />انتخاب تصویر ماهواره‌ای</button><small>ECW · GeoTIFF · JP2 · HEIC · PNG <b>کاملاً روی همین کامپیوتر</b></small></div>
      : <div className="scene-layer" style={{ width: scene.width, height: scene.height, transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})` }}>
        <img className="scene-preview" src={scene.preview_url} alt="پیش‌نمایش تصویر ماهواره‌ای" draggable={false} width={scene.width} height={scene.height} />
        <img className={`scene-full ${fullLoaded ? 'loaded' : ''}`} src={`/api/workspace/scenes/${scene.id}/full`} alt={scene.name} draggable={false} width={scene.width} height={scene.height} onLoad={() => setFullLoaded(true)} />
        <svg className="annotation-layer" width={scene.width} height={scene.height} viewBox={`0 0 ${scene.width} ${scene.height}`}>
          {examples.filter(item => item.scene === scene.id && item.bbox).map(item => <rect key={item.id} {...item.bbox!} className={item.label ? 'example-positive' : 'example-negative'} strokeWidth={1.5/transform.scale}><title>{item.label ? 'نمونهٔ درست' : 'نمونهٔ غلط'}</title></rect>)}
          {detections.map(item => <g key={item.index} className={`detection-box ${selected === item.index ? 'selected' : ''}`} onPointerDown={event => { if (tool === 'pan') event.stopPropagation(); }} onClick={event => { if (tool === 'pan') { event.stopPropagation(); onSelect(item.index); } }}><rect {...item.bbox} strokeWidth={(selected === item.index ? 3 : 1.5)/transform.scale} /><rect x={item.bbox.x} y={Math.max(0,item.bbox.y-21/transform.scale)} width={55/transform.scale} height={21/transform.scale} className="detection-tag" /><text x={item.bbox.x+5/transform.scale} y={Math.max(14/transform.scale,item.bbox.y-6/transform.scale)} fontSize={11/transform.scale} direction="ltr">{fa(item.index+1)} · {fa(item.score,2)}</text></g>)}
          {scope && <rect {...scope} className="scope-box" strokeWidth={2/transform.scale} strokeDasharray={`${8/transform.scale} ${5/transform.scale}`} />}
          {draft && <rect {...draft} className={`draft-box ${tool}`} strokeWidth={2/transform.scale} />}
        </svg>
      </div>}
      {scene && <div className={`canvas-instruction ${tool}`}><span />{tool === 'positive' ? 'دور چیزی که می‌خواهی پیدا شود کادر بکش' : tool === 'negative' ? 'دور چیزی که هدف تو نیست کادر بکش' : tool === 'scope' ? 'بخشی از تصویر را برای جست‌وجو انتخاب کن' : 'برای جابجایی بکش؛ برای بزرگ‌نمایی از چرخ ماوس استفاده کن'}</div>}
    </div>
    <div className="canvas-status"><span>{scene ? <><span className="status-dot" />{scene.name}<i />{fa(scene.width)} × {fa(scene.height)} پیکسل</> : 'هنوز تصویری وارد نشده'}</span><span dir="ltr">{cursor ? `X ${fa(cursor[0])} · Y ${fa(cursor[1])}` : scene ? 'LOCAL IMAGE' : 'OFFLINE WORKSPACE'}</span></div>
  </section>;
}
