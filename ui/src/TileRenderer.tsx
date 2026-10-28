import { useEffect, useRef } from 'react';

type Tile = { key: string; url: string; x: number; y: number; width: number; height: number };
type Props = { scene: string; preview: string; width: number; height: number; tiles: Tile[]; transform: { x: number; y: number; scale: number }; ready: (active: boolean) => void };
type Texture = { texture: WebGLTexture; image: HTMLImageElement; loaded: boolean; used: number };

/** Viewport-sized GPU surface: never allocates a texture as large as the source raster. */
export default function TileRenderer(props: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const latest = useRef(props); latest.current = props;
  const schedule = useRef<() => void>(() => {});
  useEffect(() => {
    const element = canvas.current!;
    const gl = element.getContext('webgl', { alpha: false, antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false });
    if (!gl) { element.dataset.backend = 'fallback'; latest.current.ready(false); return; }
    let disposed = false, frame = 0, stamp = 0, failed = false;
    const textures = new Map<string, Texture>();
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)!; gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) { gl.deleteShader(shader); throw new Error('Shader unavailable'); }
      return shader;
    };
    let program: WebGLProgram, vertex: WebGLShader, fragment: WebGLShader;
    try {
      vertex = compile(gl.VERTEX_SHADER, 'attribute vec2 corner; uniform vec4 rect; uniform vec2 viewport; varying vec2 uv; void main(){uv=corner;vec2 pixel=rect.xy+corner*rect.zw;gl_Position=vec4(pixel/viewport*vec2(2.0,-2.0)+vec2(-1.0,1.0),0.0,1.0);}');
      fragment = compile(gl.FRAGMENT_SHADER, 'precision mediump float; uniform sampler2D image; varying vec2 uv; void main(){gl_FragColor=texture2D(image,uv);}');
      program = gl.createProgram()!; gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Renderer unavailable');
    } catch { element.dataset.backend = 'fallback'; latest.current.ready(false); gl.getExtension('WEBGL_lose_context')?.loseContext(); return; }
    gl.useProgram(program);
    const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]), gl.STATIC_DRAW);
    const corner = gl.getAttribLocation(program, 'corner'); gl.enableVertexAttribArray(corner); gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    const rectangle = gl.getUniformLocation(program, 'rect'), viewport = gl.getUniformLocation(program, 'viewport');
    gl.uniform1i(gl.getUniformLocation(program, 'image'), 0);
    const fallback = () => { if (!disposed) { failed = true; element.dataset.backend = 'fallback'; latest.current.ready(false); } };
    const request = () => { if (!disposed && !failed && !frame) frame = requestAnimationFrame(draw); };
    const obtain = (key: string, url: string): Texture => {
      const existing = textures.get(key); if (existing) { existing.used = ++stamp; return existing; }
      const texture = gl.createTexture()!, image = new Image();
      const item = { texture, image, loaded: false, used: ++stamp }; textures.set(key, item);
      image.decoding = 'async';
      image.onload = () => {
        if (disposed || failed || textures.get(key) !== item) return;
        try {
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
          if (gl.getError() !== gl.NO_ERROR) { fallback(); return; }
          item.loaded = true; request();
        } catch { fallback(); }
      };
      image.onerror = () => { if (key.startsWith('preview:')) fallback(); };
      image.src = url; return item;
    };
    function draw() {
      frame = 0; if (disposed || failed || gl!.isContextLost()) return;
      const p = latest.current, bounds = element.getBoundingClientRect(), ratio = Math.min(2, devicePixelRatio || 1);
      const w = Math.max(1, Math.round(bounds.width * ratio)), h = Math.max(1, Math.round(bounds.height * ratio));
      if (element.width !== w || element.height !== h) { element.width = w; element.height = h; }
      gl!.viewport(0, 0, w, h); gl!.clearColor(.063, .094, .078, 1); gl!.clear(gl!.COLOR_BUFFER_BIT);
      gl!.uniform2f(viewport, bounds.width || 1, bounds.height || 1);
      const paint = (item: Texture, x: number, y: number, width: number, height: number) => {
        if (!item.loaded) return;
        gl!.bindTexture(gl!.TEXTURE_2D, item.texture);
        gl!.uniform4f(rectangle, p.transform.x + x * p.transform.scale, p.transform.y + y * p.transform.scale, width * p.transform.scale, height * p.transform.scale);
        gl!.drawArrays(gl!.TRIANGLES, 0, 6);
      };
      const previewKey = `preview:${p.scene}`, base = obtain(previewKey, p.preview);
      paint(base, 0, 0, p.width, p.height);
      const visible = new Set([previewKey]);
      for (const tile of p.tiles) { visible.add(tile.key); paint(obtain(tile.key, tile.url), tile.x, tile.y, tile.width, tile.height); }
      element.dataset.visibleTiles = String(p.tiles.length);
      element.dataset.loadedTiles = String(p.tiles.filter(t => textures.get(t.key)?.loaded).length);
      // 128 bounded tiles, normally under 100 MiB including the bounded overview.
      if (textures.size > 128) for (const [key, item] of [...textures].sort((a, b) => a[1].used - b[1].used)) {
        if (textures.size <= 128) break;
        if (!visible.has(key)) { item.image.onload = null; item.image.onerror = null; gl!.deleteTexture(item.texture); textures.delete(key); }
      }
      element.dataset.cachedTextures = String(textures.size);
      if (base.loaded) { element.dataset.backend = 'webgl'; latest.current.ready(true); }
    }
    const lost = (event: Event) => { event.preventDefault(); fallback(); };
    element.addEventListener('webglcontextlost', lost);
    const resize = new ResizeObserver(request); resize.observe(element);
    schedule.current = request; request();
    return () => {
      disposed = true; cancelAnimationFrame(frame); schedule.current = () => {}; resize.disconnect();
      element.removeEventListener('webglcontextlost', lost);
      for (const item of textures.values()) { item.image.onload = null; item.image.onerror = null; gl.deleteTexture(item.texture); }
      gl.deleteBuffer(buffer); gl.deleteProgram(program); gl.deleteShader(vertex); gl.deleteShader(fragment);
      gl.getExtension('WEBGL_lose_context')?.loseContext(); latest.current.ready(false);
    };
  }, [props.scene]);
  useEffect(() => schedule.current(), [props.transform, props.tiles, props.width, props.height]);
  return <canvas ref={canvas} className="scene-gpu" aria-hidden="true" data-backend="initializing" />;
}
