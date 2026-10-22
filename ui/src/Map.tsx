import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Crosshair, Layers, LocateFixed, MapPin, Minus, Plus, WifiOff } from 'lucide-react';
import { Building, fa } from './api';

type Props = { center: [number, number]; radius: number; buildings: Building[]; selected: string | null; onCenter: (position: [number, number]) => void; onSelect: (id: string) => void };

export default function ScoutMap({ center, radius, buildings, selected, onCenter, onSelect }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const area = useRef<L.Circle | null>(null);
  const markers = useRef<L.LayerGroup | null>(null);
  const layer = useRef<L.TileLayer | null>(null);
  const [aerial, setAerial] = useState(false);
  const [offline, setOffline] = useState(false);
  const onCenterRef = useRef(onCenter);
  onCenterRef.current = onCenter;

  useEffect(() => {
    if (!element.current) return;
    const map = L.map(element.current, { center, zoom: 16, zoomControl: false, attributionControl: true, preferCanvas: true });
    mapRef.current = map;
    area.current = L.circle(center, { radius, color: '#b5d7a5', weight: 1.5, dashArray: '6 7', fillColor: '#b5d7a5', fillOpacity: 0.08, interactive: false }).addTo(map);
    markers.current = L.layerGroup().addTo(map);
    map.on('click', event => onCenterRef.current([event.latlng.lat, event.latlng.lng]));
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(element.current);
    return () => { observer.disconnect(); map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    layer.current?.remove();
    setOffline(false);
    const tiles = aerial ? L.tileLayer('/api/map/aerial/{z}/{x}/{y}', {
      attribution: 'Luchtfoto RGB · <a href="https://www.pdok.nl/" target="_blank" rel="noreferrer">PDOK</a>', maxZoom: 20,
    }) : L.tileLayer('/api/map/street/{z}/{x}/{y}', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a>', maxZoom: 19,
    });
    let errors = 0;
    tiles.on('tileerror', () => { errors++; if (errors >= 3) setOffline(true); });
    tiles.on('tileload', () => { setOffline(false); });
    tiles.addTo(map);
    layer.current = tiles;
    return () => { tiles.remove(); tiles.off(); if (layer.current === tiles) layer.current = null; };
  }, [aerial]);

  useEffect(() => { area.current?.setLatLng(center); mapRef.current?.panTo(center, { animate: true }); }, [center]);
  useEffect(() => { area.current?.setRadius(radius); }, [radius]);

  useEffect(() => {
    const group = markers.current;
    if (!group) return;
    group.clearLayers();
    buildings.forEach((building, index) => {
      const active = selected === building.id;
      if (building.geometry) L.geoJSON(building.geometry, { style: { color: active ? '#dcf5bd' : '#a2bc96', weight: active ? 3 : 1, fillOpacity: active ? 0.24 : 0.06 } }).addTo(group);
      const icon = L.divIcon({ className: '', html: `<span class="map-marker ${active ? 'active' : ''}">${fa(index + 1)}</span>`, iconSize: [30, 36], iconAnchor: [15, 36] });
      L.marker([building.lat, building.lon], { icon }).on('click', event => { L.DomEvent.stopPropagation(event); onSelect(building.id); }).addTo(group);
    });
  }, [buildings, selected, onSelect]);

  useEffect(() => {
    const building = buildings.find(item => item.id === selected);
    if (building) mapRef.current?.flyTo([building.lat, building.lon], 18, { duration: 0.6 });
  }, [selected, buildings]);

  const recenter = () => mapRef.current?.flyTo(center, 16, { duration: 0.6 });
  return <section className={`map-panel ${aerial ? 'aerial' : ''}`} aria-label="نقشهٔ محدودهٔ جست‌وجو">
    <div ref={element} className="map-canvas" />
    <div className="map-topline"><span className="map-location"><MapPin size={14} /> نقشه <span>/</span> محدودهٔ انتخابی</span><span className="map-instruction">با کلیک روی نقشه، مرکز محدوده را تغییر دهید</span></div>
    <div className="map-tools"><button title="بزرگ‌نمایی" aria-label="بزرگ‌نمایی" onClick={() => mapRef.current?.zoomIn()}><Plus size={19} /></button><button title="کوچک‌نمایی" aria-label="کوچک‌نمایی" onClick={() => mapRef.current?.zoomOut()}><Minus size={19} /></button><i /><button title="بازگشت به محدوده" aria-label="بازگشت به محدوده" onClick={recenter}><LocateFixed size={19} /></button></div>
    <button className={`layer-button ${aerial ? 'selected' : ''}`} onClick={() => setAerial(!aerial)}><Layers size={17} />{aerial ? 'تصویر هوایی' : 'نقشهٔ خیابان'}</button>
    <div className="coordinate-label" dir="ltr"><Crosshair size={12} /><span>{center[0].toFixed(4)}° N · {center[1].toFixed(4)}° E</span></div>
    {offline && <div className="map-offline"><WifiOff size={16} /><span>نقشهٔ آنلاین در دسترس نیست. کتابخانهٔ محلی همچنان قابل استفاده است.</span></div>}
  </section>;
}
