import type { Job } from './offline-api';

type Selection = { job?: Job; cancelled?: boolean; error?: string };
type Bridge = { chooseScene: (copy: boolean, callback: (value: Selection) => void) => void };
declare global {
  interface Window {
    qt?: { webChannelTransport: unknown };
    QWebChannel?: new (transport: unknown, callback: (channel: { objects: { atlasFiles: Bridge } }) => void) => unknown;
    __atlasNativeReady?: boolean;
  }
}
const bridge = new Promise<Bridge | null>((resolve) => {
  if (!window.qt || !window.QWebChannel) { resolve(null); return; }
  new window.QWebChannel(window.qt.webChannelTransport, channel => {
    window.__atlasNativeReady = true;
    resolve(channel.objects.atlasFiles);
  });
});
export async function chooseScene(copy: boolean): Promise<Selection | null> {
  const native = await bridge;
  if (!native) return null;
  return new Promise(resolve => native.chooseScene(copy, resolve));
}
