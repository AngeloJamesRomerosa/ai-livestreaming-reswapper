// Client-side ONNX inference via onnxruntime-web
// Runs SCRFD face detection + reswapper_256 face swap entirely in the browser
import { decodeSCRFD, normCrop, pasteBack } from './face-align.js';

const DET_SIZE  = 640;
const SWAP_SIZE = 256;
const DET_MEAN  = 127.5;
const DET_STD   = 128.0;
const ORT_CDN   = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
const EP_LIST   = ['webgpu', 'webgl', 'wasm'];

async function _waitOrt(ms = 10000) {
    const t = Date.now();
    while (!window.ort) {
        if (Date.now() - t > ms) throw new Error('onnxruntime-web not loaded (CDN timeout)');
        await new Promise(r => setTimeout(r, 50));
    }
}

async function _fetchBuf(url, onProg) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
    const total = parseInt(res.headers.get('content-length') || '0', 10);
    if (!res.body || total === 0) {
        onProg?.(1);
        return res.arrayBuffer();
    }
    const reader = res.body.getReader();
    const parts  = [];
    let got = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        got += value.length;
        onProg?.(got / total);
    }
    const out = new Uint8Array(got);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out.buffer;
}

async function _bestSwapUrl() {
    try {
        const r = await fetch('/models/reswapper_256_fp16.onnx', { method: 'HEAD' });
        if (r.ok) return { swapUrl: '/models/reswapper_256_fp16.onnx', swapLabel: 'swap model FP16 (~264 MB)' };
    } catch (_) {}
    return { swapUrl: '/models/reswapper_256.onnx', swapLabel: 'swap model FP32 (~529 MB)' };
}

async function _newSession(buf, epList) {
    const ort = window.ort;
    ort.env.wasm.wasmPaths = ORT_CDN;
    for (const ep of epList) {
        try {
            const sess = await ort.InferenceSession.create(buf, {
                executionProviders:     [ep],
                graphOptimizationLevel: 'all',
                enableMemPattern:       true,
            });
            return { sess, ep };
        } catch (e) {
            console.warn(`[ORT] EP "${ep}" failed:`, e.message ?? e);
        }
    }
    throw new Error('No ONNX execution provider available — update your browser or GPU driver');
}

export class InferenceEngine {
    constructor() {
        this._det     = null;
        this._swap    = null;
        this.detEp    = 'none';
        this.swapEp   = 'none';
        this._latent  = null;   // Float32Array [512]
        this._tick    = 0;
        this._kps     = null;   // cached 5-point keypoints
        this.detSkip  = 4;      // re-detect every N frames
        this.ready    = false;
    }

    async loadModels(onProgress) {
        await _waitOrt();

        const cb = (stage, label, p) => onProgress?.({ stage, label, progress: p });

        cb('det', 'Downloading face detector…', 0);
        const detBuf = await _fetchBuf('/models/det_10g.onnx',
            p => cb('det', 'Downloading face detector…', p));

        cb('det', 'Initializing face detector…', 1);
        const { sess: det, ep: de } = await _newSession(detBuf, EP_LIST);
        this._det  = det;
        this.detEp = de;

        const { swapUrl, swapLabel } = await _bestSwapUrl();
        cb('swap', `Downloading ${swapLabel}…`, 0);
        const swapBuf = await _fetchBuf(swapUrl,
            p => cb('swap', `Downloading ${swapLabel}…`, p));

        cb('swap', 'Initializing swap model…', 1);
        const { sess: swap, ep: se } = await _newSession(swapBuf, EP_LIST);
        this._swap  = swap;
        this.swapEp = se;

        this.ready = true;
    }

    setSourceLatent(arr) {
        const flat = Array.isArray(arr[0]) ? arr.flat() : arr;
        this._latent = new Float32Array(flat);
    }

    resetCache() { this._tick = 0; this._kps = null; }

    // Returns OffscreenCanvas with swapped face composited into original frame,
    // or null if no face detected.
    async runFrame(bmp, w, h) {
        if (!this.ready || !this._latent) return null;
        this._tick++;
        if (this._tick % this.detSkip === 1 || !this._kps) {
            this._kps = await this._detect(bmp, w, h);
        }
        if (!this._kps) return null;
        return this._swapFace(bmp, this._kps, w, h);
    }

    async _detect(bmp, origW, origH) {
        const ort = window.ort;
        // Stretch frame to DET_SIZE×DET_SIZE, normalize (pixel - 127.5) / 128 → CHW RGB
        const off  = new OffscreenCanvas(DET_SIZE, DET_SIZE);
        const octx = off.getContext('2d', { willReadFrequently: true });
        octx.drawImage(bmp, 0, 0, DET_SIZE, DET_SIZE);
        const { data } = octx.getImageData(0, 0, DET_SIZE, DET_SIZE);

        const N = DET_SIZE * DET_SIZE;
        const t = new Float32Array(3 * N);
        for (let i = 0; i < N; i++) {
            t[i]       = (data[i*4]   - DET_MEAN) / DET_STD;
            t[N+i]     = (data[i*4+1] - DET_MEAN) / DET_STD;
            t[N*2+i]   = (data[i*4+2] - DET_MEAN) / DET_STD;
        }

        const feeds  = { [this._det.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, DET_SIZE, DET_SIZE]) };
        const result = await this._det.run(feeds);
        // Pass outputs in model-defined order so decodeSCRFD index offsets are correct
        const outs   = this._det.outputNames.map(n => result[n]);
        const faces  = decodeSCRFD(outs, DET_SIZE, DET_SIZE, origW, origH, 0.5, 0.4);
        return faces.length ? faces[0].kps : null;
    }

    async _swapFace(bmp, kps, frameW, frameH) {
        const ort = window.ort;
        // Align and crop to 256×256 face-aligned patch
        const { canvas: crop, M } = normCrop(bmp, kps, SWAP_SIZE);
        const { data } = crop.getContext('2d').getImageData(0, 0, SWAP_SIZE, SWAP_SIZE);

        // Preprocess: pixel / 255, CHW RGB  (INSwapper: input_mean=0, input_std=255)
        const N = SWAP_SIZE * SWAP_SIZE;
        const tgt = new Float32Array(3 * N);
        for (let i = 0; i < N; i++) {
            tgt[i]     = data[i*4]   / 255;
            tgt[N+i]   = data[i*4+1] / 255;
            tgt[N*2+i] = data[i*4+2] / 255;
        }

        const [tn, ln] = this._swap.inputNames;
        const feeds = {
            [tn]: new ort.Tensor('float32', tgt,          [1, 3, SWAP_SIZE, SWAP_SIZE]),
            [ln]: new ort.Tensor('float32', this._latent, [1, 512]),
        };
        const res = await this._swap.run(feeds);
        const out = res[this._swap.outputNames[0]].data;

        // Post-process: [0,1] → [0,255] RGB → ImageData
        const img = new ImageData(SWAP_SIZE, SWAP_SIZE);
        const px  = img.data;
        for (let i = 0; i < N; i++) {
            px[i*4]   = Math.max(0, Math.min(255, out[i]     * 255)) | 0;
            px[i*4+1] = Math.max(0, Math.min(255, out[N+i]   * 255)) | 0;
            px[i*4+2] = Math.max(0, Math.min(255, out[N*2+i] * 255)) | 0;
            px[i*4+3] = 255;
        }

        const swapBmp = await createImageBitmap(img);
        const canvas  = await pasteBack(swapBmp, M, bmp, frameW, frameH);
        swapBmp.close();
        return canvas;
    }
}
