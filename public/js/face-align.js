// InsightFace face alignment math — pure JS, no ORT dependency
// Ports: face_align.py, scrfd.py distance decoding, NMS

// ── ArcFace 5-point reference template (at 112×112) ─────────────────────────

const ARCFACE_DST_112 = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

export function getArcFaceTemplate(size = 112) {
    const s = size / 112.0;
    return ARCFACE_DST_112.map(([x, y]) => [x * s, y * s]);
}

// ── Least-squares similarity transform ──────────────────────────────────────
// Finds 2×3 matrix M where M * [x,y,1]^T ≈ [x',y']^T
// Uses the linear parameterisation: x' = a*x - b*y + tx, y' = b*x + a*y + ty

export function estimateSimilarityTransform(srcPts, dstPts) {
    const n = srcPts.length;
    // Build 2n×4 system: [x, -y, 1, 0; y, x, 0, 1] * [a,b,tx,ty]^T = [x',y']
    const A = [];
    const bv = [];
    for (let i = 0; i < n; i++) {
        const [x, y] = srcPts[i];
        const [xp, yp] = dstPts[i];
        A.push([x, -y, 1, 0]);
        A.push([y,  x, 0, 1]);
        bv.push(xp);
        bv.push(yp);
    }
    const [a, b, tx, ty] = _leastSquares4(A, bv);
    return [[a, -b, tx], [b, a, ty]];
}

function _leastSquares4(A, bv) {
    // Normal equations: (A^T A) x = A^T b
    const AtA = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    const Atb = [0,0,0,0];
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < 4; j++) {
            for (let k = 0; k < 4; k++) AtA[j][k] += A[i][j] * A[i][k];
            Atb[j] += A[i][j] * bv[i];
        }
    }
    return _gaussElim4(AtA, Atb);
}

function _gaussElim4(A, b) {
    const M = A.map((row, i) => [...row, b[i]]);
    const n = 4;
    for (let col = 0; col < n; col++) {
        let maxR = col;
        for (let r = col + 1; r < n; r++) {
            if (Math.abs(M[r][col]) > Math.abs(M[maxR][col])) maxR = r;
        }
        [M[col], M[maxR]] = [M[maxR], M[col]];
        if (Math.abs(M[col][col]) < 1e-12) continue;
        for (let r = 0; r < n; r++) {
            if (r === col) continue;
            const f = M[r][col] / M[col][col];
            for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
        }
    }
    return M.map((row, i) => row[n] / row[i]);
}

export function invertAffine2x3(M) {
    const [[a, b, tx], [c, d, ty]] = M;
    const det = a * d - b * c;
    return [
        [ d / det, -b / det,  (b * ty - d * tx) / det],
        [-c / det,  a / det,  (c * tx - a * ty) / det],
    ];
}

// ── Aligned face crop (equivalent to cv2.warpAffine) ─────────────────────────
// Returns { canvas: OffscreenCanvas, M: [[a,b,tx],[c,d,ty]] }

export function normCrop(srcImageBitmap, kps, size = 256) {
    const dst = getArcFaceTemplate(size);
    const M = estimateSimilarityTransform(kps, dst);
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext("2d");
    // canvas.setTransform(a,b,c,d,e,f) matrix layout:
    //   [a c e]   so for M = [[m00,m01,m02],[m10,m11,m12]]:
    //   [b d f]   a=m00, b=m10, c=m01, d=m11, e=m02, f=m12
    ctx.setTransform(M[0][0], M[1][0], M[0][1], M[1][1], M[0][2], M[1][2]);
    ctx.drawImage(srcImageBitmap, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return { canvas, M };
}

// ── Paste swapped face back into original frame ───────────────────────────────
// Uses inverse affine + soft ellipse mask for edge blending

export async function pasteBack(swappedBitmap, M, origBitmap, frameW, frameH) {
    const size = swappedBitmap.width;
    const Minv = invertAffine2x3(M);

    // Soft ellipse mask over the swapped 256×256 face
    const maskCanvas = new OffscreenCanvas(size, size);
    const mctx = maskCanvas.getContext("2d");
    mctx.filter = "blur(10px)";
    mctx.fillStyle = "white";
    mctx.beginPath();
    mctx.ellipse(size / 2, size / 2, size / 2 - 10, size / 2 - 10, 0, 0, Math.PI * 2);
    mctx.fill();

    // Apply mask to swapped face
    const tmpCanvas = new OffscreenCanvas(size, size);
    const tctx = tmpCanvas.getContext("2d");
    tctx.drawImage(swappedBitmap, 0, 0);
    tctx.globalCompositeOperation = "destination-in";
    const maskBmp = await createImageBitmap(maskCanvas);
    tctx.drawImage(maskBmp, 0, 0);
    maskBmp.close();

    // Composite: original frame + inverse-warped masked swapped face
    const outCanvas = new OffscreenCanvas(frameW, frameH);
    const ctx = outCanvas.getContext("2d");
    ctx.drawImage(origBitmap, 0, 0);
    ctx.setTransform(Minv[0][0], Minv[1][0], Minv[0][1], Minv[1][1], Minv[0][2], Minv[1][2]);
    const maskedBmp = await createImageBitmap(tmpCanvas);
    ctx.drawImage(maskedBmp, 0, 0);
    maskedBmp.close();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    return outCanvas;
}

// ── SCRFD decoder ─────────────────────────────────────────────────────────────
// det_10g.onnx: 9 outputs, strides [8,16,32], 2 anchors/loc, use_kps=true

const SCRFD_STRIDES    = [8, 16, 32];
const SCRFD_N_ANCHORS  = 2;
const SCRFD_FMC        = 3;         // number of strides (feature map count)

export function decodeSCRFD(outputs, inputW, inputH, origW, origH,
                             confThresh = 0.5, iouThresh = 0.4) {
    const scaleX = origW / inputW;
    const scaleY = origH / inputH;
    const allBoxes = [], allScores = [], allKps = [];

    for (let si = 0; si < SCRFD_STRIDES.length; si++) {
        const stride = SCRFD_STRIDES[si];

        // outputs order: [score_s8, score_s16, score_s32,
        //                 bbox_s8,  bbox_s16,  bbox_s32,
        //                 kps_s8,   kps_s16,   kps_s32]
        let scores = _flattenOutput(outputs[si]);
        let bboxes = _flattenOutput(outputs[si + SCRFD_FMC]);
        let kpss   = _flattenOutput(outputs[si + SCRFD_FMC * 2]);

        const fH = Math.floor(inputH / stride);
        const fW = Math.floor(inputW / stride);
        const anchors = _genAnchors(fH, fW, stride, SCRFD_N_ANCHORS);
        const N = anchors.length;   // fH*fW*N_ANCHORS

        for (let i = 0; i < N; i++) {
            const score = scores[i];
            if (score < confThresh) continue;

            const [cx, cy] = anchors[i];
            const d0 = bboxes[i * 4 + 0] * stride;
            const d1 = bboxes[i * 4 + 1] * stride;
            const d2 = bboxes[i * 4 + 2] * stride;
            const d3 = bboxes[i * 4 + 3] * stride;
            allBoxes.push([
                (cx - d0) * scaleX,
                (cy - d1) * scaleY,
                (cx + d2) * scaleX,
                (cy + d3) * scaleY,
            ]);
            allScores.push(score);

            const faceKps = [];
            for (let k = 0; k < 5; k++) {
                faceKps.push([
                    (cx + kpss[i * 10 + k * 2]     * stride) * scaleX,
                    (cy + kpss[i * 10 + k * 2 + 1] * stride) * scaleY,
                ]);
            }
            allKps.push(faceKps);
        }
    }

    if (allBoxes.length === 0) return [];
    const keep = _nms(allBoxes, allScores, iouThresh);
    return keep.map(i => ({ bbox: allBoxes[i], score: allScores[i], kps: allKps[i] }))
               .sort((a, b) => _boxArea(b.bbox) - _boxArea(a.bbox));
}

function _flattenOutput(tensor) {
    // ort tensor → plain Float32Array regardless of batch dim
    if (tensor.dims.length === 3) return tensor.data;   // [1, N, C] → already flat
    if (tensor.dims.length === 2) return tensor.data;   // [N, C]
    return tensor.data;
}

function _genAnchors(fH, fW, stride, nAnchors) {
    const centers = [];
    for (let y = 0; y < fH; y++) {
        for (let x = 0; x < fW; x++) {
            for (let a = 0; a < nAnchors; a++) {
                centers.push([x * stride, y * stride]);
            }
        }
    }
    return centers;
}

function _nms(boxes, scores, iouThresh) {
    const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).map(x => x[1]);
    const keep = [];
    const suppressed = new Uint8Array(boxes.length);
    for (const i of order) {
        if (suppressed[i]) continue;
        keep.push(i);
        for (const j of order) {
            if (suppressed[j] || j === i) continue;
            if (_iou(boxes[i], boxes[j]) > iouThresh) suppressed[j] = 1;
        }
    }
    return keep;
}

function _iou(a, b) {
    const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
    const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    return inter / (_boxArea(a) + _boxArea(b) - inter + 1e-6);
}

function _boxArea(b) {
    return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
}
