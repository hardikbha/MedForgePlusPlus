"""Per-image metadata + global statistics for the shortcut audit. Read-only on data; CPU only."""
import json, os, sys, hashlib, io, warnings
import numpy as np, cv2, scipy.fft as sfft
from PIL import Image, JpegImagePlugin
from multiprocessing import Pool
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS = None
STD_L = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
                  18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], float)
STD_C = np.array([17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99]+[99]*32, float)
def ijg(std, q):
    s = 5000.0 / q if q < 50 else 200.0 - 2 * q
    t = np.floor((std * s + 50) / 100); t[t < 1] = 1; t[t > 255] = 255; return t
TAB = {q: (np.sort(ijg(STD_L, q)), np.sort(ijg(STD_C, q))) for q in range(1, 101)}
def est_q(table, chroma=False):
    a = np.sort(np.asarray(table, float)); best = (1e18, None)
    for q in range(1, 101):
        d = np.abs(a - TAB[q][1 if chroma else 0]).sum()
        if d < best[0]: best = (d, q)
    return best[1], best[0]   # quality, L1 residual (0 = exact IJG/libjpeg table)

def radial_ratios(g):
    h, w = g.shape
    if max(h, w) > 2048:  # centre crop to bound cost
        y0 = max(0, (h - 2048) // 2); x0 = max(0, (w - 2048) // 2); g = g[y0:y0 + 2048, x0:x0 + 2048]; h, w = g.shape
    g = g - g.mean()
    P = np.abs(sfft.rfft2(g, workers=1)) ** 2
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.rfftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5
    tot = P[r > 0.02].sum() + 1e-12
    return float(P[r > 0.5].sum() / tot), float(P[(r > 0.25) & (r <= 0.5)].sum() / tot), float(P[(r > 0.02) & (r <= 0.1)].sum() / tot)

def blockiness(g):
    dh = np.abs(np.diff(g, axis=1)); dv = np.abs(np.diff(g, axis=0))
    if dh.shape[1] < 16 or dv.shape[0] < 16: return np.nan
    ih = np.arange(dh.shape[1]); iv = np.arange(dv.shape[0])
    bh = dh[:, (ih % 8) == 7].mean(); nh = dh[:, (ih % 8) != 7].mean()
    bv = dv[(iv % 8) == 7, :].mean(); nv = dv[(iv % 8) != 7, :].mean()
    return float((bh + bv) / (nh + nv + 1e-9))

def gstats(rgb, prefix):
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    o = {}
    o[prefix + 'mean'] = float(g.mean()); o[prefix + 'std'] = float(g.std())
    o[prefix + 'lapvar'] = float(cv2.Laplacian(g, cv2.CV_32F).var())
    hf, mid, low = radial_ratios(g)
    o[prefix + 'fft_hf'] = hf; o[prefix + 'fft_mid'] = mid; o[prefix + 'fft_low'] = low
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    o[prefix + 'sat'] = float(hsv[..., 1].mean() / 255.0)
    rf = rgb.astype(np.int16)
    o[prefix + 'chroma'] = float((np.abs(rf[..., 0] - rf[..., 1]) + np.abs(rf[..., 1] - rf[..., 2])).mean())
    gu = np.clip(g, 0, 255).astype(np.uint8)
    o[prefix + 'noise'] = float((g - cv2.medianBlur(gu, 3).astype(np.float32)).std())
    hist = np.bincount(gu.ravel(), minlength=256).astype(float); p = hist / hist.sum()
    o[prefix + 'entropy'] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    o[prefix + 'dark_frac'] = float(p[:6].sum()); o[prefix + 'bright_frac'] = float(p[250:].sum())
    o[prefix + 'zero_frac'] = float(p[0]); o[prefix + 'max255_frac'] = float(p[255])
    o[prefix + 'nlevels'] = int((hist > 0).sum())
    return o, g

def work(rec):
    p = rec['path']; o = {'path': p}
    try:
        b = open(p, 'rb').read()
        o['file_size'] = len(b); o['md5'] = hashlib.md5(b).hexdigest()
        o['ext'] = os.path.splitext(p)[1].lower()
        im = Image.open(io.BytesIO(b))
        o['format'] = im.format; o['mode'] = im.mode; o['width'], o['height'] = im.size
        info = im.info
        o['has_exif'] = int(bool(info.get('exif')) or len(im.getexif()) > 0)
        o['has_icc'] = int('icc_profile' in info and bool(info['icc_profile']))
        o['has_jfif'] = int('jfif' in info); o['has_adobe'] = int('adobe' in info)
        o['progressive'] = int(bool(info.get('progressive') or info.get('progression')))
        o['has_comment'] = int('comment' in info)
        dpi = info.get('dpi'); o['dpi_x'] = float(dpi[0]) if dpi else np.nan
        o['info_keys'] = ','.join(sorted(k for k in info.keys() if k not in ('icc_profile', 'exif')))
        if im.format == 'JPEG':
            qt = im.quantization or {}
            o['n_qtables'] = len(qt)
            if 0 in qt: o['jpeg_q'], o['jpeg_q_resid'] = est_q(qt[0])
            if 1 in qt: o['jpeg_qc'], o['jpeg_qc_resid'] = est_q(qt[1], True)
            o['subsampling'] = int(JpegImagePlugin.get_sampling(im))
            o['qt_hash'] = hashlib.md5(json.dumps({k: list(v) for k, v in qt.items()}).encode()).hexdigest()[:10]
        else:
            o['n_qtables'] = 0; o['subsampling'] = -2
        o['png_bitdepth16'] = int(im.mode in ('I;16', 'I', 'I;16B'))
        if im.mode in ('I;16', 'I', 'I;16B', 'F'):
            a = np.asarray(im).astype(np.float32); a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255
            rgb = np.stack([a] * 3, -1).astype(np.uint8)
        else:
            rgb = np.asarray(im.convert('RGB'))
        so, g = gstats(rgb, 'n_')            # native resolution
        o.update(so); o['n_block'] = blockiness(g)
        r512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        so2, _ = gstats(r512, 'r_')          # resolution-normalised (512x512)
        o.update(so2)
        th = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA)
        o['thumb'] = [round(float(v), 2) for v in th.ravel()]
        o['ok'] = 1
    except Exception as e:
        o['ok'] = 0; o['err'] = repr(e)[:200]
    return o

if __name__ == '__main__':
    src, dst, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    recs = [json.loads(l) for l in open(src)]
    if len(sys.argv) > 4: recs = recs[:int(sys.argv[4])]
    with Pool(nproc) as pool, open(dst, 'w') as f:
        for i, o in enumerate(pool.imap_unordered(work, recs, chunksize=16)):
            f.write(json.dumps(o) + '\n')
            if i % 2000 == 0: print(i, flush=True)
    print('done', len(recs))
