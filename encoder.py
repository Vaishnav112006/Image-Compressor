import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from math import sqrt, log10
import heapq
import pickle

QY = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], dtype=np.float32)

QC = np.array([
 [17,18,24,47,99,99,99,99],
 [18,21,26,66,99,99,99,99],
 [24,26,56,99,99,99,99,99],
 [47,66,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99]
], dtype=np.float32)

def scale_quant_matrix(Q, quality):
    if quality < 1: quality = 1
    if quality > 100: quality = 100
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2*quality
    Qs = ((Q * scale) + 50) // 100
    Qs[Qs == 0] = 1
    return Qs.astype(np.float32)

def block_process(channel, block_size, func):
    h, w = channel.shape
    h_pad = (block_size - (h % block_size)) % block_size
    w_pad = (block_size - (w % block_size)) % block_size
    padded = np.pad(channel, ((0,h_pad),(0,w_pad)), mode='edge')
    out = np.zeros_like(padded)
    for i in range(0, padded.shape[0], block_size):
        for j in range(0, padded.shape[1], block_size):
            block = padded[i:i+block_size, j:j+block_size]
            out[i:i+block_size, j:j+block_size] = func(block)
    return out[:h, :w]

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))

def zigzag_order(block):
    idxs = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
    ]
    return [block[i,j] for (i,j) in idxs]

def rle_encode(arr):
    res = []
    if len(arr) == 0:
        return res
    prev = arr[0]
    cnt = 1
    for x in arr[1:]:
        if x == prev:
            cnt += 1
        else:
            res.append((int(prev), cnt))
            prev = x
            cnt = 1
    res.append((int(prev), cnt))
    return res

def build_huffman(freqs):
    heap = []
    for sym, f in freqs.items():
        heapq.heappush(heap, (f, sym))
    if len(heap) == 0:
        return {}
    # build simple codebook via heap merging (not tree nodes to keep simple)
    # This returns a sorted list of symbols by freq and assign binary codes (demo only)
    heap.sort()
    codes = {}
    for i, (_, sym) in enumerate(heap):
        codes[sym] = format(i, 'b')  # simple unique code - not optimal but deterministic
    return codes

def encode_coeffs_to_bin(all_coeffs, out_path):
    flat_blocks = []
    for ch in all_coeffs:
        for block in ch:
            zz = zigzag_order(block)
            zz_int = [int(round(x)) for x in zz]
            rle = rle_encode(zz_int)
            flat_blocks.append(rle)
    freqs = {}
    for rle in flat_blocks:
        for pair in rle:
            freqs[pair] = freqs.get(pair, 0) + 1
    codebook = build_huffman(freqs)
    # encode as list of (codebook, encoded_blocks) using pickle
    encoded_blocks = []
    for rle in flat_blocks:
        codes = [codebook[pair] for pair in rle]
        encoded_blocks.append(codes)
    with open(out_path, "wb") as f:
        pickle.dump({"codebook": codebook, "blocks": encoded_blocks}, f)

def compress_image_rgb(img_rgb, quality=50, subsample=False):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    if subsample:
        img_ycrcb[:, :, 1] = cv2.resize(img_ycrcb[:, :, 1], (img_ycrcb.shape[1]//2, img_ycrcb.shape[0]//2), interpolation=cv2.INTER_AREA).repeat(2, axis=0).repeat(2, axis=1)[:img_ycrcb.shape[0], :img_ycrcb.shape[1]]
        img_ycrcb[:, :, 2] = cv2.resize(img_ycrcb[:, :, 2], (img_ycrcb.shape[1]//2, img_ycrcb.shape[0]//2), interpolation=cv2.INTER_AREA).repeat(2, axis=0).repeat(2, axis=1)[:img_ycrcb.shape[0], :img_ycrcb.shape[1]]
    channels = cv2.split(img_ycrcb)
    QY_s = scale_quant_matrix(QY, quality)
    QC_s = scale_quant_matrix(QC, quality)
    out_channels = []
    all_coeffs = []
    for idx, ch in enumerate(channels):
        ch_shifted = ch - 128.0
        Q = QY_s if idx == 0 else QC_s
        blocks = []
        def proc(block):
            B = dct2(block)
            Bq = np.round(B / Q)
            blocks.append(Bq.copy())
            Bd = Bq * Q
            rec = idct2(Bd)
            return rec
        rec = block_process(ch_shifted, 8, proc)
        rec = rec + 128.0
        rec = np.clip(rec, 0, 255)
        out_channels.append(rec)
        all_coeffs.append(blocks)
    ycrcb_rec = cv2.merge(out_channels).astype(np.uint8)
    bgr_rec = cv2.cvtColor(ycrcb_rec, cv2.COLOR_YCrCb2BGR)
    rgb_rec = cv2.cvtColor(bgr_rec, cv2.COLOR_BGR2RGB)
    return rgb_rec, all_coeffs

def compute_psnr(orig, rec):
    orig = orig.astype(float)
    rec = rec.astype(float)
    mse = np.mean((orig - rec)**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def compute_ssim_safe(orig, rec):
    try:
        return ssim(orig, rec, channel_axis=2, data_range=255)
    except TypeError:
        return ssim(orig, rec, multichannel=True, data_range=255)
    except Exception:
        return -1
