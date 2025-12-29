import cv2
import numpy as np
import pandas as pd

# ============================================================
# Colony extraction (96-well) - v3.1 (draw-shape fallback added)
# - ROI: use cropped image as-is
# - Grid: UNIFORM 12x8 + SHIFT + last-col fix
# - Per-cell:
#   * well mask + inner mask (rim excluded)
#   * illumination flatten (big sigma)
#   * rim band 제거 후 percentile threshold
#   * light morphology (open/close)
#   * CC select seeded by dense point
#   * shape: fitEllipse (fallback to minEnclosingCircle)
#   * dense point: spots -> if blob exists, texture-based dense in blob
# - IMPORTANT FIX:
#   * blob이 실패하더라도 "원은 무조건" 그리도록 fallback(=well circle) 추가
#     -> A3, C12처럼 원이 아예 안 그려지는 케이스 방지
# - Output: grid_debug.png, colony_result.png, colony_info.csv
# ============================================================

IMG_PATH = r"C:\Users\USER\Desktop\ablelabs\input2.png"

# --------------------
# 1) Load
# --------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image load failed: {IMG_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape[:2]

# ✅ crop 이미지를 ROI로 그대로 사용
roi = gray
roi_color = img
roiH, roiW = roi.shape[:2]
x1, y1 = 0, 0  # crop 이미지면 0

# --------------------
# 2) (optional) grid center candidates - kept for reference (not used for bounds)
# --------------------
th = cv2.adaptiveThreshold(
    roi, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51, 5
)
th = cv2.morphologyEx(
    th,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    iterations=1
)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th)

cands = []
for i in range(1, num_labels):
    area = int(stats[i, cv2.CC_STAT_AREA])
    xx, yy, ww, hh, _ = stats[i]
    if area < 80:
        continue
    if ww == 0 or hh == 0:
        continue
    ar = max(ww / hh, hh / ww)
    if ar > 2.8:
        continue
    cx, cy = centroids[i]
    cands.append((float(cx), float(cy)))

if len(cands) < 30:
    raise RuntimeError(f"Too few candidates for grid estimation: {len(cands)}. Tune threshold/morph.")

xs = np.array([c[0] for c in cands], dtype=np.float32)
ys = np.array([c[1] for c in cands], dtype=np.float32)

def kmeans_1d(values, k, iters=40):
    v = np.asarray(values, dtype=np.float32)
    qs = np.linspace(0.05, 0.95, k)
    centers = np.quantile(v, qs).astype(np.float32)
    for _ in range(iters):
        d = np.abs(v[:, None] - centers[None, :])
        idx = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            sel = v[idx == j]
            if len(sel) > 0:
                new_centers[j] = sel.mean()
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    centers.sort()
    return centers

x_centers = kmeans_1d(xs, k=12)
y_centers = kmeans_1d(ys, k=8)

# --------------------
# 3) Grid cell bounds: uniform + shift + last-col fix
# --------------------
cell_w = roiW / 12.0
cell_h = roiH / 8.0

GRID_SHIFT_X = -14
GRID_SHIFT_Y = 5

def cell_bounds(r, c):
    left  = int(round(c * cell_w + GRID_SHIFT_X))
    right = int(round((c + 1) * cell_w + GRID_SHIFT_X))
    top   = int(round(r * cell_h + GRID_SHIFT_Y))
    bot   = int(round((r + 1) * cell_h + GRID_SHIFT_Y))

    if c == 11:
        right = roiW
    if r == 0:
        top = 0

    left = max(0, min(left, roiW - 1))
    right = max(left + 1, min(right, roiW))
    top = max(0, min(top, roiH - 1))
    bot = max(top + 1, min(bot, roiH))
    return left, top, right, bot

def draw_grid_overlay(img_bgr, cell_bounds_func, rows=8, cols=12,
                      color=(0, 255, 255), thickness=1, with_label=True):
    rows_name = "ABCDEFGH"
    for r in range(rows):
        for c in range(cols):
            L, T, R, B = cell_bounds_func(r, c)
            cv2.rectangle(img_bgr, (L, T), (R - 1, B - 1), color, thickness)
            if with_label:
                well = f"{rows_name[r]}{c+1}"
                cv2.putText(img_bgr, well, (L + 2, T + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1]-1, img_bgr.shape[0]-1), color, 1)
    return img_bgr

# --------------------
# Dense point from spots (center exclusion)
# --------------------
def dense_point_from_spots(cell_gray, mask):
    h, w = cell_gray.shape[:2]
    cx0, cy0 = w // 2, h // 2

    final_mask = mask.copy()
    cv2.circle(final_mask, (cx0, cy0), int(0.12 * min(w, h)), 0, -1)
    m_final = (final_mask > 0).astype(np.uint8)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    eq = clahe.apply(cell_gray)

    bh = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    bh = cv2.bitwise_and(bh, bh, mask=final_mask)

    vals = bh[m_final > 0]
    if vals.size < 50:
        # ✅ 중앙 말고 "bh 최대"로
        _, _, _, loc = cv2.minMaxLoc(bh, mask=m_final)
        return loc if loc is not None else (cx0, cy0)

    thr = np.percentile(vals, 92)
    spots = (bh >= thr).astype(np.uint8) * 255
    spots = cv2.bitwise_and(spots, spots, mask=final_mask)

    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

    dens_map = cv2.GaussianBlur(spots.astype(np.float32), (0, 0), 3.0)
    _, max_val, _, loc = cv2.minMaxLoc(dens_map, mask=m_final)
    if max_val > 1e-6:
        return loc

    # ✅ 여기서도 중앙 말고 bh 최대
    _, _, _, loc = cv2.minMaxLoc(bh, mask=m_final)
    return loc if loc is not None else (cx0, cy0)



# --------------------
# Dense point inside blob: texture energy weighted centroid
# --------------------
def dense_point_from_texture(cell_gray, blob_mask):
    m = (blob_mask > 0).astype(np.uint8)
    if cv2.countNonZero(m) < 50:
        return None

    clahe = cv2.createCLAHE(2.0, (8, 8))
    eq = clahe.apply(cell_gray)

    lap = cv2.Laplacian(eq, cv2.CV_32F, ksize=3)
    e = np.abs(lap)
    e[m == 0] = 0
    e = cv2.GaussianBlur(e, (0, 0), 2.0)

    vals = e[m > 0]
    if vals.size < 50:
        return None

    thr = np.percentile(vals, 85)
    w = e.copy()
    w[w < thr] = 0

    s = float(w.sum())
    if s < 1e-6:
        ys, xs = np.where(m > 0)
        return (int(xs.mean()), int(ys.mean()))

    yy, xx = np.indices(w.shape)
    dx = int(round(float((w * xx).sum()) / s))
    dy = int(round(float((w * yy).sum()) / s))
    return (dx, dy)

# --------------------
# Colony blob: flatten + rim-remove + percentile + light morphology + seeded CC select
# --------------------
def segment_colony_blob(cell_gray, well_mask, seed=None):
    h, w = cell_gray.shape[:2]
    if h < 20 or w < 20:
        return None

    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # inner (rim 제외)
    inner = cv2.erode(well_mask, kernel11, iterations=3)
    if cv2.countNonZero(inner) < 200:
        inner = well_mask.copy()

    inner_area = float(cv2.countNonZero(inner))
    if inner_area < 200:
        return None

    # rim band
    ring = cv2.subtract(well_mask, inner)

    # illumination flatten (big sigma!)
    g = cell_gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 9.0) + 1e-6
    norm = g / bg
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv = 255 - norm

    # remove rim band 영향
    inv2 = inv.copy()
    inv2[ring > 0] = 0

    vals = inv2[inner > 0]
    if vals.size < 50:
        return None

    # adaptive percentile by contrast
    sigma = float(vals.std())
    p = 62
    if sigma < 18:
        p = 58
    if sigma < 12:
        p = 54

    thr = np.percentile(vals, p)
    bw = (inv2 >= thr).astype(np.uint8) * 255
    bw = cv2.bitwise_and(bw, bw, mask=inner)

    # fallback if too small
    if cv2.countNonZero(bw) < 0.02 * inner_area:
        thr2 = np.percentile(vals, max(p - 7, 40))
        bw = (inv2 >= thr2).astype(np.uint8) * 255
        bw = cv2.bitwise_and(bw, bw, mask=inner)

    # light morphology
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1
    )

    n, lab, st, cen = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return None

    if seed is None:
        sx, sy = w // 2, h // 2
    else:
        sx, sy = seed
    sx = int(max(0, min(int(sx), w - 1)))
    sy = int(max(0, min(int(sy), h - 1)))
    seed_lbl = int(lab[sy, sx])

    best, best_score = None, -1e18
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue
        if area > 0.75 * inner_area:
            continue

        cx, cy = cen[i]
        d2 = ((cx - sx) / w) ** 2 + ((cy - sy) / h) ** 2
        seed_hit = 1 if (seed_lbl == i and seed_lbl != 0) else 0

        score = (1e9 * seed_hit) + area - 20000.0 * d2
        if score > best_score:
            best_score = score
            best = i

    if best is None:
        return None

    blob = (lab == best).astype(np.uint8) * 255
    blob = cv2.bitwise_and(blob, blob, mask=inner)
    return blob

# --------------------
# Shape drawing + measurement (FIX)
# --------------------
def draw_shape_and_measure(result_img, L, T, blob, cx0, cy0, rad, well_mask=None):
    """
    - blob 있으면: ellipse(가능하면) / 아니면 minEnclosingCircle
    - blob 없으면: fallback으로 well 원(rad)을 무조건 그림 (A3/C12 방지)
    return: center_roi, area_px, major_d, minor_d, diameter_px, angle_deg, aspect
    """
    center_roi = (int(L + cx0), int(T + cy0))
    area_px = np.nan
    major_d = np.nan
    minor_d = np.nan
    diameter_px = np.nan
    angle_deg = np.nan
    aspect = np.nan

    # blob 실패 -> fallback well circle
    if blob is None or cv2.countNonZero(blob) < 50:
        cv2.circle(result_img, center_roi, int(rad), (0, 0, 255), 1)
        diameter_px = float(2.0 * rad)
        return center_roi, area_px, major_d, minor_d, diameter_px, angle_deg, aspect

    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        cv2.circle(result_img, center_roi, int(rad), (0, 0, 255), 1)
        diameter_px = float(2.0 * rad)
        return center_roi, area_px, major_d, minor_d, diameter_px, angle_deg, aspect

    cnt = max(cnts, key=cv2.contourArea)
    area_px = float(cv2.contourArea(cnt))

    # ellipse 먼저 시도
    if len(cnt) >= 5:
        (ex, ey), (MA, ma), angle = cv2.fitEllipse(cnt)
        center_roi = (int(round(L + ex)), int(round(T + ey)))

        major_d = float(max(MA, ma))
        minor_d = float(min(MA, ma))
        diameter_px = float((major_d + minor_d) / 2.0)
        angle_deg = float(angle)
        aspect = float(major_d / max(minor_d, 1e-6))

        axes = (int(round(major_d / 2.0)), int(round(minor_d / 2.0)))
        cv2.ellipse(result_img, center_roi, axes, angle, 0, 360, (0, 0, 255), 1)
        return center_roi, area_px, major_d, minor_d, diameter_px, angle_deg, aspect

    # ellipse 안되면 minEnclosingCircle
    (x, y), rr = cv2.minEnclosingCircle(cnt)
    center_roi = (int(round(L + x)), int(round(T + y)))
    diameter_px = float(2.0 * rr)
    cv2.circle(result_img, (L + int(round(x)), T + int(round(y))), int(round(rr)), (0, 0, 255), 1)
    return center_roi, area_px, major_d, minor_d, diameter_px, angle_deg, aspect

# --------------------
# 4) Run 96 cells and save
# --------------------
result_img = roi_color.copy()
rows_name = "ABCDEFGH"
records = []

# grid debug
grid_debug = roi_color.copy()
draw_grid_overlay(grid_debug, cell_bounds, rows=8, cols=12,
                  color=(0, 255, 255), thickness=1, with_label=True)
cv2.imwrite("grid_debug.png", grid_debug)
print("Saved: grid_debug.png")

for r in range(8):
    for c in range(12):
        L, T, R, B = cell_bounds(r, c)
        cell = roi[T:B, L:R]
        h, w = cell.shape[:2]
        if h < 20 or w < 20:
            continue

        # well mask
        cx0, cy0 = w // 2, h // 2
        rad = int(0.43 * min(w, h))
        well_mask = np.zeros((h, w), np.uint8)
        cv2.circle(well_mask, (cx0, cy0), rad, 255, -1)

        # inner mask (for dense)
        inner_mask = cv2.erode(
            well_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=3
        )
        if cv2.countNonZero(inner_mask) < 200:
            inner_mask = well_mask.copy()

        well = f"{rows_name[r]}{c+1}"

        # dense point (spots) first
        dx_cell, dy_cell = dense_point_from_spots(cell, inner_mask)

        # blob seeded by dense
        blob = segment_colony_blob(cell, well_mask, seed=(dx_cell, dy_cell))

        # re-dense inside blob using texture centroid
        if blob is not None and cv2.countNonZero(blob) > 200:
            loc = dense_point_from_texture(cell, blob)
            if loc is not None:
                dx_cell, dy_cell = loc

        dx = int(L + dx_cell)
        dy = int(T + dy_cell)

        # draw shape + measure (✅ blob 실패해도 원은 무조건 그림)
        center_roi, area_px, major_d, minor_d, diameter_px, angle_deg, aspect = \
            draw_shape_and_measure(result_img, L, T, blob, cx0, cy0, rad, well_mask=well_mask)

        # visualize dense + label
        cv2.drawMarker(result_img, (dx, dy), (255, 0, 255), cv2.MARKER_CROSS, 5, 1)
        cv2.putText(result_img, well, (center_roi[0] + 2, center_roi[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)

        records.append({
            "well": well, "row": r, "col": c,
            "area_px": area_px,
            "diameter_px": diameter_px,
            "major_d_px": major_d,
            "minor_d_px": minor_d,
            "aspect": aspect,
            "angle_deg": angle_deg,
            "center_x_roi": center_roi[0], "center_y_roi": center_roi[1],
            "dense_x_roi": dx, "dense_y_roi": dy,
            "center_x_img": center_roi[0] + x1, "center_y_img": center_roi[1] + y1,
            "dense_x_img": dx + x1, "dense_y_img": dy + y1
        })

# overlay grid (optional)
draw_grid_overlay(result_img, cell_bounds, rows=8, cols=12,
                  color=(0, 255, 255), thickness=1, with_label=False)

df = pd.DataFrame(records).sort_values(["row", "col"]).reset_index(drop=True)
print(f"Detected wells: {len(df)} / 96")

df.to_csv("colony_info.csv", index=False)
cv2.imwrite("colony_result.png", result_img)
print("Saved: colony_info.csv, colony_result.png")
