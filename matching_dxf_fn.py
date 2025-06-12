import os
import glob
import os
import numpy as np
import ezdxf
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pandas as pd


# ============================
# 1. DXF → 선분 점 샘플링
# ============================
def extract_points_from_dxf(filename, sample_dist=0.7):
    try:
        doc = ezdxf.readfile(filename)
    except Exception as e:
        print(f"[ERROR] DXF 파일 열기 실패: {filename} → {e}")
        return np.empty((0, 2))

    msp = doc.modelspace()
    points = []

    for e in msp:
        coords = []

        try:
            etype = e.dxftype()

            if etype == "LWPOLYLINE":
                coords = [(pt[0], pt[1]) for pt in e.get_points()]

            elif etype == "LINE":
                start = e.dxf.start
                end = e.dxf.end
                coords = [(start[0], start[1]), (end[0], end[1])]

            elif etype == "CIRCLE":
                center = np.array(tuple(e.dxf.center)[:2])
                radius = e.dxf.radius
                n_points = max(16, int(2 * np.pi * radius // sample_dist))
                angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
                coords = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]

            elif etype == "ARC":
                center = np.array(tuple(e.dxf.center)[:2])
                radius = e.dxf.radius
                start_angle = np.deg2rad(e.dxf.start_angle)
                end_angle = np.deg2rad(e.dxf.end_angle)
                if end_angle < start_angle:
                    end_angle += 2*np.pi
                n_points = max(8, int(radius * abs(end_angle - start_angle) // sample_dist))
                angles = np.linspace(start_angle, end_angle, n_points)
                coords = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]

            elif etype == "SPLINE":
                try:
                    spline = e.construction_tool()
                    spline_points = spline.approximate_uniform(segments=100)  # 균일 간격 샘플링
                    coords = [(p[0], p[1]) for p in spline_points]
                except Exception as err:
                    #print(f"[WARNING] SPLINE 보간 실패: {type(err).__name__}: {err}")
                    continue
            
            elif etype == "POLYLINE":
                coords = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()]

            else:
                continue

            if len(coords) < 2:
                continue

            pl = LineString(coords)
            n_points = max(2, int(pl.length // sample_dist))
            for i in range(n_points):
                p = pl.interpolate(i / (n_points - 1), normalized=True)
                points.append([p.x, p.y])

        except Exception as err:
            print(f"[WARNING] 엔티티 처리 실패 ({e.dxftype()}): {type(err).__name__}: {err}")
            continue

    return np.array(points)


# ============================
# 2. ICP 정합
# ============================

def icp_2d(source, target, max_iter=100, tol=1e-30):
    src = np.copy(source)
    prev_error = float('inf')
    tree = cKDTree(target)  # ← 반드시 scipy 버전

    for _ in range(max_iter):
        distances, indices = tree.query(src, k=1)  # indices: (N,), distances: (N,)
        matched = target[indices]  # [:, 0] 없음

        src_centered = src - src.mean(axis=0)
        matched_centered = matched - matched.mean(axis=0)

        H = src_centered.T @ matched_centered
        U, _, VT = np.linalg.svd(H)
        R = U @ VT

        if np.linalg.det(R) < 0:
            VT[1, :] *= -1
            R = U @ VT

        t = matched.mean(axis=0) - src.mean(axis=0) @ R
        src = (src @ R) + t

        rmse_error = np.mean(distances ** 2)
        if abs(prev_error - rmse_error) < tol:
            break
        prev_error = rmse_error

    return src, rmse_error



# ============================
# 3. Multi-Start ICP 시도
# ============================
def try_multiple_icp_inits(source, target, translations, rotations_deg):

    best_error = float('inf')
    best_aligned = None
    best_params = None

    for angle in rotations_deg:
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        rotated = source @ R.T

        for t in translations:
            # print(t, angle)
            translated = rotated + t
            aligned, error = icp_2d(translated, target)

            if error < best_error:
                best_error = error
                best_aligned = aligned
                best_params = (R, t, angle)

    return best_aligned, best_error, best_params


# ============================
# 4. 시각화 함수
# ============================
def plot_alignment(source, target, title=""):
    plt.figure(figsize=(25, 20))
    plt.scatter(target[:,0], target[:,1], c='b', label='Target (real)', alpha=0.5, s=1)
    plt.scatter(source[:,0], source[:,1], c='r', label='Source (view)', alpha=1, s=1)
    plt.legend()
    plt.axis("equal")
    plt.title(title)
    plt.grid(True)
    plt.show()

# ============================
# 5. 실행부
# ============================
def main(real_dxf_path):
    data = []
    real_dxf = extract_points_from_dxf(real_dxf_path)
    if real_dxf.size == 0:
        raise RuntimeError(f"[ERROR] 실제 도면에서 유효한 점이 추출되지 않았습니다: {real_dxf_path}")

    result = {}
    view_list = ["front", "back", "left", "right", "top", "bottom"]

    # 위치 및 회전 초기화 파라미터 설정
    grid_range = np.linspace(-150, 150, 16)  # 예: -100 ~ +100 mm
    translations = np.array([[x, y] for x in grid_range for y in grid_range])
    rotations = np.arange(0, 360, 30)  # 0 ~ 330 deg

    for view_name in view_list:
        view_path = f"{real_dxf_path.replace('.dxf','')}_{view_name}_view_all_edges.dxf"
        test_dxf = extract_points_from_dxf(view_path)

        if test_dxf.size == 0:
            print(f"[SKIP] {view_name:>6} → 점이 없음 또는 처리 실패")
            continue

        try:
            aligned_src, error, best_params = try_multiple_icp_inits(test_dxf, real_dxf, translations, rotations)

            match_score = 1 / (1 + error)
            result[view_name] = (match_score, error)

#            print(f"[MATCH] {view_name:>6} → 정합률: {match_score:.4f} (RMSE: {error:.4f}), θ={best_params[2]:.1f}°")

            # 시각화
#            plot_alignment(aligned_src, real_dxf, title=f"{view_name} / θ={best_params[2]:.1f}°")

        except Exception as e:
            print(f"[FAIL]  {view_name:>6} → ICP 실패: {e}")

#    print("\n--- 정렬된 매칭 결과 ---")
    # 정렬 및 출력
    for view, (score, rmse) in sorted(result.items(), key=lambda x: x[1][0], reverse=True):
#        print(f"{view:<10} → match score: {score:.4f}, RMSE: {rmse:.4f}")
        data.append({"view": view, "score":score, "rmse":rmse})
    return data