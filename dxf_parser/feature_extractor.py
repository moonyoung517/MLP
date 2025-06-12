import ezdxf
import numpy as np
from .units import convert_units
from scipy.spatial import ConvexHull

def extract_dxf_features(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    unit_code = doc.header.get('$INSUNITS', 4)

    AUX_LAYERS = {'CENTER', 'AUX', 'DIM', 'HIDDEN'}
    AUX_LTYPES = {'CENTER', 'DASHED', 'HIDDEN', 'PHANTOM', 'DASHDOT'}

    features = []

    for entity in msp:
        try:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ''
            ltype = entity.dxf.linetype if hasattr(entity.dxf, 'linetype') else ''
            is_aux = (layer.upper() in AUX_LAYERS) or (ltype.upper() in AUX_LTYPES)

            if entity.dxftype() == 'LINE':
                start = (convert_units(entity.dxf.start.x, unit_code),
                         convert_units(entity.dxf.start.y, unit_code))
                end = (convert_units(entity.dxf.end.x, unit_code),
                       convert_units(entity.dxf.end.y, unit_code))
                dx = end[0]-start[0]
                dy = end[1]-start[1]
                features.append({
                    'id': None,
                    'type': 'LINE',
                    'handle': entity.dxf.handle,
                    'length': np.hypot(dx, dy),
                    'angle': np.degrees(np.arctan2(dy, dx))%360,
                    'start': start,
                    'end': end,
                    'layer': layer,
                    'linetype': ltype,
                    'is_auxiliary': is_aux
                })

            elif entity.dxftype() == 'CIRCLE':
                center = (convert_units(entity.dxf.center.x, unit_code),
                          convert_units(entity.dxf.center.y, unit_code))
                radius = convert_units(entity.dxf.radius, unit_code)
                features.append({
                    'id': None,
                    'type': 'CIRCLE',
                    'handle': entity.dxf.handle,
                    'radius': radius,
                    'center': center,
                    'circumference': 2*np.pi*radius,
                    'layer': layer,
                    'linetype': ltype,
                    'is_auxiliary': is_aux
                })

            elif entity.dxftype() == 'ARC':
                center = (convert_units(entity.dxf.center.x, unit_code),
                          convert_units(entity.dxf.center.y, unit_code))
                radius = convert_units(entity.dxf.radius, unit_code)
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                arc_length = 2*np.pi*radius*(end_angle-start_angle)/360
                features.append({
                    'id': None,
                    'type': 'ARC',
                    'handle': entity.dxf.handle,
                    'radius': radius,
                    'center': center,
                    'arc_length': arc_length,
                    'start_angle': start_angle,
                    'end_angle': end_angle,
                    'layer': layer,
                    'linetype': ltype,
                    'is_auxiliary': is_aux
                })

            elif entity.dxftype() == 'POLYLINE':
                features.append({
                    'id': None,
                    'type': 'POLYLINE',
                    'handle': entity.dxf.handle,
                    'vertex_count': len(entity.vertices),
                    'is_closed': entity.closed,
                    'layer': layer,
                    'linetype': ltype,
                    'is_auxiliary': is_aux
                })

        except Exception as e:
            print(f"Error processing {getattr(entity.dxf, 'handle', 'unknown')}: {str(e)}")

    # 병합 및 id 재부여
    features = merge_connected_lines(features)
    features = reassign_ids(features)
    return features


def merge_connected_lines(features, angle_tol=1e-3, dist_tol=1e-4):
    lines = [f for f in features if f['type']=='LINE' and not f.get('is_auxiliary', False)]
    others = [f for f in features if not (f['type']=='LINE' and not f.get('is_auxiliary', False))]
    used = set()
    merged_lines = []

    def points_close(p1, p2, tol=dist_tol):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1]) < tol

    n = len(lines)
    for i in range(n):
        if i in used:
            continue
        line = lines[i]
        group = [line]
        used.add(i)
        start, end = line['start'], line['end']
        angle = line['angle'] % 180  # 0~180도만 비교

        changed = True
        while changed:
            changed = False
            for j in range(n):
                if j in used:
                    continue
                other = lines[j]
                other_angle = other['angle'] % 180
                if abs(angle - other_angle) < angle_tol:
                    if (points_close(start, other['start']) or
                        points_close(start, other['end']) or
                        points_close(end, other['start']) or
                        points_close(end, other['end'])):
                        group.append(other)
                        used.add(j)
                        pts = [g['start'] for g in group] + [g['end'] for g in group]
                        if abs(angle-90)<1e-2:
                            pts.sort(key=lambda p: p[1])
                        else:
                            pts.sort(key=lambda p: p[0])
                        start, end = pts[0], pts[-1]
                        changed = True
                        break

        merged_lines.append({
            # id는 나중에 부여
            'type': 'LINE',
            'handle': 'Merged_' + '_'.join(str(l['handle']) for l in group),
            'length': np.hypot(end[0]-start[0], end[1]-start[1]),
            'angle': angle,
            'start': start,
            'end': end,
            'layer': line['layer'],
            'linetype': line['linetype'],
            'is_auxiliary': False,
            'merged_from': [l['handle'] for l in group]
        })

    return merged_lines + others

def reassign_ids(features):
    for idx, f in enumerate(features, start=1):
        f['id'] = idx
    return features


def mark_outer_features(features, tol=1e-4):
    # Convex Hull 기반 최외각 판정
    candidates = []
    candidate_indices = []
    for idx, f in enumerate(features):
        if f.get('is_auxiliary', False):
            continue
        if f['type'] == 'LINE':
            candidates.append(f['start'])
            candidate_indices.append((idx, 'start'))
            candidates.append(f['end'])
            candidate_indices.append((idx, 'end'))
        elif f['type'] in ('CIRCLE', 'ARC'):
            candidates.append(f['center'])
            candidate_indices.append((idx, 'center'))

    if len(candidates) < 3:
        for f in features:
            f['is_outer'] = False
        return features


    # 중복 점 제거 (좌표 기준)
    unique_points = []
    unique_indices = []
    seen = set()
    for point, cidx in zip(candidates, candidate_indices):
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)
            unique_indices.append(cidx)
    points = np.array(unique_points)

    if len(points) < 3:
        for f in features:
            f['is_outer'] = False
        return features

    # ConvexHull 계산 (예외 처리)
    try:
        hull = ConvexHull(points)
        hull_vertices = set(hull.vertices)
    except Exception as e:
        print(f"ConvexHull 실패: {e}")
        for f in features:
            f['is_outer'] = False
        return features

    is_outer_flags = [False] * len(features)
    for hidx in hull_vertices:
        idx, pos = unique_indices[hidx]
        is_outer_flags[idx] = True

    for i, f in enumerate(features):
        f['is_outer'] = is_outer_flags[i] if not f.get('is_auxiliary', False) else False

    return features