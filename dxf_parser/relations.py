from .id_generator import IDGenerator
import numpy as np


def compute_shape_relations(features, id_gen, tol_angle=1e-3, tol_dist=1e-6):

    relations = []

    def line_direction(line):
        dx = line['end'][0] - line['start'][0]
        dy = line['end'][1] - line['start'][1]
        norm = np.hypot(dx, dy)
        return (dx / norm, dy / norm) if norm != 0 else (0, 0)

    def point_line_distance(point, line):
        x0, y0 = point
        x1, y1 = line['start']
        x2, y2 = line['end']
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = np.hypot(y2-y1, x2-x1)
        return num / den if den != 0 else 0.0

    def circle_circle_distance(c1, c2):
        center_dist = np.hypot(c1['center'][0] - c2['center'][0], c1['center'][1] - c2['center'][1])
        return max(0, center_dist - c1['radius'] - c2['radius'])

    def line_angle(line1, line2):
        dir1 = line_direction(line1)
        dir2 = line_direction(line2)
        dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return min(angle, 180-angle)

    n = len(features)
    for i in range(n):
        f1 = features[i]
        for j in range(i+1, n):
            f2 = features[j]
            is_aux_rel = f1.get('is_auxiliary', False) or f2.get('is_auxiliary', False)
            is_outer_rel = f1.get('is_outer', False) or f2.get('is_outer', False)

            # 1. LINE - LINE 관계
            if f1['type'] == 'LINE' and f2['type'] == 'LINE':
                angle = line_angle(f1, f2)
                if abs(angle) < tol_angle or abs(angle-180) < tol_angle:
                    dist = point_line_distance(f1['start'], f2)
                    if dist > tol_dist:
                        relations.append({
                            'id': id_gen.get_next(),
                            'type': 'LINE_LINE_PARALLEL',
                            'handles': (f1['handle'], f2['handle']),
                            'distance': dist,
                            'is_auxiliary_relation': is_aux_rel,
                            'is_outer_relation': is_outer_rel
                        })
                else:
                    if not np.isclose(angle, 90, atol=tol_angle):
                        relations.append({
                            'id': id_gen.get_next(),
                            'type': 'LINE_LINE_ANGLE',
                            'handles': (f1['handle'], f2['handle']),
                            'angle': angle,
                            'is_auxiliary_relation': is_aux_rel,
                            'is_outer_relation': is_outer_rel
                        })

            # 2. CIRCLE - CIRCLE 거리
            elif f1['type'] == 'CIRCLE' and f2['type'] == 'CIRCLE':
                dist = circle_circle_distance(f1, f2)
                if dist > tol_dist:
                    relations.append({
                        'id': id_gen.get_next(),
                        'type': 'CIRCLE_CIRCLE_DISTANCE',
                        'handles': (f1['handle'], f2['handle']),
                        'distance': dist,
                        'is_auxiliary_relation': is_aux_rel,
                        'is_outer_relation': is_outer_rel
                    })

            # 3. LINE - CIRCLE 거리
            elif (f1['type'], f2['type']) == ('LINE', 'CIRCLE'):
                dist = point_line_distance(f2['center'], f1) - f2['radius']
                dist = max(0, dist)
                if dist > tol_dist:
                    relations.append({
                        'id': id_gen.get_next(),
                        'type': 'LINE_CIRCLE_DISTANCE',
                        'handles': (f1['handle'], f2['handle']),
                        'distance': dist,
                        'is_auxiliary_relation': is_aux_rel,
                        'is_outer_relation': is_outer_rel
                    })
            elif (f1['type'], f2['type']) == ('CIRCLE', 'LINE'):
                dist = point_line_distance(f1['center'], f2) - f1['radius']
                dist = max(0, dist)
                if dist > tol_dist:
                    relations.append({
                        'id': id_gen.get_next(),
                        'type': 'LINE_CIRCLE_DISTANCE',
                        'handles': (f2['handle'], f1['handle']),
                        'distance': dist,
                        'is_auxiliary_relation': is_aux_rel,
                        'is_outer_relation': is_outer_rel
                    })

            # 4. ARC - LINE 거리
            elif (f1['type'], f2['type']) == ('ARC', 'LINE'):
                dist = point_line_distance(f1['center'], f2)
                if dist > tol_dist:
                    relations.append({
                        'id': id_gen.get_next(),
                        'type': 'ARC_CENTER_LINE_DISTANCE',
                        'handles': (f1['handle'], f2['handle']),
                        'distance': dist,
                        'is_auxiliary_relation': is_aux_rel,
                        'is_outer_relation': is_outer_rel
                    })
            elif (f1['type'], f2['type']) == ('LINE', 'ARC'):
                dist = point_line_distance(f2['center'], f1)
                if dist > tol_dist:
                    relations.append({
                        'id': id_gen.get_next(),
                        'type': 'ARC_CENTER_LINE_DISTANCE',
                        'handles': (f2['handle'], f1['handle']),
                        'distance': dist,
                        'is_auxiliary_relation': is_aux_rel,
                        'is_outer_relation': is_outer_rel
                    })

    return relations