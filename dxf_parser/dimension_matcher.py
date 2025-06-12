# dimension_matcher.py 예시 임포트

# 1. 필요시, 형상/피처 관련 모듈 임포트
from dxf_parser.feature_extractor import extract_dxf_features  # (필요시)

# 2. 필요시, 치수 관련 모듈 임포트
from dxf_parser.dimensions import extract_dimensions           # (필요시)

# 3. 필요시, 관계 관련 모듈 임포트
from dxf_parser.relations import compute_shape_relations       # (필요시)

# 4. 유틸리티 함수 임포트
from dxf_parser.utils import points_close                      # (필요시)
from dxf_parser.utils import get_line_from_relation            # (필요시)
from dxf_parser.utils import get_distinct_colors               # (필요시)



def assign_dimension_matching_idx(features, relations, dimensions, val_weight=1.0, pos_weight=1.0):
    """
    치수와 형상/관계 정보를 값+좌표로 비교하여, 가장 가까운(오차가 최소인) matching_idx 부여
    val_weight, pos_weight: 값/좌표 오차의 상대적 가중치
    """
    used_ids = set()
    next_id = 1

    for d in dimensions:
        best_score = float('inf')
        best_id = None

        # 1. features에서 최적 매칭 찾기
        for f in features:
            score = None
            if d['type'] == 'LINEAR' and f['type'] == 'LINE':
                val_diff = abs(d['nominal_value'] - f['length'])
                if len(d['definition_points']) >= 2:
                    pos_diff = min(
                        points_close(d['definition_points'][0], f['start']) + points_close(d['definition_points'][1], f['end']),
                        points_close(d['definition_points'][1], f['start']) + points_close(d['definition_points'][0], f['end'])
                    )
                else:
                    pos_diff = 0
                score = val_weight * val_diff + pos_weight * pos_diff
            elif d['type'] == 'DIAMETER' and f['type'] == 'CIRCLE':
                val_diff = abs(d['nominal_value'] - 2*f['radius'])
                pos_diff = 0  # 필요시 중심점 비교 추가 가능
                score = val_weight * val_diff + pos_weight * pos_diff
            elif d['type'] == 'RADIUS' and f['type'] in ['CIRCLE', 'ARC']:
                val_diff = abs(d['nominal_value'] - f['radius'])
                pos_diff = 0  # 필요시 중심점 비교 추가 가능
                score = val_weight * val_diff + pos_weight * pos_diff

            if score is not None and score < best_score:
                best_score = score
                best_id = f['id']

        # 2. relations에서 최적 매칭 찾기
        for r in relations:
            score = None
            if d['type'] == 'LINEAR' and 'distance' in r:
                val_diff = abs(d['nominal_value'] - r['distance'])
                if 'points' in r and len(d['definition_points']) >= 2:
                    pos_diff = min(
                        points_close(d['definition_points'][0], r['points'][0]) + points_close(d['definition_points'][1], r['points'][1]),
                        points_close(d['definition_points'][1], r['points'][0]) + points_close(d['definition_points'][0], r['points'][1])
                    )
                else:
                    pos_diff = 0
                score = val_weight * val_diff + pos_weight * pos_diff
            elif d['type'] == 'ANGULAR' and 'angle' in r:
                val_diff = abs(d['nominal_value'] - r['angle'])
                pos_diff = 0  # 필요시 꼭짓점 비교 추가 가능
                score = val_weight * val_diff + pos_weight * pos_diff

            if score is not None and score < best_score:
                best_score = score
                best_id = r['id']

        # 3. 최적 매칭 id 할당, 없으면 새 id
        if best_id is not None:
            d['matching_idx'] = best_id
            used_ids.add(best_id)
        else:
            while next_id in used_ids:
                next_id += 1
            d['matching_idx'] = next_id
            used_ids.add(next_id)
            next_id += 1

    return dimensions