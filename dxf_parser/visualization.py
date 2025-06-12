import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

def get_item_type_and_value(item):
    item_type = item.get('type', None)
    for key in ['nominal_value', 'length', 'radius', 'distance', 'angle']:
        if key in item:
            return item_type, round(float(item[key]), 6)
    return item_type, None

def get_line_from_relation(relation, feature_dict):
    handles = relation.get('handles', [])
    points = []
    for handle in handles:
        feature = next((f for f in feature_dict.values() if f.get('handle') == handle), None)
        if feature:
            if feature['type'] == 'LINE':
                points.append([(feature['start'][0] + feature['end'][0])/2, (feature['start'][1] + feature['end'][1])/2])
            elif feature['type'] in ['CIRCLE', 'ARC']:
                points.append(feature['center'])
    if len(points) >= 2:
        return [points[0], points[1]]
    else:
        return [[0, 0], [1, 1]]

def visualize_dxf_dim_match_colored(features, relations, dimensions, predicted_indices=None, gt_type_val_set=None, all_items=None):
    if all_items is None:
        all_items = features + relations
    id_to_index = {item['id']: idx for idx, item in enumerate(all_items)}
    id_to_item = {item['id']: item for item in all_items}

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    # 기본 도형 그리기 (배경)
    for f in features:
        if f['type'] == 'LINE':
            ax.plot([f['start'][0], f['end'][0]], [f['start'][1], f['end'][1]], color='#212121', linewidth=1, zorder=1)
        elif f['type'] == 'CIRCLE':
            ax.add_patch(Circle(f['center'], f['radius'], fill=False, color='#212121', linewidth=1, zorder=1))
        elif f['type'] == 'ARC':
            arc = Arc(f['center'], 2*f['radius'], 2*f['radius'], angle=0, theta1=f['start_angle'], theta2=f['end_angle'], color='#212121', linewidth=1, zorder=1)
            ax.add_patch(arc)

    color_list = plt.cm.get_cmap("tab10", len(dimensions))
    text_positions = []  # (x, y) 좌표 저장

    for idx, dim in enumerate(dimensions):
        match_id = dim.get('matching_idx')
        if not match_id:
            continue

        color = color_list(idx)
        if match_id in id_to_item:
            item = id_to_item[match_id]
            is_correct = (predicted_indices is not None and id_to_index[match_id] in predicted_indices) and \
                         (gt_type_val_set is None or get_item_type_and_value(item) in gt_type_val_set)
            # 치수값(텍스트) 색상 및 두께 결정: 예측 성공 시 빨간색+굵게, 아니면 파란색
            text_color = 'red' if is_correct else 'blue'
            text_weight = 'bold' if is_correct else 'normal'

            # 치수값 위치 계산
            if item['type'] in ['LINE', 'CIRCLE', 'ARC']:
                if item['type'] == 'LINE':
                    x = (item['start'][0] + item['end'][0]) / 2
                    y = (item['start'][1] + item['end'][1]) / 2
                else:  # CIRCLE, ARC
                    x = item['center'][0]
                    y = item['center'][1]
            else:
                # relation이면 handles를 통해 feature 찾아서 그리기
                for handle in item.get('handles', []):
                    f = next((f for f in features if f['handle'] == handle), None)
                    if f:
                        if f['type'] == 'LINE':
                            x = (f['start'][0] + f['end'][0]) / 2
                            y = (f['start'][1] + f['end'][1]) / 2
                            break
                        elif f['type'] in ['CIRCLE', 'ARC']:
                            x = f['center'][0]
                            y = f['center'][1]
                            break
                else:
                    # relation에 대한 선분 그리기 (예시: get_line_from_relation 필요)
                    line = get_line_from_relation(item, {f['id']: f for f in features})
                    x = (line[0][0] + line[1][0]) / 2
                    y = (line[0][1] + line[1][1]) / 2

            # 겹침 감지 및 y좌표 조정
            y_offset = 0.0
            threshold = 0.3  # 겹침 기준 거리
            for (tx, ty) in text_positions:
                if abs(tx - x) < threshold and abs(ty - y) < threshold:
                    if is_correct:
                        # 빨간색 값이 보이도록 y좌표를 조정
                        y_offset += 0.2
                    else:
                        # 파란색 값은 빨간색 값과 겹칠 때만 조정
                        pass
            y += y_offset

            # 도형 그리기
            if item['type'] in ['LINE', 'CIRCLE', 'ARC']:
                if item['type'] == 'LINE':
                    ax.plot([item['start'][0], item['end'][0]], [item['start'][1], item['end'][1]], color=color, linewidth=3, zorder=2)
                elif item['type'] == 'CIRCLE':
                    ax.add_patch(Circle(item['center'], item['radius'], fill=False, color=color, linewidth=3, zorder=2))
                elif item['type'] == 'ARC':
                    arc = Arc(item['center'], 2*item['radius'], 2*item['radius'], angle=0, theta1=item['start_angle'], theta2=item['end_angle'], color=color, linewidth=3, zorder=2)
                    ax.add_patch(arc)
            else:
                for handle in item.get('handles', []):
                    f = next((f for f in features if f['handle'] == handle), None)
                    if f:
                        if f['type'] == 'LINE':
                            ax.plot([f['start'][0], f['end'][0]], [f['start'][1], f['end'][1]], color=color, linewidth=3, zorder=2)
                        elif f['type'] == 'CIRCLE':
                            ax.add_patch(Circle(f['center'], f['radius'], fill=False, color=color, linewidth=3, zorder=2))
                        elif f['type'] == 'ARC':
                            arc = Arc(f['center'], 2*f['radius'], 2*f['radius'], angle=0, theta1=f['start_angle'], theta2=f['end_angle'], color=color, linewidth=3, zorder=2)
                            ax.add_patch(arc)
                line = get_line_from_relation(item, {f['id']: f for f in features})
                ax.plot([p[0] for p in line], [p[1] for p in line], color=color, linewidth=2, linestyle='--', zorder=2)

            # 치수값 텍스트 그리기
            ax.text(x, y, f"{dim.get('nominal_value', 0):.2f}",
                    color=text_color, fontsize=10, ha='center', zorder=3, fontweight=text_weight)
            text_positions.append((x, y))

    ax.set_title('Dimension Matches with Prediction Highlights')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle('DXF Visualization: Dimension Matches with Prediction Highlights', y=1.02)
    plt.show()
