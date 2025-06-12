import torch
import torch.nn as nn
from model.model import EnsembleMLP
from dxf_parser.visualization import visualize_dxf_dim_match_colored
from model.dataset import get_unique_topk
import pandas as pd


def train_model(train_dataset, model_save_path, epochs=100):
    input_dim = train_dataset[0][0].shape[1]  # x, weights 반환
    model = EnsembleMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_acc = 0
        for i in range(len(train_dataset)):
            x, weights = train_dataset[i]  # x: [N, D], weights: [N]
            scores = model(x)
            topk = torch.topk(scores, 3).indices.cpu().numpy().tolist()
            gt_indices = train_dataset.important_indices_list[i]
            correct = sum([1 for idx in topk if idx in gt_indices])
            acc = correct / max(1, len(gt_indices))
            target = torch.zeros_like(scores)
            if gt_indices:
                target[gt_indices] = 1.0
            # BCEWithLogitsLoss에 weight 인자 사용 (샘플별 가중치)
            loss = nn.BCEWithLogitsLoss(weight=weights)(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_acc += acc
        print(f"Epoch {epoch+1} | Train p@3: {total_acc/len(train_dataset):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# def test_model(test_dataset, model_save_path):
#     input_dim = test_dataset[0].shape[1]
#     model = EnsembleMLP(input_dim)  # 새로 정의한 앙상블 모델
#     model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
#     model.eval()
#     with torch.no_grad():
#         for i in range(len(test_dataset)):
#             x = test_dataset[i]
#             scores = model(x)
#             features = test_dataset.features_list[i]
#             relations = test_dataset.relations_list[i]
#             all_items = features + relations
#             id_list = test_dataset.id_list[i]
#             topk = get_unique_topk(scores, all_items, k=10)
#             gt_indices = test_dataset.important_indices_list[i]
#             correct = sum([1 for idx in topk if idx in gt_indices])

#             # 실제 중요 치수 정보 추출
#             print(f"\n[테스트 {i+1}] 실제 중요 치수 정보:")
#             dimensions = test_dataset.dimensions_list[i]
#             feature_dict = {f['id']: f for f in features}
#             relation_dict = {r['id']: r for r in relations}
#             for d in dimensions:
#                 match_id = d.get('matching_idx')
#                 if match_id in feature_dict:
#                     f = feature_dict[match_id]
#                     desc = f"type={f['type']}"
#                     if f['type'] == 'LINE':
#                         desc += f", length={f.get('length',0):.3f}"
#                     elif f['type'] in ['CIRCLE', 'ARC']:
#                         desc += f", radius={f.get('radius',0):.3f}"
#                     print(f"  id={match_id} (feature) | {desc}")
#                 elif match_id in relation_dict:
#                     r = relation_dict[match_id]
#                     desc = f"type={r['type']}"
#                     if 'distance' in r:
#                         desc += f", distance={r['distance']:.3f}"
#                     elif 'angle' in r:
#                         desc += f", angle={r['angle']:.3f}"
#                     print(f"  id={match_id} (relation) | {desc}")
#                 else:
#                     print(f"  id={match_id} (unknown)")

#             print(f"\n예측 상위 10개 index: {topk}")
#             print(f"실제 중요 치수 index: {gt_indices}")
#             print(f"정답 개수(top-10): {correct} / {len(gt_indices)}")

#             print("\n[예측 상위 10개 형상/관계 정보]")
#             for rank, idx in enumerate(topk, 1):
#                 item = all_items[idx]
#                 print(f"{rank}. id: {item['id']}, type: {item['type']}", end='')
#                 if item['type'] == 'LINE':
#                     print(f", length: {item.get('length', 0)}")
#                 elif item['type'] in ['CIRCLE', 'ARC']:
#                     print(f", radius: {item.get('radius', 0)}")
#                 elif 'distance' in item:
#                     print(f", distance: {item.get('distance', 0)}")
#                 elif 'angle' in item:
#                     print(f", angle: {item.get('angle', 0)}")
#                 else:
#                     print()

#             # 시각화: 실제 치수(dimensions), 예측 topk 인덱스 모두 표시
#             visualize_dxf_dim_match_colored(
#                 features=features,
#                 relations=relations,
#                 dimensions=dimensions,      # 실제 치수 정보
#                 predicted_indices=topk      # 예측 결과 인덱스
#             )



# def test_model(test_dataset, model_save_path):
#     input_dim = test_dataset[0].shape[1]
#     model = EnsembleMLP(input_dim)
#     model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
#     model.eval()
#     with torch.no_grad():
#         for i in range(len(test_dataset)):
#             x = test_dataset[i]
#             scores = model(x)
#             features = test_dataset.features_list[i]
#             relations = test_dataset.relations_list[i]
#             all_items = features + relations
#             id_list = test_dataset.id_list[i]
#             topk = get_unique_topk(scores, all_items, k=10)
#             gt_indices = test_dataset.important_indices_list[i]
#             correct = sum([1 for idx in topk if idx in gt_indices])
#             dimensions = test_dataset.dimensions_list[i]

#             # 실제 중요 치수 정보 (nominal_value, matching_idx, feature/relation 정보)
#             print("\n=== 실제 중요 치수 목록 ===")
#             actual_dim_info = []
#             feature_dict = {str(f['id']): f for f in features}
#             relation_dict = {str(r['id']): r for r in relations}
#             for d in dimensions:
#                 if 'matching_idx' not in d:
#                     continue
#                 nominal_value = d.get('nominal_value', None)
#                 if nominal_value is None:
#                     continue
#                 match_id = str(d['matching_idx'])
#                 item = None
#                 if match_id in feature_dict:
#                     item = feature_dict[match_id]
#                 elif match_id in relation_dict:
#                     item = relation_dict[match_id]
#                 if item:
#                     info = f"nominal_value={nominal_value:.3f}, id={match_id}, type={item['type']}"
#                     if item['type'] == 'LINE':
#                         info += f", length={item.get('length', 0):.3f}"
#                     elif item['type'] in ['CIRCLE', 'ARC']:
#                         info += f", radius={item.get('radius', 0):.3f}"
#                     elif 'distance' in item:
#                         info += f", distance={item.get('distance', 0):.3f}"
#                     elif 'angle' in item:
#                         info += f", angle={item.get('angle', 0):.3f}"
#                     print(f"  {info}")
#                     actual_dim_info.append((match_id, nominal_value, item))
#                 else:
#                     print(f"  nominal_value={nominal_value:.3f}, id={match_id} (unknown)")





#             # 예측 중요 치수 후보 정보 (Top-10)
#             print("\n=== 예측 중요 치수 후보 (Top-10) ===")
#             predicted_info = []
#             for rank, idx in enumerate(topk, 1):
#                 item = all_items[idx]
#                 info = f"{rank}. id={item['id']}, type={item['type']}"
#                 if item['type'] == 'LINE':
#                     info += f", length={item.get('length', 0):.3f}"
#                 elif item['type'] in ['CIRCLE', 'ARC']:
#                     info += f", radius={item.get('radius', 0):.3f}"
#                 elif 'distance' in item:
#                     info += f", distance={item.get('distance', 0):.3f}"
#                 elif 'angle' in item:
#                     info += f", angle={item.get('angle', 0):.3f}"
#                 print(info)
#                 predicted_info.append((str(item['id']), item))

#             # 실제 중요 치수와 예측 후보 비교
#             print("\n=== 예측값과 실제 중요 치수 비교 ===")
#             actual_ids = {info[0] for info in actual_dim_info}
#             predicted_ids = {info[0] for info in predicted_info}
#             matched = actual_ids & predicted_ids
#             matched_info = []
#             for match_id in matched:
#                 for info in actual_dim_info:
#                     if info[0] == match_id:
#                         matched_info.append(info)
#                         break
#             for info in matched_info:
#                 print(f"  id={info[0]} (실제: nominal_value={info[1]:.3f}, 예측: O)")
#             for info in actual_dim_info:
#                 if info[0] not in matched:
#                     print(f"  id={info[0]} (실제: nominal_value={info[1]:.3f}, 예측: X)")
#             print(f"\n맞춘 개수: {len(matched)} / {len(actual_dim_info)}")
#             print(f"정확도: {len(matched)/max(1, len(actual_dim_info)):.2%}")



def test_model(test_dataset, model_save_path):
    input_dim = test_dataset[0][0].shape[1]
    model = EnsembleMLP(input_dim)
    model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, weights = test_dataset[i]
            scores = model(x)
            all_items = test_dataset.all_items_list[i]
            id_list = test_dataset.id_list[i]
            topk = get_unique_topk(scores, all_items, k=10)
            gt_indices = test_dataset.important_indices_list[i]

            # --- 정답 개수(type, value 기준) 계산 ---
            def get_item_type_and_value(item):
                item_type = item.get('type', None)
                for key in ['nominal_value', 'length', 'radius', 'distance', 'angle']:
                    if key in item:
                        return item_type, round(float(item[key]), 6)
                return item_type, None

            gt_type_val_set = set()
            for idx in gt_indices:
                gt_item = all_items[idx]
                gt_type_val_set.add(get_item_type_and_value(gt_item))

            correct = 0
            matched = set()
            for idx in topk:
                pred_type_val = get_item_type_and_value(all_items[idx])
                if pred_type_val in gt_type_val_set and pred_type_val not in matched:
                    correct += 1
                    matched.add(pred_type_val)

            # --- 실제 중요 치수 정보 출력 ---
            print(f"\n[테스트 {i+1}] 실제 중요 치수 정보:")

            gt_type_val_list = []
            for idx in gt_indices:
                item = all_items[idx]
                type_val = get_item_type_and_value(item)
                gt_type_val_list.append(type_val)
                # print(f"  - type={type_val[0]}, value={type_val[1]}")


            dimensions = test_dataset.dimensions_list[i]
            features = test_dataset.features_list[i]
            relations = test_dataset.relations_list[i]
            feature_dict = {f['id']: f for f in features}
            relation_dict = {r['id']: r for r in relations}
            for d in dimensions:
                match_id = d.get('matching_idx')
                if match_id in feature_dict:
                    f = feature_dict[match_id]
                    desc = f"type={f['type']}"
                    if f['type'] == 'LINE':
                        desc += f", length={f.get('length',0):.3f}"
                    elif f['type'] in ['CIRCLE', 'ARC']:
                        desc += f", radius={f.get('radius',0):.3f}"
                    print(f"{desc}")
                elif match_id in relation_dict:
                    r = relation_dict[match_id]
                    desc = f"type={r['type']}"
                    if 'distance' in r:
                        desc += f", distance={r['distance']:.3f}"
                    elif 'angle' in r:
                        desc += f", angle={r['angle']:.3f}"
                    print(f"{desc}")
                else:
                    print(f"  id={match_id} (unknown)")

            # print(f"\n예측 상위 10개 index: {topk}")
            # print(f"실제 중요 치수 index: {gt_indices}")

            print("\n[예측 상위 10개 형상/관계 정보]")
            for rank, idx in enumerate(topk, 1):
                item = all_items[idx]


                pred_type_val = get_item_type_and_value(item)
                is_correct = '(예측: O)' if pred_type_val in gt_type_val_list else ''





                print(f"{rank}. type: {item['type']}", end='')
                if item['type'] == 'LINE':
                    print(f", length: {item.get('length', 0):.2f} {is_correct}")
                elif item['type'] in ['CIRCLE', 'ARC']:
                    print(f", radius: {item.get('radius', 0):.2f} {is_correct}")
                elif 'distance' in item:
                    print(f", distance: {item.get('distance', 0):.2f} {is_correct}")
                elif 'angle' in item:
                    print(f", angle: {item.get('angle', 0):.2f} {is_correct}")
                else:
                    print()

            print(f"\n정답 개수(top-10): {correct} / {len(gt_indices)}\n")

            # 시각화: 실제 치수(dimensions), 예측 topk 인덱스 모두 표시
            visualize_dxf_dim_match_colored(
                features=features,
                relations=relations,
                dimensions=dimensions,
                predicted_indices=topk,
                gt_type_val_set=gt_type_val_set,  # 추가
               all_items=all_items  # ← 추가

            )
