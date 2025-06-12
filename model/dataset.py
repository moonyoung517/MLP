from torch.utils.data import Dataset
import torch
from dxf_parser.feature_extractor import extract_dxf_features
from dxf_parser.dimensions import extract_dimensions
from dxf_parser.dimension_matcher import assign_dimension_matching_idx
from dxf_parser.feature_extractor import reassign_ids
from dxf_parser.feature_extractor import mark_outer_features
from dxf_parser.utils import extract_dxf_all


import torch
from torch.utils.data import Dataset

def find_feature_by_handle(features, handle):
    for f in features:
        if f['handle'] == handle:
            return f
    return None

def get_feature_location_info(feature):
    if feature is None:
        return [0.0, 0.0, 0.0]
    if feature['type'] == 'LINE':
        center_x = (feature['start'][0] + feature['end'][0]) / 2
        center_y = (feature['start'][1] + feature['end'][1]) / 2
        size = feature.get('length', 0.0)
    elif feature['type'] in ['CIRCLE', 'ARC']:
        center_x, center_y = feature['center'][0], feature['center'][1]
        size = feature.get('radius', 0.0)
    else:
        center_x, center_y, size = 0.0, 0.0, 0.0
    return [center_x, center_y, size]

def combined_feature_vector(item, features=None):
    type_dict = {'LINE': 0, 'CIRCLE': 1, 'ARC': 2, 'POLYLINE': 3,
                 'LINE_LINE_PARALLEL': 4, 'LINE_LINE_ANGLE': 5,
                 'CIRCLE_CIRCLE_DISTANCE': 6, 'LINE_CIRCLE_DISTANCE': 7, 'ARC_CENTER_LINE_DISTANCE': 8}
    type_vec = [0] * 9
    if item['type'] in type_dict:
        type_vec[type_dict[item['type']]] = 1

    num_info = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if item['type'] == 'LINE':
        num_info[0] = item.get('length', 0)
        num_info[1] = item.get('angle', 0)
    elif item['type'] in ['CIRCLE', 'ARC']:
        num_info[2] = item.get('radius', 0)
        if item['type'] == 'ARC':
            num_info[3] = item.get('start_angle', 0)
            num_info[4] = item.get('end_angle', 0)
    elif 'distance' in item:
        num_info[5] = item.get('distance', 0)
    elif 'angle' in item:
        num_info[1] = item.get('angle', 0)

    if item['type'] in ['LINE', 'CIRCLE', 'ARC']:
        pos_x, pos_y, size = get_feature_location_info(item)
        pos_info = [pos_x, pos_y, size]
    else:
        pos_info = [0.0, 0.0, 0.0]

    related_info = []
    if item['type'].startswith('LINE_') or item['type'].startswith('CIRCLE_') or item['type'].startswith('ARC_'):
        if features and 'handles' in item:
            for handle in item['handles']:
                f = find_feature_by_handle(features, handle)
                related_info += get_feature_location_info(f)
        else:
            related_info = [0.0] * 6
    else:
        related_info = [0.0] * 6

    outer_val = float(item.get('is_outer', False) or item.get('is_outer_relation', False))
    final_vec = type_vec + num_info + pos_info + related_info + [outer_val]
    return torch.tensor(final_vec, dtype=torch.float)

def build_combined_data(features, relations):
    # 1. LINE 타입 feature는 제외
    filtered_features = [f for f in features if f['type'] != 'LINE']
    all_items = filtered_features + relations
    x = torch.stack([combined_feature_vector(item, features) for item in all_items])
    id_list = [item['id'] for item in all_items]

    # 2. 가중치 계산
    weights = []
    for item in all_items:
        is_outer = item.get('is_outer', False) or item.get('is_outer_relation', False)
        has_radius = 'radius' in item and item.get('radius', 0) != 0
        has_diameter = 'diameter' in item and item.get('diameter', 0) != 0
        if is_outer or has_radius or has_diameter:
            weights.append(1.5)
        else:
            weights.append(1.0)
    weights = torch.tensor(weights, dtype=torch.float)
    return x, id_list, weights, filtered_features

def get_important_indices_from_combined(id_list, dimensions, all_items, topk=10):
    important_dim_ids = set()
    all_item_ids = [item['id'] for item in all_items]
    for d in dimensions[:topk]:
        match_id = d['matching_idx']
        if match_id in all_item_ids:
            important_dim_ids.add(match_id)
    important_indices = [i for i, id in enumerate(id_list) if id in important_dim_ids]
    return important_indices

def get_item_value(item):
    for key in ['nominal_value', 'length', 'radius', 'distance', 'angle']:
        if key in item:
            return round(float(item[key]), 6)
    return None

def get_unique_topk(scores, all_items, k=10):
    sorted_indices = torch.argsort(scores, descending=True).cpu().numpy().tolist()
    seen = set()
    unique_topk = []
    for idx in sorted_indices:
        item = all_items[idx]
        item_type = item.get('type', None)
        item_val = get_item_value(item)
        if item_type is not None and item_val is not None:
            key = (item_type, item_val)
            if key in seen:
                continue
            seen.add(key)
        unique_topk.append(idx)
        if len(unique_topk) >= k:
            break
    return unique_topk

class DXFGraphDataset(Dataset):
    def __init__(self, file_paths):
        self.x_list = []
        self.id_list = []
        self.important_indices_list = []
        self.dimensions_list = []
        self.features_list = []
        self.relations_list = []
        self.weights_list = []
        self.all_items_list = []  # filtered_features + relations

        for path in file_paths:
            features, relations, dimensions = extract_dxf_all(path)
            dimensions = assign_dimension_matching_idx(features, relations, dimensions)
            features = mark_outer_features(features)
            x, id_list, weights, filtered_features = build_combined_data(features, relations)
            all_items = filtered_features + relations
            self.x_list.append(x)
            self.id_list.append(id_list)
            self.dimensions_list.append(dimensions)
            self.features_list.append(features)      # 원본 features 저장
            self.relations_list.append(relations)
            self.weights_list.append(weights)
            self.all_items_list.append(all_items)
            important_indices = get_important_indices_from_combined(id_list, dimensions, all_items)
            self.important_indices_list.append(important_indices)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.weights_list[idx]



# def combined_feature_vector(item):
#     type_dict = {'LINE': 0, 'CIRCLE': 1, 'ARC': 2, 'POLYLINE': 3,
#                  'LINE_LINE_PARALLEL': 4, 'LINE_LINE_ANGLE': 5,
#                  'CIRCLE_CIRCLE_DISTANCE': 6, 'LINE_CIRCLE_DISTANCE': 7, 'ARC_CENTER_LINE_DISTANCE': 8}
#     type_vec = [0] * 9
#     if item['type'] in type_dict:
#         type_vec[type_dict[item['type']]] = 1

#     if item['type'] == 'LINE':
#         val = [item.get('length', 0)]
#     elif item['type'] in ['CIRCLE', 'ARC']:
#         val = [item.get('radius', 0)]
#     elif 'distance' in item:
#         val = [item.get('distance', 0)]
#     elif 'angle' in item:
#         val = [item.get('angle', 0)]
#     else:
#         val = [0]

#     outer_val = float(item.get('is_outer', False) or item.get('is_outer_relation', False))
#     return torch.tensor(type_vec + val + [outer_val], dtype=torch.float)

# def build_combined_data(features, relations):
#     all_items = features + relations
#     x = torch.stack([combined_feature_vector(item) for item in all_items])
#     id_list = [item['id'] for item in all_items]
#     return x, id_list

# def get_important_indices_from_combined(id_list, dimensions, topk=10):
#     important_dim_ids = set()
#     for d in dimensions[:topk]:
#         if isinstance(d['matching_idx'], list):
#             important_dim_ids.update(d['matching_idx'])
#         else:
#             important_dim_ids.add(d['matching_idx'])
#     important_indices = [i for i, id in enumerate(id_list) if id in important_dim_ids]
#     return important_indices


# class DXFGraphDataset(Dataset):
#     def __init__(self, file_paths):
#         self.x_list = []
#         self.id_list = []
#         self.important_indices_list = []
#         self.dimensions_list = []
#         self.features_list = []
#         self.relations_list = []
#         for path in file_paths:
#             features, relations, dimensions = extract_dxf_all(path)
#             dimensions = assign_dimension_matching_idx(features, relations, dimensions)
#             features = mark_outer_features(features)
#             x, id_list = build_combined_data(features, relations)
#             self.x_list.append(x)
#             self.id_list.append(id_list)
#             self.dimensions_list.append(dimensions)
#             self.features_list.append(features)
#             self.relations_list.append(relations)
#             important_indices = get_important_indices_from_combined(id_list, dimensions)
#             self.important_indices_list.append(important_indices)
#     def __len__(self):
#         return len(self.x_list)

#     def __getitem__(self, idx):
#         return self.x_list[idx]


# def get_item_value(item):
#     for key in ['nominal_value', 'length', 'radius', 'distance', 'angle']:
#         if key in item:
#             return round(float(item[key]), 6)
#     return None

# def get_unique_topk(scores, all_items, k=10):
#     sorted_indices = torch.argsort(scores, descending=True).cpu().numpy().tolist()
#     seen = set()
#     unique_topk = []
#     for idx in sorted_indices:
#         item = all_items[idx]
#         item_type = item.get('type', None)
#         item_val = get_item_value(item)
#         if item_type is not None and item_val is not None:
#             key = (item_type, item_val)
#             if key in seen:
#                 continue
#             seen.add(key)
#         unique_topk.append(idx)
#         if len(unique_topk) >= k:
#             break
#     return unique_topk

