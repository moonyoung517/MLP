import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dxf_parser.feature_extractor import extract_dxf_features
from dxf_parser.dimensions import extract_dimensions
from dxf_parser.relations import compute_shape_relations
from dxf_parser.id_generator import IDGenerator

def normalize_angle(angle):
    return angle % 180  # 0~180도 범위로 정규화

def project_point(point, origin, direction):
    vec = (point[0]-origin[0], point[1]-origin[1])
    return (vec[0]*direction[0] + vec[1]*direction[1]) / (direction[0]**2 + direction[1]**2)


def points_close(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_line_from_relation(r, feature_dict):
    h1, h2 = r['handles']
    f1 = next((f for f in feature_dict.values() if f['handle'] == h1), None)
    f2 = next((f for f in feature_dict.values() if f['handle'] == h2), None)
    if f1 and f2:
        if f1['type'] == 'LINE':
            p1 = ((f1['start'][0] + f1['end'][0]) / 2, (f1['start'][1] + f1['end'][1]) / 2)
        else:
            p1 = f1.get('center', f1.get('start', (0, 0)))
        if f2['type'] == 'LINE':
            p2 = ((f2['start'][0] + f2['end'][0]) / 2, (f2['start'][1] + f2['end'][1]) / 2)
        else:
            p2 = f2.get('center', f2.get('start', (0, 0)))
        return np.array([p1, p2])
    return np.zeros((2,2))

def get_distinct_colors(n):
    base_colors = plt.get_cmap('tab10').colors
    if n <= 10:
        return base_colors[:n]
    else:
        cmap = plt.get_cmap('hsv')
        return [cmap(i / n) for i in range(n)]
    

def extract_dxf_all(file_path):
    features = extract_dxf_features(file_path)
    id_gen = IDGenerator(start=len(features) + 1)
    relations = compute_shape_relations(features, id_gen)
    dimensions = extract_dimensions(file_path)

    return features, relations, dimensions
