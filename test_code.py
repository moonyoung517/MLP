import pandas as pd
from dxf_parser.feature_extractor import extract_dxf_features, mark_outer_features
from dxf_parser.dimension_matcher import assign_dimension_matching_idx
from dxf_parser.visualization import visualize_dxf_dim_match_colored
from dxf_parser.relations import compute_shape_relations
from dxf_parser.id_generator import IDGenerator
from dxf_parser.dimensions import extract_dimensions


def extract_dxf_all(file_path):
    features = extract_dxf_features(file_path)
    id_gen = IDGenerator(start=len(features) + 1)
    relations = compute_shape_relations(features, id_gen)
    dimensions = extract_dimensions(file_path)

    return features, relations, dimensions

if __name__ == "__main__":

    features, relations, dimensions = extract_dxf_all("test/Model_#02_02_012_View_Left.dxf")

    # 치수 매칭 및 최외각 표시
    dimensions = assign_dimension_matching_idx(features, relations, dimensions)
    features = mark_outer_features(features)

    # DataFrame 생성 및 출력
    df_features = pd.DataFrame(features)
    df_relations = pd.DataFrame(relations)
    df_dimensions = pd.DataFrame(dimensions)
    
    print("\n=== 형상 정보 ===")
    print(df_features)

    print("\n=== 형상 관계 ===")
    print(df_relations)

    print("\n=== 치수 정보 ===")
    print(df_dimensions)


    visualize_dxf_dim_match_colored(features, relations, dimensions)
