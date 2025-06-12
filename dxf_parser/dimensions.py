import ezdxf
from .units import convert_units


def extract_dimensions(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    unit_code = doc.header.get('$INSUNITS', 4)

    dimensions = []

    for entity in msp:
        if entity.dxftype() == 'DIMENSION':
            try:
                dim_type = entity.dxf.dimtype  # 0: linear, 1: aligned, 2: angular, 3: diameter, 4: radius, ...
                dimlfac = getattr(entity.dxf, 'dimlfac', 1.0)
                actual_value = entity.get_measurement() * dimlfac
                actual_value_mm = convert_units(actual_value, unit_code) if dim_type != 2 else actual_value

                # 치수 타입 이름 매핑
                dim_type_str = {
                    160: 'LINEAR',
                    1: 'LINEAR',
                    162: 'ANGULAR',
                    163: 'DIAMETER',
                    164: 'RADIUS'
                }.get(dim_type, 'UNKNOWN')

                # 각도 처리
                actual_angle = None
                if dim_type == 2:
                    actual_angle = getattr(entity.dxf, 'angle', None)

                # 공차 정보 추출
                override = entity.override()
                tolerance = None
                if override.get('dimtol', 0) or override.get('dimlim', 0):
                    tol_upper = override.get('dimtp', 0.0)
                    tol_lower = override.get('dimtm', 0.0)
                    if dim_type == 2:  # 각도
                        tol_upper_conv = tol_upper
                        tol_lower_conv = -tol_lower
                    else:
                        tol_upper_conv = tol_upper
                        tol_lower_conv = -tol_lower
                    tolerance = {
                        'upper': tol_upper_conv,
                        'lower': tol_lower_conv,
                        'display_type': 'LIMITS' if override.get('dimlim', 0) else 'DEVIATION',
                        'text_scale': override.get('dimtfac', 1.0)
                    }

                # 정의점 처리
                def_points = []
                for name in ['defpoint', 'defpoint2', 'defpoint3']:
                    if hasattr(entity.dxf, name):
                        p = getattr(entity.dxf, name)
                        def_points.append((
                            convert_units(p.x, unit_code),
                            convert_units(p.y, unit_code)
                        ))

                # 결과 추가
                dimensions.append({
                    'handle': entity.dxf.handle,
                    'type': dim_type_str,
                    'nominal_value': actual_angle if dim_type == 2 else actual_value_mm,
                    'dimlfac': dimlfac,
                    'definition_points': def_points,
                    'angle' if dim_type == 2 else 'length': actual_angle if dim_type == 2 else actual_value_mm,
                    'tolerance': tolerance
                })

            except Exception as e:
                print(f"Dimension error {entity.dxf.handle}: {str(e)}")

    # 공차 있는 치수 우선 정렬
    dimensions = sorted(dimensions, key=lambda d: d['tolerance'] is None)

    return dimensions