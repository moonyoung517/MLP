# 단위 변환 함수 및 상수
UNIT_CONVERSION = {
    1: 25.4,    # inch → mm
    4: 1.0,     # mm
    5: 10.0,    # cm → mm
    6: 1000.0   # m → mm
}

def convert_units(value, unit_code):
    return value * UNIT_CONVERSION.get(unit_code, 1.0)
