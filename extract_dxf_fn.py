# 표준 라이브러리
import os
import glob

# 외부 라이브러리
import ezdxf

# pythonOCC (OpenCASCADE Python 래퍼)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2

from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import (
    HLRBRep_Algo,
    HLRBRep_HLRToShape,
    HLRBRep_TypeOfResultingEdge
)

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.GeomAbs import GeomAbs_Line

from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape


# ========================================
# --- (기존) STEP 파일 로드 함수 ---
# ========================================
def load_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 파일 로딩 실패: {filename}")
    reader.TransferRoots()
    return reader.Shape()

# ========================================
# --- (기존) HLR 선 추출 함수 ---
# ========================================
def extract_all_edges(shape, direction):
    projector = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0,0,0), direction))
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)

    o_vis = hlr.OutLineVCompound()
    vis   = hlr.VCompound()

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    if o_vis and not o_vis.IsNull():
        builder.Add(compound, o_vis)
    if vis and not vis.IsNull():
        builder.Add(compound, vis)

    return compound

# ========================================
# --- (기존) DXF 저장 함수 ---
# ========================================
def save_edges_as_dxf_all_types(edge_shape, filename, num_points=50):
    doc = ezdxf.new()
    msp = doc.modelspace()
    exp = TopExp_Explorer(edge_shape, TopAbs_EDGE)
    count = 0

    while exp.More():
        edge = exp.Current()
        curve = BRepAdaptor_Curve(edge)
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        pts = []
        for i in range(num_points+1):
            u = u0 + (u1-u0)*i/num_points
            p = curve.Value(u)
            pts.append((p.X(), p.Y()))
        if len(pts) >= 2:
            msp.add_lwpolyline(pts)
            count += 1
        exp.Next()

    doc.saveas(filename)
    print(f"[SAVED] {os.path.basename(filename)} (edge 수: {count})")

# ========================================
# --- 메인 처리 로직: 폴더 내 모든 .stp → .dxf ---
# ========================================
if __name__ == "__main__":
    # 1) 입력 폴더 지정
    input_folder = "data"  # ← .stp 파일이 있는 폴더 경로

    # 2) 처리할 STEP 파일 목록 수집
    patterns = ["*.stp", "*.step"]  # 필요시 확장자 추가
    step_files = []
    for pat in patterns:
        step_files.extend(glob.glob(os.path.join(input_folder, pat)))

    # 3) 뷰 정의
    view_directions = {
        "front":  gp_Dir(0,  1, 0),
        "back":   gp_Dir(0, -1, 0),
        "left":   gp_Dir(-1, 0, 0),
        "right":  gp_Dir(1,  0, 0),
        "top":    gp_Dir(0,  0, 1),
        "bottom": gp_Dir(0,  0,-1),
    }

    # 4) 파일별 반복 처리
    for step_path in step_files:
        base = os.path.splitext(os.path.basename(step_path))[0]
        try:
            shape = load_step_file(step_path)
        except RuntimeError as e:
            print(e)
            continue

        for name, direction in view_directions.items():
            edges = extract_all_edges(shape, direction)
            dxf_name = f"{base}_{name}_view_all_edges.dxf"
            dxf_path = os.path.join(input_folder, dxf_name)
            save_edges_as_dxf_all_types(edges, dxf_path)
