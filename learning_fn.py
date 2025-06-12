import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import dropout_edge
from sklearn.model_selection import KFold
from occwl.compound import Compound
from occwl.graph import face_adjacency
from occwl.solid import Solid
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1. 데이터셋 클래스 정의
def build_graph_data(model_path):
    compound = Compound.load_from_step(model_path)
    solid_shape = next(compound.solids(), None)
    solid = solid_shape if isinstance(solid_shape, Solid) else Solid(solid_shape)
    g = face_adjacency(solid, self_loops=True)
    features = []
    for node_id, data in sorted(g.nodes(data=True)):
        face = data['face']
        topo_face = face.topods_face() if hasattr(face, 'topods_face') else face._shape
        parbox = face.uv_bounds()
        umin, vmin = parbox.min_point()
        umax, vmax = parbox.max_point()
        u_center = umin + 0.5 * (umax - umin)
        v_center = vmin + 0.5 * (vmax - vmin)
        center_pt = face.point((u_center, v_center))
        center = np.array(center_pt, dtype=np.float32)
        surf_handle = BRep_Tool.Surface(topo_face)
        props = GeomLProp_SLProps(surf_handle, u_center, v_center, 1, 1e-6)
        if props.IsNormalDefined():
            gp_norm = props.Normal()
            normal = np.array([gp_norm.X(), gp_norm.Y(), gp_norm.Z()], dtype=np.float32)
        else:
            normal = np.zeros(3, dtype=np.float32)
        area = np.array([face.area()], dtype=np.float32)
        features.append(np.concatenate([center, normal, area]))
    x = torch.tensor(np.stack(features), dtype=torch.float)
    edge_list = [[u, v] for u, v in g.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return x, edge_index

class StepViewDataset(Dataset):
    def __init__(self, step_dir, matching_excel, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.step_dir = step_dir
        self.match_df = pd.read_excel(matching_excel)
        self.records = self.match_df.groupby('path')
    def len(self): return len(self.records)
    def get(self, idx):
        path_key = list(self.records.groups.keys())[idx]
        recs = self.records.get_group(path_key)
        name_no_ext = os.path.splitext(os.path.basename(path_key))[0]
        for ext in ['.step', '.stp']:
            model_path = os.path.join(self.step_dir, name_no_ext+ext)
            if os.path.exists(model_path): break
        x, edge_index = build_graph_data(model_path)
        order = ['bottom','top','front','left','right','back']
        y = [float(recs[recs['view']==v]['score']) if v in recs['view'].values else 0.0 for v in order]
        return Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))

class GINViewModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1); self.conv2 = GINConv(nn2)
        self.pool = global_mean_pool
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, 6))
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        out = self.fc(x)
        return torch.sigmoid(out)   # ← Sigmoid 적용

# 학습/평가 함수 정의

def train_epoch(model, loader, optimizer, criterion, device, drop_prob=0.2):
    model.train(); total_loss=0
    for data in loader:
        data = data.to(device)
        eidx, _ = dropout_edge(data.edge_index, p=drop_prob)
        data.edge_index = eidx
        optimizer.zero_grad(); loss=criterion(model(data), data.y.view(-1,6))
        loss.backward(); optimizer.step()
        total_loss += loss.item()*data.num_graphs
    return total_loss/len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval(); total_loss=0
    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            total_loss += criterion(model(data), data.y.view(-1,6)).item()*data.num_graphs
    return total_loss/len(loader.dataset)

# 테스트용 데이터셋 클래스
class TestStepDataset(Dataset):
    def __init__(self, step_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.file_paths = [os.path.join(step_dir,f) for f in os.listdir(step_dir) if f.endswith(('.step','.stp'))]
    def len(self): return len(self.file_paths)
    def get(self, idx):
        p = self.file_paths[idx]
        x, edge_index = build_graph_data(p)
        data = Data(x=x, edge_index=edge_index)
        data.filename = os.path.basename(p)
        return data

# 메인 실행: 학습 + 교차검증 + 앙상블 예측
def main():
    step_dir = 'data'
    excel_path = 'data/step_to_dxf_matching_data_new_preprocessing.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StepViewDataset(step_dir, excel_path)
    in_channels = dataset[0].num_node_features
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_val_models = []

    g = torch.Generator()
    g.manual_seed(300)

    # 5-fold training
    for fold,(tr,vl) in enumerate(kf.split(dataset)):
        train_loader=DataLoader(torch.utils.data.Subset(dataset,tr),batch_size=8,shuffle=True, generator=g)
        val_loader  =DataLoader(torch.utils.data.Subset(dataset,vl),batch_size=8,shuffle=False, generator=g)
        model=GINViewModel(in_channels).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
        crit=torch.nn.MSELoss()
        best_loss=1e9;pat=0
        for epoch in range(1,51):
            tl=train_epoch(model,train_loader,opt,crit,device)
            vloss=eval_epoch(model,val_loader,crit,device)
            sched.step(vloss)
            if vloss<best_loss:
                best_loss=vloss;pat=0;
                torch.save(model.state_dict(),f'model/best_model_fold_preprocessing{fold}.pth')
            else:
                pat+=1
                if pat>=10: break
        all_val_models.append(f'model/best_model_fold_preprocessing{fold}.pth')

    # 테스트 데이터셋 로드 및 앙상블 예측
    test_dir = 'data/test'
    test_ds = TestStepDataset(test_dir)
    test_loader = DataLoader(test_ds, batch_size=1)
    # 모델 로드
    models = []
    for mpath in all_val_models:
        m = GINViewModel(in_channels).to(device)
        m.load_state_dict(torch.load(mpath,map_location=device))
        m.eval(); models.append(m)
    # 예측
    results = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            preds = [m(data).cpu().numpy().flatten() for m in models]
            avg = np.mean(preds, axis=0)
            row = {'filename': data.filename[0]}
            for i,view in enumerate(['bottom','top','front','left','right','back']): row[view]=float(avg[i])
            results.append(row)
    df = pd.DataFrame(results)
    df.to_csv('test_predictions_new_preprocessing.csv',index=False)
    print(df)

