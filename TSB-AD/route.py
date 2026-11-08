"""Morphology routing + adaptive ensemble: route to a strong model by source prior, rank-weighted fusion, official-metric evaluation."""
import sys, io, contextlib, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
import numpy as np, pandas as pd, torch, random
from scipy.stats import rankdata
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

seed = 2024
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
CHECKPOINTS = {'Time_RCD': 'Time-RCD/best_model/pretrain_checkpoint_best_uni.pth'}
ROOT = Path(__file__).resolve().parent

def model_score(model_name, data, data_train):
    HP = Optimal_Uni_algo_HP_dict[model_name] if data.shape[1] == 1 else Optimal_Multi_algo_HP_dict[model_name]
    HP = dict(HP)
    if model_name in CHECKPOINTS:
        HP['checkpoint'] = str(ROOT.parent / CHECKPOINTS[model_name])
    if model_name in Semisupervise_AD_Pool:
        out = run_Semisupervise_AD(model_name, data_train, data, **HP)
    else:
        out = run_Unsupervise_AD(model_name, data, **HP)
    return np.asarray(out).ravel()

# morphology → primary model + alternatives (rank-weighted)
ROUTING = {
    'level_shift':   {'Syntropic': 0.5, 'OmniAnomaly': 0.2, 'Time_RCD': 0.2, 'Sub_PCA': 0.1},
    'spike':         {'OmniAnomaly': 0.5, 'Syntropic': 0.3, 'Time_RCD': 0.2},
    'dense_cluster': {'OmniAnomaly': 0.5, 'Syntropic': 0.3, 'Sub_PCA': 0.2},
    'flatline':      {'Syntropic': 0.5, 'OmniAnomaly': 0.3, 'Time_RCD': 0.2},
    'variance_change': {'Syntropic': 0.4, 'OmniAnomaly': 0.3, 'Time_RCD': 0.3},
}
SOURCE_MORPH = {'NAB':'level_shift','IOPS':'level_shift','SMAP':'level_shift','SMD':'level_shift',
                'Stock':'spike','YAHOO':'spike','TODS':'dense_cluster','UCR':'flatline',
                'MGAB':'level_shift','NEK':'variance_change','Power':'variance_change','SED':'spike',
                'MSL':'level_shift','PSM':'variance_change','SWaT':'variance_change','WSD':'level_shift'}

CASES = [
    ('NAB-001','level_shift','Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'),
    ('NAB-008','level_shift','Datasets/TSB-AD-U/008_NAB_id_8_Synthetic_tr_1007_1st_2734.csv'),
    ('TODS','dense_cluster','Datasets/TSB-AD-U/287_TODS_id_1_Synthetic_tr_500_1st_11.csv'),
    ('YAHOO','spike','Datasets/TSB-AD-U/551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv'),
    ('Stock','spike','Datasets/TSB-AD-U/149_Stock_id_1_Finance_tr_500_1st_7.csv'),
    ('UCR','flatline','Datasets/TSB-AD-U/303_UCR_id_1_Medical_tr_2499_1st_8000.csv'),
    ('IOPS','level_shift','Datasets/TSB-AD-U/260_IOPS_id_1_WebService_tr_5000_1st_11804.csv'),
    ('SMAP','level_shift','Datasets/TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv'),
]

def eval_route(case_name, morph, f):
    df = pd.read_csv(f).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    train_index = Path(f).stem.split('_')[-3]
    data_train = data[:int(train_index), :]
    weights = ROUTING.get(morph, {'Syntropic': 0.5, 'OmniAnomaly': 0.5})
    ranks, ws = [], []
    for m, w in weights.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                s = model_score(m, data, data_train)
            s = np.nan_to_num(np.asarray(s, float), nan=0.0)
            ranks.append(rankdata(s) / len(s)); ws.append(w)
        except Exception as e:
            print(f'  [{m}] FAIL {str(e)[:60]}')
    if not ranks:
        return None
    ws = np.array(ws); ws = ws / ws.sum()
    fused = sum(w * r for w, r in zip(ws, ranks))
    r = get_metrics(fused, label, slidingWindow=sw)
    return r['Affiliation-F'] * 100

if __name__ == '__main__':
    print('=== morphology routing + adaptive ensemble (8-case Affiliation-F) ===')
    vals = []
    for name, morph, f in CASES:
        v = eval_route(name, morph, f)
        vals.append(v)
        print(f'{name:10s} [{morph:14s}] routed Aff-F = {v:.2f}' if v else f'{name:10s} FAIL')
    vals = [v for v in vals if v is not None]
    print(f'\nrouting mean Affiliation-F = {np.mean(vals):.2f}')
    print('comparison: Syntropic≈90.24, Time_RCD≈86.77, best single model (per-case oracle)≈97.76')
