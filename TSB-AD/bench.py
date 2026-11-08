"""Run multiple models × multiple files, output official Affiliation-F / VUS-PR / AUC, and cache the scores."""
import sys, os, io, contextlib, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
import numpy as np, pandas as pd, torch, random
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

seed = 2024
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

CHECKPOINTS = {'Time_RCD': 'Time-RCD/best_model'}
ROOT = Path(__file__).resolve().parent

def run_model(model_name, filename):
    df = pd.read_csv(filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    train_index = Path(filename).stem.split('_')[-3]
    data_train = data[:int(train_index), :]
    HP = Optimal_Uni_algo_HP_dict[model_name] if data.shape[1] == 1 else Optimal_Multi_algo_HP_dict[model_name]
    HP = dict(HP)
    if model_name in CHECKPOINTS:
        HP['checkpoint'] = str(ROOT.parent / CHECKPOINTS[model_name])
    if model_name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(model_name, data_train, data, **HP)
    elif model_name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(model_name, data, **HP)
    else:
        raise ValueError(model_name)
    m = get_metrics(np.asarray(output).ravel(), label, slidingWindow=slidingWindow)
    return m

CASES = [
    ('NAB-001', 'Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'),
    ('NAB-008', 'Datasets/TSB-AD-U/008_NAB_id_8_Synthetic_tr_1007_1st_2734.csv'),
    ('TODS', 'Datasets/TSB-AD-U/287_TODS_id_1_Synthetic_tr_500_1st_11.csv'),
    ('YAHOO', 'Datasets/TSB-AD-U/551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv'),
    ('Stock', 'Datasets/TSB-AD-U/149_Stock_id_1_Finance_tr_500_1st_7.csv'),
    ('UCR', 'Datasets/TSB-AD-U/303_UCR_id_1_Medical_tr_2499_1st_8000.csv'),
    ('IOPS', 'Datasets/TSB-AD-U/260_IOPS_id_1_WebService_tr_5000_1st_11804.csv'),
    ('SMAP', 'Datasets/TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv'),
]

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='Syntropic,Time_RCD,IForest')
    ap.add_argument('--cases', default='all')
    args = ap.parse_args()
    models = args.models.split(',')
    cache = ROOT / 'bench_cache.json'
    if cache.exists():
        results = json.load(open(cache))
    else:
        results = {}
    for m in models:
        results.setdefault(m, {})
        for name, f in CASES:
            if name in results[m]:
                continue
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    r = run_model(m, f)
                results[m][name] = {'Aff-F': round(r['Affiliation-F']*100, 2), 'VUS-PR': round(r['VUS-PR']*100, 2), 'AUC': round(r['AUC-ROC']*100, 2), 'StdF1': round(r['Standard-F1']*100, 2)}
                print(f'[{m}] {name}: Aff-F={results[m][name]["Aff-F"]} VUS={results[m][name]["VUS-PR"]} AUC={results[m][name]["AUC"]} ({time.time()-t0:.0f}s)', flush=True)
            except Exception as e:
                results[m][name] = {'err': str(e)[:80]}
                print(f'[{m}] {name}: FAIL {str(e)[:80]}', flush=True)
            json.dump(results, open(cache, 'w'), indent=1)
    print('\n=== Affiliation-F matrix ===')
    names = [c[0] for c in CASES]
    print(f'{"model":12s} ' + ' '.join(f'{n:>9s}' for n in names))
    for m in models:
        row = ' '.join(f'{results[m].get(n,{}).get("Aff-F",0):9.2f}' for n in names)
        print(f'{m:12s} {row}')
