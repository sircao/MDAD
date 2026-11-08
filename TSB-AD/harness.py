"""Official protocol benchmark harness: aligned with Run_Detector_U.py."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make import TSB_AD available
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))  # hurst and other stubs
import pandas as pd, numpy as np, torch, random
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

seed = 2024
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

CHECKPOINTS = {
    'Time_RCD': 'Time-RCD/best_model/pretrain_checkpoint_best_uni.pth',
    'TimeRCD_MAFT': 'Time-RCD/best_model/pretrain_checkpoint_best_uni.pth',
}

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
        HP['checkpoint'] = os.path.join(os.path.dirname(__file__), '..', CHECKPOINTS[model_name])
        HP['checkpoint'] = os.path.abspath(HP['checkpoint'])
    if model_name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(model_name, data_train, data, **HP)
    elif model_name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(model_name, data, **HP)
    else:
        raise ValueError(model_name)
    m = get_metrics(output, label, slidingWindow=slidingWindow)
    return m, slidingWindow
