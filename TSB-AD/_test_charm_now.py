import os, sys, time, contextlib, io
os.environ['CHARM_BASE_URL'] = 'http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080'
os.environ['CHARM_API_KEY'] = 'token'
sys.path.insert(0, '.')
sys.path.insert(0, 'benchmark_exp')
import numpy as np, pandas as pd
from Run_CHARM import run_CHARM_kNN
f = '010_NAB_id_10_WebService_tr_500_1st_271.csv'
df = pd.read_csv(f'Datasets/TSB-AD-U/{f}').dropna()
data = df.iloc[:,0:-1].values.astype(float)
ti = int(f.split('_')[-3])
t0 = time.time()
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sc = run_CHARM_kNN(data[:ti], data)
    print('OK len=%d took %.1fs' % (len(sc), time.time()-t0), flush=True)
except Exception as e:
    print('FAIL took %.1fs: %s' % (time.time()-t0, str(e)[:120]), flush=True)
