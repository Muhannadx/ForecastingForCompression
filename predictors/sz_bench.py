import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import xarray

from pysz import sz, szConfig, szErrorBoundMode

MAX_TIMESTEPS = 202  # cap the outer (time) loop


e_user = int(sys.argv[1])
eb = 10 ** -e_user

results_dir = '/sz/'
pred_dir = os.path.join(results_dir, f'e-{e_user}')
os.makedirs(pred_dir, exist_ok=True)
print(f'finite eb=1e-{e_user}, with compression')
print(f'path for saving results: {pred_dir}')

config = szConfig()
config.errorBoundMode = szErrorBoundMode.ABS

SURFACE_VARS = [
 '2m_temperature',
 '10m_u_component_of_wind',
 '10m_v_component_of_wind',
 'mean_sea_level_pressure',
 '100m_u_component_of_wind',
 '100m_v_component_of_wind',
 'surface_pressure',
 'total_cloud_cover',
 'total_precipitation',
]
ATM_VARS = [
 'temperature',
 'u_component_of_wind',
 'v_component_of_wind',
 'specific_humidity',
 'geopotential',
 'vertical_velocity',
 ]

#computed ranges for each variable at each level (max - min) across the dataset
with open('./summ_stats.pkl', 'rb') as f:
    stats_dict = pickle.load(f)


def get_ds():
    ds = xarray.open_zarr("/ERA/ERA5.zarr", consolidated=True,)
    return ds, ds.level.values


def compress_field(data, v, t, k, hPa, v_range):
    abs_eb = eb * v_range
    config.absErrorBound = abs_eb
    compressed, ratio, quant = sz.compress(data, config, return_quant_inds=True)
    decompressed, _ = sz.decompress(compressed, np.float32, data.shape)
    # one index per value,  field shape
    quant = np.asarray(quant).reshape(data.shape)   
    entry = {
        'variable': v,
        'timestep': t-2,
        'level': k,
        'hPa': hPa,
        'eb': eb,
        'abs_eb': abs_eb,
        'compressor': 'SZ',
        'org_size': data.size * data.itemsize,
        'cmp_size': len(compressed),
        'CR': ratio,
    }
    return entry, decompressed, quant


ds, LEVELS = get_ds()
N = min(len(ds.time), MAX_TIMESTEPS)
print(f'running {N} of {len(ds.time)} timesteps')

df_entries = []
for t in range(2, N+2):
    start_time = time.time()
    print(f'[{t + 1}/{N}]: compressing t={t} --> {ds.isel(time=[t]).time.values[0]}')

    curr_ds = ds.isel(time=[t]).load()
    ts = {'timestamp': ds.isel(time=[t]).time.values[0]}  # reconstructions
    qs = {'_levels_hPa': np.asarray(LEVELS),
          'timestamp': ds.isel(time=[t]).time.values[0]}              # quantized residual indices

    # ---- surface variables ----
    for v in SURFACE_VARS:
        data = curr_ds[v].values
        v_range = stats_dict[v]
        entry, recon, quant = compress_field(data, v, t, k=-1, hPa=-1, v_range=v_range)
        df_entries.append(entry)
        ts[v] = recon.squeeze()
        qs[v] = quant.squeeze()          # (H, W)

    # ---- atmospheric variables (per pressure level) ----
    for v in ATM_VARS:
        recons, quants = [], []
        for k, l in enumerate(LEVELS):
            data = curr_ds[v].sel(level=[l]).values
            v_range = stats_dict[v][k]
            entry, recon, quant = compress_field(data, v, t, k=k, hPa=l, v_range=v_range)
            df_entries.append(entry)
            recons.append(recon)
            quants.append(quant.squeeze())     # (H, W)
        ts[v] = np.concatenate(recons, 2).squeeze()
        qs[v] = np.stack(quants)               # (n_levels, H, W), row k == LEVELS[k]

    # ---- persist results for this timestep ----
    pd.DataFrame.from_records(df_entries).to_csv(
        f'{pred_dir}/sz_results_e-{e_user}.csv', index=False)

    with open(f'{pred_dir}/p{t-2}.pkl', 'wb') as f:
        pickle.dump(ts, f)

    with open(f'{pred_dir}/quant_{t-2}.pkl', 'wb') as f:
        pickle.dump(qs, f)

    elapsed = time.time() - start_time
    print(f"Execution time: {int(elapsed // 60)} min(s) {int(elapsed % 60)} sec(s)")
    print('=' * 88)
