from pathlib import Path
import xarray
import numpy as np
import pickle
import os
from cmp import my_compress, decompress
import time
import pandas as pd
import pickle
import sys


e_user = int(sys.argv[1])

results_dir = '/cra5/'

pred_dir = os.path.join(results_dir, f'e-{e_user}')
print(f'finite eb=1e-{e_user}, with compression')
print(f'path for saving results: {pred_dir}')
    
os.makedirs(pred_dir, exist_ok=True)
        
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

def compress_eb(org, pred, eb, v_range, file_name):    
    org_shape = pred.shape

    # compress org using pred as the predictor
    cmp_size, quant = my_compress(org.flatten(), pred.flatten(), eb, file_name,
                                  v_range, return_quant=True)
    # quantized residual codes (before Huffman/zstd).
    cr_residual = org.nbytes / cmp_size
    dec = decompress(file_name, org.flatten(), pred.flatten(), eb*v_range).reshape(org_shape)
    quant = quant.reshape(org_shape)

    return cmp_size, cr_residual, dec, quant
        


def get_ds():
    ds = xarray.open_zarr("/ERA/ERA5.zarr", consolidated=True,)
    LEVELS_IDX = ds.level.values
    ds = ds.sortby('lat', False)
    return ds, LEVELS_IDX


print('loading data...')
DS, LEVELS_IDX = get_ds()
L_index = {int(v): i for i, v in enumerate(LEVELS_IDX)}
N = 200
print(f'Data loaded. N = {N} time steps...')


print('starting now...')
c = 0
EB = 10**-e_user
df_entries = []
for t in range(2, N):
    start_time = time.time()
    org_slice = DS.isel(time=[t]).compute()
    bounded_res = {'timestamp': org_slice.time.values[0]}
    quant_res = {'timestamp': org_slice.time.values[0]}
    with open(f'/CRA5/p{t}.pkl', 'rb') as f:
        pred_slice = pickle.load(f)
    
    for v in SURFACE_VARS:
        org_var = org_slice[v].values
        pred_var = pred_slice[v]
        v_range = stats_dict[v]
        cmp_size, cr, dec, quant = compress_eb(org=org_var, 
                                                  pred=pred_var.copy(), 
                                                  eb=EB, 
                                                  file_name=f'{pred_dir}/cra5.cmp', 
                                                  v_range=v_range, 
                                                  )
        entry = dict()
        entry['variable'] = f'{v}'
        entry['timestep'] = t-2
        entry['level'] = -1
        entry['hPa'] = -1
        entry['eb'] = EB
        entry['abs_eb'] = EB*v_range
        entry['compressor'] = 'CRA'
        entry['org_size'] = int(org_var.size * org_var.itemsize)
        entry['cmp_size'] = cmp_size
        entry['CR'] = cr
        df_entries.append(entry.copy())
        bounded_res[v] = dec.squeeze()
        quant_res[v] = quant.squeeze()

    for v in ATM_VARS:
        levels = []
        quant_levels = []
        for level in range(len(org_slice.level.values)):
            org_var = org_slice[v].isel(level=level).squeeze().values
            pred_var = pred_slice[v][level, ...].squeeze()
            v_range = stats_dict[v][level]
            cmp_size, cr, dec, quant = compress_eb(org=org_var, 
                                                      pred=pred_var.copy(),
                                                      eb=EB, 
                                                      file_name=f'{pred_dir}/cra5.cmp',
                                                      v_range=v_range, 
                                                      )
            levels.append(dec.squeeze())
            quant_levels.append(quant.squeeze())
            l = L_index[DS.level.values[level]]
            
            entry = dict()
            entry['variable'] = f'{v}'
            entry['timestep'] = t-2
            entry['level'] = l
            entry['hPa'] = int(DS.level.values[level])
            entry['eb'] = EB
            entry['abs_eb'] = EB*v_range
            entry['compressor'] = 'CRA'
            entry['org_size'] = int(org_var.size * org_var.itemsize)
            entry['cmp_size'] = cmp_size
            entry['CR'] = cr
            df_entries.append(entry.copy())
            
        bounded_res[v] = np.stack(levels)
        quant_res[v] = np.stack(quant_levels)

    with open(f'{pred_dir}/p{t-2}.pkl', 'wb') as f:
        pickle.dump(bounded_res, f)

    with open(f'{pred_dir}/quant_{t-2}.pkl', 'wb') as f:
        pickle.dump(quant_res, f)

    end_time = time.time()
    elapsed = end_time - start_time

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"[{t+1}/{N}]: Execution time: {mins} min(s) {secs} sec(s)")

    df = pd.DataFrame.from_records(df_entries)

    df.to_csv(f'{pred_dir}/cr_results_e-{e_user}.csv', index=False)