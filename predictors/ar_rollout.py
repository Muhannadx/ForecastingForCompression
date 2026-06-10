from pathlib import Path
import xarray
import numpy as np
import pickle
from aurora import Batch, Metadata
from aurora import AuroraPretrained
import torch
import os
import dataclasses
import copy
from cmp import my_compress, decompress
import time
import pandas as pd
import pickle
import sys


e_user = int(sys.argv[1])

MAX_STEPS = 200

results_dir = 'aurora/'

pred_dir = os.path.join(results_dir, f'e-{e_user}')
print(f'finite eb=1e-{e_user}, with compression')
print(f'path for saving results: {pred_dir}')
    
os.makedirs(pred_dir, exist_ok=True)


short_to_long = {'2t':'2m_temperature' , '10u':'10m_u_component_of_wind', '10v':'10m_v_component_of_wind', 'msl':'mean_sea_level_pressure',
't':'temperature', 'u':'u_component_of_wind', 'v':'v_component_of_wind', 'q':'specific_humidity', 'z':'geopotential'}
long_to_short = {v: k for k, v in short_to_long.items()}

org_names = {'2t':'2m_temperature' , '10u':'10m_u_component_of_wind', '10v':'10m_v_component_of_wind', 'msl':'mean_sea_level_pressure',
            't':'temperature', 'u':'u_component_of_wind', 'v':'v_component_of_wind', 'q':'specific_humidity', 'z':'geopotential'}


os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

torch.use_deterministic_algorithms(True)
        
SURFACE_VARS = [
 '2m_temperature',
 '10m_u_component_of_wind',
 '10m_v_component_of_wind',
 'mean_sea_level_pressure',
]
ATM_VARS = [
 'temperature',
 'u_component_of_wind',
 'v_component_of_wind',
 'specific_humidity',
 'geopotential'
 ]
#computed ranges for each variable at each level (max - min) across the dataset
with open('./summ_stats.pkl', 'rb') as f:
    stats_dict = pickle.load(f)
#obtained from the Aurora model and stored as a pickle for easy loading (refer to Aurora's documentation)
with open('./aurora-0.25-static.pickle', 'rb') as f:
    static_vars_ds = pickle.load(f)


def _squeeze_np(t):
    """torch tensor or ndarray -> squeezed numpy array (float32 for tensors)."""
    if isinstance(t, torch.Tensor):
        return t.float().numpy(force=True).squeeze()
    return np.asarray(t).squeeze()


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
        


def my_rollout_mod(model: AuroraPretrained, batch: Batch, steps: int, ds=None, eb=1e-2, quant_only=True, overwrite=True):
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    

    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    
    for i in range(steps):
            start_time = time.time()
            df_entries = []
            pred = model.forward(batch)
            pred = pred.to('cpu')
        
            org = ds.isel(time=[i+2,]).load()
            # create a modifiable version
            mod_surf_vars = {}
            quant_surf_vars = {}
            for k, v in pred.surf_vars.items():
                # v: [B, T, H, W]
                org_var = copy.deepcopy(org[org_names[k]].values)[:, :, :-1, :]
                v_range = stats_dict[org_names[k]]
                cmp_size, cr, dec, quant = compress_eb(org=org_var, 
                                                          pred=v.clone().numpy(), 
                                                          eb=eb, 
                                                          file_name=f'{pred_dir}/ar.cmp', 
                                                          v_range=v_range, 
                                                          )

                if overwrite:
                    mod_surf_vars[k] = torch.from_numpy(dec)
                else:
                    mod_surf_vars[k] = v.clone()
                quant_surf_vars[k] = quant
                
                entry = dict()
                entry['variable'] = f'{org_names[k]}'
                entry['timestep'] = i
                entry['level'] = -1
                entry['hPa'] = -1
                entry['eb'] = eb
                entry['abs_eb'] = eb*v_range
                entry['compressor'] = 1
                entry['org_size'] = int(org_var.size * org_var.itemsize)
                entry['cmp_size'] = cmp_size
                entry['CR'] = cr
                df_entries.append(entry.copy())

            mod_atmos_vars = {}
            quant_atmos_vars = {}
            for k, v in pred.atmos_vars.items():
                # v: [B, T, L, H, W]
                levels = []
                quant_levels = []
                for level in range(v.shape[2]):
                    org_var = copy.deepcopy(org[org_names[k]].values)[:, :, [level,], :-1, :]
                    
                    l = L_index[ds.level.values[level]]
                    v_range = stats_dict[org_names[k]][l]
                    cmp_size, cr, dec, quant = compress_eb(org=org_var, 
                                                          pred=v[:, :, [level,], :, :].clone().numpy(),
                                                          eb=eb, 
                                                          file_name=f'{pred_dir}/ar.cmp',
                                                          v_range=v_range, 
                                                          )
                   
                    if overwrite:
                        levels.append(torch.from_numpy(dec))
                    else:
                        levels.append(v[:, :, [level,], :, :].clone())
                    quant_levels.append(quant)
                    

                    entry = dict()
                    entry['variable'] = f'{org_names[k]}'
                    entry['timestep'] = i
                    entry['level'] = l
                    entry['hPa'] = int(ds.level.values[level])
                    entry['eb'] = eb
                    entry['abs_eb'] = eb*v_range
                    entry['compressor'] = 1
                    entry['org_size'] = int(org_var.size * org_var.itemsize)
                    entry['cmp_size'] = cmp_size
                    entry['CR'] = cr
                    df_entries.append(entry.copy())

                mod_atmos_vars[k] = torch.cat(levels, dim=2)  # [B, T, L, H, W]
                quant_atmos_vars[k] = np.concatenate(quant_levels, axis=2)  # [B, T, L, H, W]
                
    
            modified_pred = dataclasses.replace(pred,
                                                surf_vars=mod_surf_vars, 
                                                atmos_vars=mod_atmos_vars,
                                               )
            end_time = time.time()
            elapsed = end_time - start_time
        
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"[{i+1}/{steps}]: Execution time: {mins} min(s) {secs} sec(s)")
            
            ts = org.time.values[0]
            pred_np = {'timestamp': ts}
            mod_np = {'timestamp': ts}
            quant_np = {'timestamp': ts}
            for k in pred.surf_vars:
                pred_np[org_names[k]] = _squeeze_np(pred.surf_vars[k])
                mod_np[org_names[k]] = _squeeze_np(mod_surf_vars[k])
                quant_np[org_names[k]] = _squeeze_np(quant_surf_vars[k])
            for k in pred.atmos_vars:
                pred_np[org_names[k]] = _squeeze_np(pred.atmos_vars[k])
                mod_np[org_names[k]] = _squeeze_np(mod_atmos_vars[k])
                quant_np[org_names[k]] = _squeeze_np(quant_atmos_vars[k])

            yield pred_np, mod_np, df_entries, quant_np

            modified_pred = modified_pred.to(p.device)
    
            # Prepare for next step
            batch = dataclasses.replace(
                modified_pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], modified_pred.surf_vars[k]], dim=1)
                    for k in modified_pred.surf_vars
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], modified_pred.atmos_vars[k]], dim=1)
                    for k in modified_pred.atmos_vars
                },
            )

def get_ds():
    ds = ds = xarray.open_zarr("/ERA/ERA5.zarr", consolidated=True,)
    LEVELS_IDX = ds.level.values
    ds = ds.sel(level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    ds = ds.sortby('lat', False)
    return ds, LEVELS_IDX




print('loading data...')
ds, LEVELS_IDX = get_ds()
L_index = {int(v): i for i, v in enumerate(LEVELS_IDX)}
N = ds.time.size
print(f'Data loaded. N = {N} time steps, rolling out for {N-2} steps.')
print('preparing data...')
surf_vars_ds = ds.isel(batch=0, time=np.arange(3))[SURFACE_VARS].load()
atmos_vars_ds = ds.isel(batch=0, time=np.arange(3))[ATM_VARS].load()
batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["2m_temperature"].values[:2][None]),
        "10u": torch.from_numpy(surf_vars_ds["10m_u_component_of_wind"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["10m_v_component_of_wind"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["mean_sea_level_pressure"].values[:2][None]),
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"]),
        "slt": torch.from_numpy(static_vars_ds["slt"]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["temperature"].values[:2][None]),
        "u": torch.from_numpy(atmos_vars_ds["u_component_of_wind"].values[:2][None]),
        "v": torch.from_numpy(atmos_vars_ds["v_component_of_wind"].values[:2][None]),
        "q": torch.from_numpy(atmos_vars_ds["specific_humidity"].values[:2][None]),
        "z": torch.from_numpy(atmos_vars_ds["geopotential"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.lat.values),
        lon=torch.from_numpy(surf_vars_ds.lon.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element. Select element 1, corresponding to time
        # 06:00.
        time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
    ),
)

print('loading model...')
model = AuroraPretrained()
model.load_checkpoint()
model.eval()

print('starting rollout now...')
model = model.to("cuda")
c = 0

EB = 10**-e_user
df_entries = []
with torch.inference_mode():
    for pred, modified_pred, df_entry, quant_dict in my_rollout_mod(model, batch, eb=EB, steps=min(MAX_STEPS, N-2), ds=ds, quant_only=True, overwrite=True):
        with open(f'{pred_dir}/pred_raw_{c}.pkl', 'wb') as f:
            pickle.dump(pred, f)
            
        with open(f'{pred_dir}/p{c}.pkl', 'wb') as f:
            pickle.dump(modified_pred, f)

        with open(f'{pred_dir}/quant_{c}.pkl', 'wb') as f:
            pickle.dump(quant_dict, f)
            
        df_entries = df_entries + df_entry
        c+=1

        df = pd.DataFrame.from_records(df_entries)
        df.to_csv(f'{pred_dir}/ar_results_e-{e_user}.csv', index=False)
    
model = model.to("cpu")
