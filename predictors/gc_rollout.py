import dataclasses
import functools
from typing import Optional
from google.cloud import storage

from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization

from graphcast import xarray_jax
from graphcast import xarray_tree

from graphcast import rollout
from graphcast import autoregressive

import haiku as hk
import jax
import numpy as np
import xarray
import gc
import glob
import pickle
from cmp import my_compress, decompress
import warnings
import time
import os
import pandas as pd
import pickle
import sys

e_user = int(sys.argv[1])

results_dir = '/graphcast/'


pred_dir = os.path.join(results_dir, f'e-{e_user}')
print(f'finite eb=1e-{e_user}, with compression')
print(f'path for saving results: {pred_dir}')
    
os.makedirs(pred_dir, exist_ok=True)


# (Lustre-specific) background thread so the transfer overlaps the next step.
import shutil
from concurrent.futures import ThreadPoolExecutor
local_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp', f'gc_e-{e_user}')
os.makedirs(local_dir, exist_ok=True)
cmp_file = os.path.join(local_dir, 'gc.cmp') 
_pool = ThreadPoolExecutor(max_workers=2)
def _ship(name):
    shutil.move(os.path.join(local_dir, name), os.path.join(pred_dir, name))


TARGET_VARS = ['10m_u_component_of_wind',
 '10m_v_component_of_wind',
 '2m_temperature',
 'geopotential',
 'mean_sea_level_pressure',
 'specific_humidity',
 'temperature',
 'total_precipitation_6hr',
 'u_component_of_wind',
 'v_component_of_wind',
 'vertical_velocity']

def extract_and_prepare(ds, ind):
    batch = ds
    batch['geopotential_at_surface'] = batch['geopotential_at_surface'][0].squeeze()
    batch['land_sea_mask'] = batch['land_sea_mask'][0].squeeze()
    x = batch.time.values
    time_coord = (x - x[0])
    datetime_coord = x.reshape(1, -1)
    batch = batch.assign_coords(time=time_coord, datetime=(("batch", "time"), datetime_coord))
    batch.attrs = {}
    return batch

def get_data(length=1):
    ds = xarray.open_zarr("/ERA/ERA5.zarr", consolidated=True,)
    return ds


gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"


# pickled modelto avoid download every time;
# same checkpoint used for GC paper evals
with open('./gc_large.pkl', 'rb') as f:
    ckpt = pickle.load(f)
    
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config


#saved stats for normalization; obtained with model
with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()

#computed ranges for each variable at each level (max - min) across the dataset
with open('./summ_stats.pkl', 'rb') as f:
    stats_dict = pickle.load(f)


# @title Build jitted functions, and possibly initialize random weights
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
    
  # predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

# loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
# grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


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


ds = get_data(300)

single_level_vars = []
multi_level_vars = []
for i in ds[TARGET_VARS].data_vars:
    if 'level' in ds[i].dims:
        multi_level_vars.append(i)
    else:
        single_level_vars.append(i)


eval_steps=1
MAX_STEPS = 200
N = len(ds.time.values) - 3
N = min(MAX_STEPS, N)
df_entries = []
eb = 10**-e_user
print(f'N={N} steps for rollout')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*The return type of `Dataset.dims` will be changed.*")
    # preds = []
    # test_batch = example_batch.copy(deep=True)
    curr_tgt_idx = 2
    current_batch = ds.isel(time=np.arange(curr_tgt_idx-2, curr_tgt_idx+1)).compute().copy(deep=True)
    for i in range(N):
        start_time = time.time()
        print(f'[{i+1}/{N}]: Predicting for t={curr_tgt_idx} --> {current_batch['datetime'].values}')
        
        eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        current_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config))
        
        p = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets,
            forcings=eval_forcings,
            verbose=True
                                            )

        new_pred = p.copy(deep=True)

        if e_user != 0:
            print('compressing...')
            quant_entry = {'timestamp': current_batch['datetime'].values[-1]}
            for v in single_level_vars:
                v_range = stats_dict[v]
                original = current_batch[v][dict(time=[-1, ])]
                prediction = p[v].isel(time=[0, ])
                
                cmp_size, cr, dec, quant = compress_eb(org=original.copy(deep=True), pred=prediction.copy(deep=True), eb=eb, file_name=cmp_file, v_range=v_range)
                entry = dict()
                entry['variable'] = f'{v}'
                entry['timestep'] = i
                entry['level'] = -1
                entry['hPa'] = -1
                entry['eb'] = eb
                entry['abs_eb'] = eb*v_range
                entry['compressor'] = 0
                entry['org_size'] = int(original.values.size * original.values.itemsize)
                entry['cmp_size'] = cmp_size
                entry['CR'] = cr
                df_entries.append(entry.copy())
                new_pred[v][dict(time=[0, ])] = dec
                quant_entry[v] = quant.squeeze().astype(np.int16)
                
                
            for v in multi_level_vars:
                quant_levels = []
                for l in range(len(ds.level)):
                    v_range = stats_dict[v][l]
                    original = current_batch[v].isel(time=[-1,], level=l)
                    prediction = p[v].isel(time=[0,], level=l)
                    
                    cmp_size, cr, dec, quant = compress_eb(org=original.copy(deep=True), pred=prediction.copy(deep=True), eb=eb, file_name=cmp_file, v_range=v_range)
                    entry = dict()
                    entry['variable'] = f'{v}'
                    entry['timestep'] = i
                    entry['level'] = l
                    entry['hPa'] = int(ds.level.values[l])
                    entry['eb'] = eb
                    entry['abs_eb'] = eb*v_range
                    entry['compressor'] = 0
                    entry['org_size'] = int(original.values.size * original.values.itemsize)
                    entry['cmp_size'] = cmp_size
                    entry['CR'] = cr
                    df_entries.append(entry.copy())
                    new_pred[v][dict(time=[0, ], level=[l, ])] = dec
                    quant_levels.append(quant.squeeze().astype(np.int16))
                quant_entry[v] = np.stack(quant_levels, axis=0)

            # (1) raw prediction and (2) quantized residual codes for this step
            pred_entry = {'timestamp': current_batch['datetime'].values[-1]}
            for tv in TARGET_VARS:
                pred_entry[tv] = p[tv].values.squeeze()
            with open(os.path.join(local_dir, f'pred_raw_{i}.pkl'), 'wb') as f:
                pickle.dump(pred_entry, f)
            with open(os.path.join(local_dir, f'quant_{i}.pkl'), 'wb') as f:
                pickle.dump(quant_entry, f)
            _pool.submit(_ship, f'pred_raw_{i}.pkl')
            _pool.submit(_ship, f'quant_{i}.pkl')

        df = pd.DataFrame.from_records(df_entries)
        df.to_csv(f'{pred_dir}/gc_results_e-{e_user}.csv', index=False)

        res_entry = dict()
        res_entry['timestamp'] = current_batch['datetime'].values[-1]
        for v in TARGET_VARS:
            res_entry[v] = new_pred[v].values.squeeze()
            
        with open(os.path.join(local_dir, f'p{i}.pkl'), 'wb') as f:
            pickle.dump(res_entry, f)
        _pool.submit(_ship, f'p{i}.pkl')

        res_entry = dict()
        
        for v in TARGET_VARS:
            current_batch[v][dict(time=[-1, ])] = new_pred[v].values

        curr_tgt_idx+=1
        current_batch = current_batch.isel(time=slice(1, None))
        new_entry = ds.isel(time=[curr_tgt_idx, ]).compute().copy(deep=True)
        current_batch = xarray.concat([current_batch, new_entry], dim='time')
        current_batch['geopotential_at_surface'] = current_batch['geopotential_at_surface'][0].squeeze()
        current_batch['land_sea_mask'] = current_batch['land_sea_mask'][0].squeeze()
        

        del p, new_pred, eval_inputs, eval_targets, eval_forcings, res_entry, new_entry, original, prediction
        gc.collect()

        if i % 100 == 0:
            jax.clear_caches()

        end_time = time.time()
        elapsed = end_time - start_time
        
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"Execution time: {mins} min(s) {secs} sec(s)")

_pool.shutdown(wait=True)
print('all background transfers complete')



    
