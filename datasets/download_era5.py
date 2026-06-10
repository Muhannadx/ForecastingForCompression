import xarray as xr
import numpy as np

singlelevelfields = [
                             '2m_temperature',
                             'mean_sea_level_pressure',
                             '10m_v_component_of_wind',
                             '10m_u_component_of_wind',
                             'total_precipitation',
                             'surface_pressure',
                        ]
    
pressurelevelfields = [
                        'temperature',
                         'geopotential',
                         'u_component_of_wind',
                         'v_component_of_wind',
                         'vertical_velocity',
                         'specific_humidity'
                         'total_precipitation',
                    ]


ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks={'time':24},
    storage_options=dict(token='anon'),
)

ds = ds[singlelevelfields+pressurelevelfields]

ds = ds.sel(time=slice('2014-01-01T00:00:00', None))
ds.to_zarr('./data/ERA5/ERA5.zarr', mode='w')