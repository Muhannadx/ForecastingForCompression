# datasets

Scripts to obtain the two inputs the evaluation needs: the raw ERA5 fields we
compress, and the compressed CRA5 binaries used by the CRA5 predictor.

## Planned contents

```
datasets/
├── download_era5.py            script for downlaoding the ERA5 data
└── download_cra5_binaries.py   script for downlaoding the CRA5 binaries
```

## ERA5

ERA5 is a global reanalysis dataset from ECMWF. We use the 0.25 degree product on the
721 x 1440 latitude-longitude grid, sampled every 6 hours to match the native lead
time of the forecasting models.

We treat each pressure level of each atmospheric field as its own variable, so every
variable is a 3D tensor with its own value range `r_X`.

The nine variables common to all three models:

- Surface (4): 2-meter temperature, 10-meter u-wind, 10-meter v-wind, mean sea level
  pressure.
- Atmospheric (5): temperature, geopotential, u-wind, v-wind, specific humidity.

Level coverage differs by model: Aurora uses 13 pressure levels (50 to 1000 hPa),
GraphCast and SZ3.1 use 37. GraphCast and SZ3.1 additionally cover vertical velocity,
total precipitation and 6hr total precipitation. 
The per-model coverage is recorded in the appendex.


## CRA5 binaries

CRA5 compresses the ERA5 archive to a learned latent bitstream. The CRA5 predictor
decodes these binaries to form its reconstruction, which then enters the shared
error-bounded back-end. `download_cra5_binaries.py` records where to fetch the
released binaries.
