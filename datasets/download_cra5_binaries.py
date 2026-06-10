from huggingface_hub import hf_hub_download
import time
from datetime import datetime, timedelta

# Starting date and number of time steps to download
start_date = "2014-01-01T00:00:00"
N = 2001  # Number of 6-hour time steps to download

repo_id = "taohan10200/CRA5-Dataset"
repo_type = "dataset"
local_dir = "./data/CRA5"

current_date = datetime.fromisoformat(start_date)
for i in range(N):
    timestamp = current_date.strftime("%Y-%m-%dT%H:%M:%S")
    year = current_date.strftime("%Y")
    filename = f"{year}/{timestamp}.bin"

    print(f"Downloading {filename} ({i+1}/{N})...")
    start_time = time.time()
    local_bin_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=local_dir,
    )
    elapsed = time.time() - start_time
    print(f"Downloaded to: {local_bin_path}, Time taken: {elapsed:.2f} seconds")

    current_date += timedelta(hours=6)
