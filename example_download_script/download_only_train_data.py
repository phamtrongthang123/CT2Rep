import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

repo_id = "ibrahimhamamci/CT-RATE"
directory_name = "dataset/train_fixed/"

data = pd.read_csv("train_labels.csv")

for name in tqdm(data["VolumeName"], total=len(data["VolumeName"])):
    try:
        folder1 = name.split("_")[0]
        folder2 = name.split("_")[1]
        folder = folder1 + "_" + folder2
        folder3 = name.split("_")[2]
        subfolder = folder + "_" + folder3
        subfolder = directory_name + folder + "/" + subfolder

        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            subfolder=subfolder,
            filename=name,
            local_dir="data_volumes",
        )
    except:
        pass
