from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    local_dir="/media/ptthang/Expansion/data/CT-RATE/dataset",
)
