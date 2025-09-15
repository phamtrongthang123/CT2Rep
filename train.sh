source .venv/bin/activate  # if uv created a venv

TOTAL_NODES=1
MASTER_ADDR=localhost
GPUS_PER_NODE=1
CURRENT_RANK=0
MASTER_PORT=12340

cd CT2Rep
torchrun --nnodes=${TOTAL_NODES} --nproc-per-node=${GPUS_PER_NODE} --master-port=${MASTER_PORT} \
    --master-addr ${MASTER_ADDR} --node-rank=${CURRENT_RANK} \
    main.py --max_seq_length 300 --threshold 10 --epochs 100 --save_dir results/test_ct2rep/ --step_size 1 --gamma 0.8 --batch_size 1 --d_vf 512 --xlsxfile ../example_data/CT2Rep/data_reports_example.xlsx --trainfolder ../example_data/CT2Rep/train --validfolder ../example_data/CT2Rep/valid
