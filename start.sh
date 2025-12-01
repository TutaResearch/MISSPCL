#!/usr/bin/env bash
mkdir -p ./client_data

CLIENT_NUM=8
WORKER_NUM=8

PROCESS_NUM=`expr $WORKER_NUM + 1`

# update the gpu mapping configuration
if [ -f "gpu_mapping.yaml" ]; then
    sed -i "s/\[\s*0,\s*[0-9]\+\s*\]/[ 0, $PROCESS_NUM ]/" gpu_mapping.yaml
    echo "gpu mapping configuration update successfully:"
    cat gpu_mapping.yaml
else
    echo "gpu_mapping.yaml file not exists."
fi

hostname > mpi_host_file

# mimic mortality / readmission
# data partition
python ./data_prepare.py --worker_num $WORKER_NUM --dataset "mimic4" --task "mortality"
# start federated learning (multiple processes for parallel computing)
mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main.py --client_num_in_total $CLIENT_NUM --client_num_per_round $WORKER_NUM --dataset "mimic4" --task "mortality" --thres 0.8 --official_run

# data partition
# python ./data_prepare.py --worker_num $WORKER_NUM --dataset "adni" --task "y"
# start federated learning (multiple processes for parallel computing)
# mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main.py --client_num_in_total $CLIENT_NUM --client_num_per_round $WORKER_NUM --dataset "adni" --monitor "auc_macro_ovo" --task "y" --thres 1.1 --official_run
