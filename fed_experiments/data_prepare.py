import sys
import os
import argparse
# from tqdm import tqdm
from memory_profiler import profile
import logging
sys.path.append(".")
sys.path.append("..")
sys.path.append("../core")
sys.path.append("../dataset")
sys.path.append("../encoder")

from dataset.adni_dataset import ADNIDataset
from dataset.mimic4_dataset import MIMIC4Dataset
from fed_experiments.utils import client_data_path, dump_pickle

# @profile
def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_num", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mimic4")
    parser.add_argument("--task", type=str, default="mortality")
    parser.add_argument("--dev", action="store_true", default=False)
    parser.add_argument("--load_no_label", type=bool, default=False)
    parser.add_argument("--rate_list", type=list, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # parser.add_argument("--rate_list", type=list, default=[0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])
    # parser.add_argument("--rate_list", type=list, default=[0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9])
    args = parser.parse_args()

    print("=========constructing data for clients...=========")

    # client data distribution
    data_map = [1.0 / (args.worker_num)] * (args.worker_num)

    for i in range(args.worker_num): 
        # construct data for client
        if args.dataset == "mimic4":
            train_set = MIMIC4Dataset(split="train", client_idx=i, num_map=data_map, 
            task=args.task, dev=args.dev, load_no_label=args.load_no_label, miss_rate=args.rate_list[i])
            val_set = MIMIC4Dataset(split="val", client_idx=i, num_map=data_map,  task=args.task, miss_rate=args.rate_list[i])
            test_set = MIMIC4Dataset(split="test", client_idx=i, num_map=data_map,  task=args.task, miss_rate=args.rate_list[i])
            out_dir = os.path.join(client_data_path, f"mimic4/task:{args.task}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            dump_pickle(train_set, os.path.join(out_dir, f"train_set_{i}_of_{args.worker_num-1}-{args.rate_list[i]}miss.pkl"))
            dump_pickle(val_set, os.path.join(out_dir, f"val_set_{i}_of_{args.worker_num-1}-{args.rate_list[i]}miss.pkl"))
            dump_pickle(test_set, os.path.join(out_dir, f"test_set_{i}_of_{args.worker_num-1}-{args.rate_list[i]}miss.pkl"))
        elif args.dataset == "adni":
            train_set = ADNIDataset(split="train", client_idx=i, num_map=data_map,  task=args.task, dev=args.dev, load_no_label=args.load_no_label, miss_rate=args.rate_list[i])
            val_set = ADNIDataset(split="val", client_idx=i, num_map=data_map,  task=args.task, miss_rate=args.rate_list[i])
            test_set = ADNIDataset(split="test", client_idx=i, num_map=data_map,  task=args.task, miss_rate=args.rate_list[i])
            out_dir = os.path.join(client_data_path, f"adni/task:{args.task}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            dump_pickle(train_set, os.path.join(out_dir, f"train_set_{i}_of_{args.worker_num-1}-{args.rate_list[i]}miss.pkl"))
            dump_pickle(val_set, os.path.join(out_dir, f"val_set_{i}_of_{args.worker_num-1}-{args.rate_list[i]}miss.pkl"))
            dump_pickle(test_set, os.path.join(out_dir, f"test_set_{i}_of_{args.worker_num-1}-{args.rate_list[i]}miss.pkl"))
        else:
            raise ValueError("Dataset not supported!")

    print("=========data constructing complete...=========")

if __name__ == "__main__":
    main_func()