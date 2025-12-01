import sys
import setproctitle
import os
import wandb

sys.path.append(".")
sys.path.append("..")
sys.path.append("../core")
sys.path.append("../dataset")
sys.path.append("../encoder")

os.environ["WANDB_PROJECT"] = "MissPCL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset.tokenizer import MIMIC4Tokenizer
from torch.utils.data import DataLoader
from core.model import MMLBackbone
from dataset.utils import mimic4_collate_fn
from fed_experiments.utils import client_data_path, load_pickle
from helper import Helper
from fed_api.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fed_api.FedAvgAPI import FED_init, FED_FedAvg_distributed


def add_args(parser):
    # FL settings
    parser.add_argument("--client_num_in_total", type=int, default=5, metavar="NN",help="number of workers in a distributed cluster")
    parser.add_argument("--client_num_per_round", type=int, default=5, metavar="NN", help="number of workers")
    parser.add_argument("--client_optimizer", type=str, default="Adam", help="SGD; Adam")
    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")
    parser.add_argument("--comm_round", type=int, default=20, help="how many round of communications we should use")
    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients."
    )
    parser.add_argument("--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file")
    # Training settings
    parser.add_argument("--monitor", type=str, default="pr_auc", help="for binary classification")
    parser.add_argument("--dataset", type=str, default="adni")
    parser.add_argument("--task", type=str, default="y")
    parser.add_argument("--dev", action="store_true", default=False)
    parser.add_argument("--load_no_label", type=bool, default=False)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--code_pretrained_embedding", type=bool, default=True)
    parser.add_argument("--code_layers", type=int, default=2)
    parser.add_argument("--code_heads", type=int, default=2)
    parser.add_argument("--bert_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="GRU")
    parser.add_argument("--rnn_bidirectional", type=bool, default=True)
    parser.add_argument("--ffn_layers", type=int, default=2)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_norm", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--monitor_criterion", type=str, default="max")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_train", type=bool, default=False)
    parser.add_argument("--note", type=str, default="mml_v19")
    parser.add_argument("--exp_name_attr", type=list, default=["dataset", "task", "note"])
    parser.add_argument("--official_run", action="store_true", default=True)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--eps", type=int, default=1)
    # missing settings
    parser.add_argument("--rate_list", type=list, default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    parser.add_argument("--thres", type=float, default=1.0, help="clients with >= this threshold will be trained for fc only; set 1.0 train with missing data")
    args = parser.parse_args()
    return args

def main_func():
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FED_init()

    # parse arguments
    helper = Helper(add_args, process_id)
    args = helper.args

    # wandb init for client
    if process_id:
        wandb_run = wandb.init(name='Client-{}'.format(str(process_id), reinit=True))

    # customize the process name
    pre_name = "FL-Thread:"
    str_process_name = pre_name + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # GPU mapping
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key
    )

    if process_id:
        # load data for client
        if args.dataset == "mimic4":
            train_set = load_pickle(os.path.join(client_data_path, f"mimic4/task:{args.task}/train_set_{process_id-1}_of_{worker_number-2}-{args.rate_list[process_id-1]}miss.pkl"))
            val_set = load_pickle(os.path.join(client_data_path, f"mimic4/task:{args.task}/val_set_{process_id-1}_of_{worker_number-2}-{args.rate_list[process_id-1]}miss.pkl"))
            test_set = load_pickle(os.path.join(client_data_path, f"mimic4/task:{args.task}/test_set_{process_id-1}_of_{worker_number-2}-{args.rate_list[process_id-1]}miss.pkl"))
            args.num_classes = 1
            collate_fn = mimic4_collate_fn
            tokenizer = train_set.tokenizer
        elif args.dataset == "adni":
            train_set = load_pickle(os.path.join(client_data_path, f"adni/task:{args.task}/train_set_{process_id-1}_of_{worker_number-2}-{args.rate_list[process_id-1]}miss.pkl"))
            val_set = load_pickle(os.path.join(client_data_path, f"adni/task:{args.task}/val_set_{process_id-1}_of_{worker_number-2}-{args.rate_list[process_id-1]}miss.pkl"))
            test_set = load_pickle(os.path.join(client_data_path, f"adni/task:{args.task}/test_set_{process_id-1}_of_{worker_number-2}-{args.rate_list[process_id-1]}miss.pkl"))
            args.num_classes = 3
            collate_fn = None
            tokenizer = None
        else:
            raise ValueError("Dataset not supported!")
        # create data loader
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            shuffle=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            shuffle=False
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            shuffle=False
        )
    else:
        # for server
        train_loader = None
        val_loader = None
        test_loader = None
        tokenizer = MIMIC4Tokenizer() if args.dataset == "mimic4" else None

    # 
    miss_flag = False if not process_id else args.rate_list[process_id-1] >= args.thres

    # create model backbone
    # initialize global model weights for the first run
    model = MMLBackbone(args, tokenizer, miss_flag).to(args.device)
    # logging.info(model)
    # logging.info("{}->Number of parameters: {}".format(setproctitle.getproctitle(), count_parameters(model)))

    # start distributed training
    FED_FedAvg_distributed(
        process_id, worker_number, device, helper, comm, args,
        model, train_loader, val_loader, test_loader, miss_flag
        )

    if process_id:
        wandb_run.finish()

if __name__ == "__main__":
    main_func()
