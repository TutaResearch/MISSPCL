from mpi4py import MPI
from .FedAvgAggregator import FedAVGAggregator
from .FedAvgTrainer import FedAVGTrainer
from .FedAvgClient import FedAVGClient
from .FedAvgServer import FedAVGServer
from .MyModel import MyModel


def FED_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FED_FedAvg_distributed(
        process_id, worker_number, device, helper, comm, args,
        model, train_loader, val_loader, test_loader, miss_flag,
        ):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model)
    else:
        init_client(
            args, device, helper, comm, process_id, worker_number, model,
            train_loader, val_loader, test_loader, miss_flag
            )


def init_server(args, device, comm, rank, size, model):
    global_model = MyModel(model)
    global_model.set_id(-1)

    worker_num = size - 1
    aggregator = FedAVGAggregator(
        worker_num,
        device,
        args,
        global_model
    )

    server_proxy = FedAVGServer(args, aggregator, comm, rank, size, args.backend)
    server_proxy.send_init_msg()
    server_proxy.run()


def init_client(
        args, device, helper, comm, process_id, size, model,
        train_loader, val_loader, test_loader, miss_flag
        ):
    local_model = MyModel(model)
    client_index = process_id - 1
    local_model.set_id(client_index)

    local_trainer = FedAVGTrainer(
        client_index,
        train_loader, val_loader, test_loader,
        device,
        helper,
        args,
        local_model
    )

    client_proxy = FedAVGClient(args, local_trainer, comm, process_id, size, args.backend, miss_flag)
    client_proxy.run()
