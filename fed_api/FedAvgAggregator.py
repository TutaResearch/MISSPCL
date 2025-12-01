import numpy as np
import logging
from torch.distributions.laplace import Laplace

class FedAVGAggregator(object):
    def __init__(self, client_num, device,
                 args, global_model):
        self.global_model = global_model
        self.args = args
        self.client_num = client_num
        self.device = device

        self.model_dict = dict()
        self.sample_num_dict = dict()

        # map for whether the client has uploaded the model
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.global_model.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.global_model.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self, round_idx):
        local_model_list = []
        training_num = 0

        for idx in range(self.client_num):
            if self.model_dict[idx] is not -1:
                local_model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                training_num += self.sample_num_dict[idx]

        logging.info("valid models = {} in round {}".format(len(local_model_list), round_idx))

        (num0, averaged_params) = local_model_list[0]
        
        for k in averaged_params.keys():
            # for all clients
            for i in range(0, len(local_model_list)):
                local_sample_number, local_model_params = local_model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
            laplace_dist = Laplace(loc=0, scale=1e-2/self.args.eps)
            noise = laplace_dist.sample(averaged_params[k].shape)
            averaged_params[k] += noise
        
        # update the global model parameters
        self.set_global_model_params(averaged_params)
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes
