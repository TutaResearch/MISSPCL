class FedAVGTrainer(object):
    """
    Wrapper class for the client model
    """
    def __init__(self, client_index, train_loader, val_loader, test_loader, 
                 device, helper, args, local_model):
        self.client_index = client_index
        self.device = device
        self.args = args
        self.helper = helper

        self.local_model = local_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.local_sample_number = len(train_loader)

    def update_model(self, global_weights):
        """
        update the local model with the global model
        """
        self.local_model.set_model_params(global_weights)

    def train(self, round_idx):
        self.args.round_idx = round_idx
        self.local_model.train(self.train_loader, self.val_loader, self.device, self.helper, self.args)
        weights = self.local_model.get_model_params()
        return weights, self.local_sample_number

    def test(self):
        self.local_model.test(self.test_loader, self.device, self.helper, self.args)
