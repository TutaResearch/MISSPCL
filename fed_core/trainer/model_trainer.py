from abc import ABC, abstractmethod


class ModelTrainer(ABC):


    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, val_data, device, helper, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, helper, args=None):
        pass
