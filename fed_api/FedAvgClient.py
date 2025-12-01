from .message_define import MyMessage
from fed_core.distributed.client.client_manager import ClientManager
from fed_core.distributed.communication.message import Message
import logging
import setproctitle

class FedAVGClient(ClientManager):
    def __init__(self, args, local_trainer, comm=None, rank=0, size=0, backend="MPI", miss_flag=False):
        super().__init__(args, comm, rank, size, backend)
        self.local_trainer = local_trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.miss_flag = miss_flag

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.local_trainer.update_model(global_model_params)

        # the beginning of entire training
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.local_trainer.update_model(model_params)

        self.round_idx += 1
        self.__train()

        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        if self.miss_flag and self.round_idx <= int(self.num_rounds / 2):
        # if self.miss_flag and self.round_idx <= int(2):
            logging.info("Client {} not train both local and global on round {}".format(setproctitle.getproctitle(), self.round_idx))
            self.send_model_to_server(0, -1, -1)
            return
        weights, local_sample_num = self.local_trainer.train(self.round_idx)
        self.local_trainer.test()
        self.send_model_to_server(0, weights, local_sample_num)
