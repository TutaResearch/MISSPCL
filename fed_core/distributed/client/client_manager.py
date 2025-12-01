import logging
from abc import abstractmethod

from mpi4py import MPI

from ..communication.message import Message
from ..communication.mpi.com_manager import MpiCommunicationManager
from ..communication.observer import Observer


class ClientManager(Observer):
    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        self.args = args
        self.size = size
        self.rank = rank
        self.backend = backend

        if backend == "MPI":
            self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="client")
        else:
            self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="client")

        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        # logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
        #     self.rank, str(msg_type), str(msg_params.get_content())))
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        msg = Message(message.get_type(),
                      message.get_sender_id(), message.get_receiver_id())
        msg.add(Message.MSG_ARG_KEY_TYPE, message.get_type())
        msg.add(Message.MSG_ARG_KEY_SENDER, message.get_sender_id())
        msg.add(Message.MSG_ARG_KEY_RECEIVER, message.get_receiver_id())
        for key, value in message.get_params().items():
            # logging.info("%s == %s" % (key, value))
            msg.add(key, value)
        logging.info("Sending message (type %d) to server" % message.get_type())
        self.com_manager.send_message(msg)
        for key, value in msg.get_params().items():
            # logging.info("%s == %s" % (key, value))
            message.add(key, value)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish client")
        if self.backend == "MPI":
            MPI.COMM_WORLD.Abort()
        # elif self.backend == "MQTT":
        #     self.com_manager.stop_receive_message()
        # elif self.backend == "MQTT_S3":
        #     logging.info("MQTT_S3")
        #     # self.com_manager.stop_receive_message()
        # elif self.backend == "GRPC":
        #     self.com_manager.stop_receive_message()
        # elif self.backend == "TRPC":
        #     self.com_manager.stop_receive_message()
