import logging
import os
import setproctitle
import wandb
from tqdm import tqdm
from fed_core.trainer.model_trainer import ModelTrainer


class MyModel(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def train(self, train_loader, val_loader, device, helper, args=None):
        model = self.model

        if args.checkpoint:
            helper.load_checkpoint(model, args.checkpoint)

        if not args.no_train:
            for epoch in tqdm(range(args.epochs)):

                logging.info("{}->|-------train: {}-------|".format(setproctitle.getproctitle(), epoch))
                scores = model.train_epoch(train_loader)
                for key in scores:
                    helper.log(f"metrics/train/{key}", scores[key])
                # helper.save_checkpoint(model, "last.ckpt")

                logging.info("{}->|-------val: {}-------|".format(setproctitle.getproctitle(), epoch))
                scores, _ = model.eval_epoch(val_loader, bootstrap=False)
                for key in scores.keys():
                    helper.log(f"metrics/val/{key}", scores[key])
                helper.save_checkpoint_if_best(model, "best-{}.ckpt".format(self.id), scores)

                if not args.official_run:
                    break


    def test(self, test_loader, device, helper, args=None):
        model = self.model.to(device)
        helper.load_checkpoint(model, os.path.join(helper.model_saved_path, "best-{}.ckpt".format(self.id)))
        logging.info(f"{setproctitle.getproctitle()}->|-------final test-------|")
        scores, predictions = model.eval_epoch(test_loader, bootstrap=True)
        wandb.log(scores)
        for key in scores.keys():
            helper.log(f"metrics/final_test/{key}", scores[key])

