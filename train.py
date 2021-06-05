import fire
import tqdm

import copy
import logging
import torch as T


logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self):
        self.data_loaders = None
        self.model = None
        self.optimizer = None
        self.criterion = None

    def _setup_data(self, path):
        from dataset import build_augmentations, get_dataset

        augs = build_augmentations()
        self.data_loaders = get_dataset(path, augs["train"])
    
    def _setup_model(self):
        from model import get_model

        self.model = get_model()
        self.model.to("cuda")

    def _setup_optimizer(self):
        from torch.optim import Adam, lr_scheduler
        
        self.optimizer = Adam(self.model.parameters(),lr=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def _setup_criterion(self):
        from torch.nn import BCEWithLogitsLoss

        self.criterion = BCEWithLogitsLoss()

    def _step(self, stage):
        correct_counts = 0
        running_loss = 0.0
        running_acc = 0.0

        for inputs, targets in tqdm.tqdm(self.data_loaders[stage]):
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")

            self.optimizer.zero_grad()

            predict = self.model(inputs).squeeze(-1)

            loss = self.criterion(predict, targets.float())

            if stage == "train":
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            classes = T.round(T.sigmoid(predict))

            correct_counts += T.sum(classes == targets.data)

        epoch_loss = running_loss / len(self.data_loaders[stage].dataset)

        epoch_acc = correct_counts.double() / len(self.data_loaders[stage].dataset)

        return {
            "loss": epoch_loss,
            "acc": epoch_acc
        }

    def _train_epoch(self):
        self.model.train()

        return self._step("train")

    def _eval_epoch(self):
        self.model.eval()

        with T.no_grad():
            return self._step("eval")

    def train(self, dataset_path, epochs):
        self._setup_data(dataset_path)
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()

        best_acc = -1.0
        best_loss = 1e4

        best_model = None

        for e in tqdm.tqdm(range(epochs)):
            out = self._train_epoch()

            logging.info(f"epoch train: {e} loss {out['loss']} acc {out['acc']}")

            out = self._eval_epoch()

            logging.info(f"epoch eval: {e} loss {out['loss']} acc {out['acc']}")

            if best_acc < out["acc"]:
                logging.info(f"updating model from loss {best_loss} to {out['loss']}, acc {best_acc} to {out['acc']}")

                best_loss = out["loss"]
                best_acc = out["acc"]

                best_model = copy.deepcopy(self.model)

        T.save(best_model.state_dict(), "./best_model.pth")
        T.save(quantized.state_dict(), "./best_model_qint8.pth")

        logging.info(f"training finished with the best loss {best_loss}")
           

if __name__ == "__main__":
    fire.Fire(Trainer)
    

