from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer(ABC):
    """Handles common training logic such as model saving, logging, epoch iteration, etc."""
    def __init__(self, name, num_epochs, device, train_dataloader, test_dataloader):
        self.num_epochs = num_epochs
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model_path = Path(f"models/{name}/{datetime.now().isoformat()}")
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter()
        self.optimizer = None

    def run(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for step, batch in tqdm(enumerate(self.train_dataloader), desc='Training'):
                total_loss = self.predict_and_score(batch)
                epoch_loss += total_loss.detach()
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            epoch_loss /= len(self.train_dataloader)

            print(f'Epoch Loss: {epoch_loss}')
            self.writer.add_scalar('Epoch Loss', epoch_loss, epoch)

            if epoch % 25 == 0 and epoch is not 0:
                self.save_models(tag=epoch)
                with torch.no_grad():
                    val_loss = sum(self.predict_and_score(batch) for batch in self.test_dataloader)
                    val_loss /= len(self.test_dataloader)
                    print(f'Validation Loss: {val_loss}')
                    self.writer.add_scalar('Validation Loss', val_loss, epoch)

        self.save_models(tag='last')

    @abstractmethod
    def save_models(self, tag) -> None:
        ...

    @abstractmethod
    def predict_and_score(self, batch: torch.Tensor) -> torch.Tensor:
        ...
