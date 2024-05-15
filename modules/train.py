import torch
from torch.utils.data import DataLoader

from models.teacher_network import TeacherNetwork
from .data.synthetic_dataset import ModelNetDataset
from .data.modelnet10_data import SYNTHETIC_DATA_PATH


train_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH / 'train')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

test_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH / 'test')
test_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

model = TeacherNetwork(d_model=64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index, tb_writer):
    """Trains the model for a single epoch and logs the results to tensorboard."""
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

