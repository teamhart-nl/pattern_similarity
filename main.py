import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional
import torch.utils.data
import torchinfo
import torchvision.transforms


# TODO: add dropout to network.
# TODO: add batch normalization to network.
# TODO: periodic boundaries: in network or in input, but before the first convolution >1 over the spatial dimensions.
# TODO: data augmentation: double amount of data if we can assume symmetric left and right on top/bottom of the arm.
# TODO: implement chamfer similarity calculation.

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, similarity_scores, pattern_dir, transform):
        self.patterns0_all = similarity_scores['pattern0'].to_numpy()
        self.patterns1_all = similarity_scores['pattern1'].to_numpy()
        self.similarities = similarity_scores['similarity'].to_numpy()

        self.pattern_dir = pattern_dir

        self.transform = transform

    def __len__(self):
        return len(self.similarities)

    def __getitem__(self, idx):
        pattern0_path = os.path.join(self.pattern_dir, 'p_{}.npy'.format(self.patterns0_all[idx]))
        pattern1_path = os.path.join(self.pattern_dir, 'p_{}.npy'.format(self.patterns1_all[idx]))

        pattern0 = np.load(pattern0_path)
        pattern1 = np.load(pattern1_path)

        pattern0 = self.transform(pattern0)
        pattern1 = self.transform(pattern1)

        similarity = self.similarities[idx]

        return pattern0, pattern1, similarity


class RandomPadding:
    def __init__(self, num_frames_max, height, width):
        self.num_frames_max = num_frames_max
        self.height = height
        self.width = width

    def __call__(self, sample):
        sample_shape = np.shape(sample)

        num_frames = sample_shape[1]

        assert sample_shape[0] == 1
        assert sample_shape[2] == self.height
        assert sample_shape[3] == self.width

        if num_frames == self.num_frames_max:
            pattern = sample
        else:
            assert num_frames <= self.num_frames_max

            pattern = np.zeros((1, self.num_frames_max, self.height, self.width))

            shift_max = self.num_frames_max - num_frames

            shift = random.randint(0, shift_max)
            pattern[:, shift:(shift + num_frames), :, :] = sample

        return pattern


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_frames):
        super(NeuralNetwork, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(8, 16, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * num_frames * 6 * 4, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 25),
        )

        self.pairwise_distance = torch.nn.PairwiseDistance()

    def forward(self, x0, x1):
        x0 = self.layers(x0)
        x1 = self.layers(x1)

        x0 = torch.nn.functional.normalize(x0)
        x1 = torch.nn.functional.normalize(x1)

        distance = self.pairwise_distance(x0, x1)

        # Maximum Euclidean distance of two normalized vectors is two, therefore the distance is divided by two.
        x = 1 - distance / 2

        return x


def train(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader)
    model.train()
    training_loss = 0
    for batch, (x0, x1, true_value) in enumerate(dataloader):
        x0 = x0.float().to(device)
        x1 = x1.float().to(device)
        true_value = true_value.float().to(device)

        prediction = model(x0, x1)

        loss = loss_fn(prediction, true_value)
        training_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_loss /= num_batches

    return training_loss


def validate(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for x0, x1, true_value in dataloader:
            x0 = x0.float().to(device)
            x1 = x1.float().to(device)
            true_value = true_value.float().to(device)

            prediction = model(x0, x1)

            loss = loss_fn(prediction, true_value)
            validation_loss += loss.item()

    validation_loss /= num_batches

    return validation_loss


def main():
    num_frames_max = 80

    # Example similarity scores from VISIL model.
    similarity_scores = pd.read_csv('data/similarity_scores.csv')

    pattern_dir = 'data/patterns/'

    dataset = CustomImageDataset(similarity_scores, pattern_dir,
                                 transform=torchvision.transforms.Compose([RandomPadding(num_frames_max, 6, 4)]))

    train_val_test_split = [0.6, 0.2, 0.2]

    if np.sum(train_val_test_split) != 1:
        raise ValueError('Train/validation/test split incorrect.')

    num_data_points = len(dataset)
    test_size = int(train_val_test_split[2] * num_data_points)
    validation_size = int(train_val_test_split[1] * num_data_points)
    train_size = num_data_points - test_size - validation_size

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                    [train_size, validation_size,
                                                                                     test_size])

    print(f'Num data points: {len(dataset)}')
    print(f'Num train data points: {len(train_dataset)}')
    print(f'Num validation data points: {len(validation_dataset)}')
    print(f'Num test data points: {len(test_dataset)}')

    batch_size = 512

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    for x0, x1, true_value in train_dataloader:
        print(f'Shape of p0: {x0.shape}')
        print(f'Shape of p1: {x1.shape}')
        print(f'Shape of true value: {true_value.shape} with type {true_value.dtype}')
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork(num_frames=num_frames_max).to(device)

    torchinfo.summary(model,
                      input_size=[(batch_size, 1, num_frames_max, 6, 4), (batch_size, 1, num_frames_max, 6, 4)],
                      verbose=2)

    # More robust to outliers than mean squared error.
    loss_fn = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 20
    epochs = np.arange(1, num_epochs + 1)
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        training_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        validation_loss = validate(validation_dataloader, model, loss_fn, device)

        training_losses[epoch] = training_loss
        validation_losses[epoch] = validation_loss

        print(f'Epoch {epoch + 1}, training loss: {training_loss:5f}, validation loss: {validation_loss:5f}')

    fig, ax = plt.subplots()
    ax.plot(epochs, training_losses, label='Training')
    ax.plot(epochs, validation_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average loss')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.show()

    model_name = 'model.pth'
    torch.save(model.state_dict(), model_name)
    model = NeuralNetwork(num_frames=num_frames_max)
    model.load_state_dict(torch.load(model_name))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    errors = []
    with torch.no_grad():
        for x0, x1, true_value in test_dataloader:
            prediction = model(x0.float(), x1.float())
            prediction_item = prediction.item()
            true_value_item = true_value.item()
            errors.append(prediction_item - true_value_item)

    fig, ax = plt.subplots()
    ax.hist(errors, range=(-1, 1))
    ax.set_xlabel('Errors')
    ax.set_ylabel('Number of occurences')
    plt.show()


if __name__ == '__main__':
    main()
