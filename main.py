import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
import torch.utils.data
import torchinfo


# TODO: add dropout to network.
# TODO: add batch normalization to network.
# TODO: periodic boundaries: in network or in input, but before the first convolution >1 over the spatial dimensions.
# TODO: implement zero padding variations for comparing long and short patterns.
# TODO: data augmentation: double amount of data if we can assume symmetric left and right on top/bottom of the arm.
# TODO: implement chamfer similarity calculation.

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, combinations, similarities, pattern_dir, num_frames_long, num_frames_short):
        if len(combinations) != len(similarities):
            raise ValueError('Number of combinations and similarity scores do not match.')

        self.combinations = combinations
        self.similarities = similarities
        self.pattern_dir = pattern_dir

        self.num_frames_long = num_frames_long
        self.num_frames_short = num_frames_short
        self.num_single_shifts = num_frames_long - num_frames_short + 1
        self.num_shifts = self.num_single_shifts ** 2

    def __len__(self):
        return len(self.similarities)*self.num_shifts


    def __getitem__(self, idx):
        i = idx // self.num_shifts

        pattern0_path = os.path.join(self.pattern_dir, 'p{}.npy'.format(self.combinations[i][0]))
        pattern1_path = os.path.join(self.pattern_dir, 'p{}.npy'.format(self.combinations[i][1]))

        pattern0 = np.load(pattern0_path)
        pattern1 = np.load(pattern1_path)

        length0 = np.shape(pattern0)[1]
        length1 = np.shape(pattern1)[1]

        which_shift = idx % self.num_shifts
        which_single_shift = which_shift % self.num_single_shifts
        which_single_shift_other = which_shift // self.num_single_shifts

        if (length0 == self.num_frames_long) and (length1 == self.num_frames_long):
            # No change necessary for 0 and 1.
            pattern0 = pattern0
            pattern1 = pattern1
        elif (length0 == self.num_frames_long) and (length1 == self.num_frames_short):
            # No change necessary for 0.
            pattern0 = pattern0

            pattern1_base = np.zeros_like(pattern0)
            pattern1_base[:,which_single_shift:(which_single_shift+self.num_frames_short),:,:] = pattern1
            pattern1 = pattern1_base
        elif (length0 == self.num_frames_short) and (length1 == self.num_frames_long):
            # No change necessary for 1.
            pattern1 = pattern1

            pattern0_base = np.zeros_like(pattern1)
            pattern0_base[:,which_single_shift:(which_single_shift+self.num_frames_short),:,:] = pattern0
            pattern0 = pattern0_base
        elif (length0 == self.num_frames_short) and (length1 == self.num_frames_short):
            pattern0_base = np.zeros((1,self.num_frames_long,6,4))
            pattern0_base[:,which_single_shift:(which_single_shift+self.num_frames_short),:,:] = pattern0
            pattern0 = pattern0_base

            pattern1_base = np.zeros((1,self.num_frames_long,6,4))
            pattern1_base[:,which_single_shift_other:(which_single_shift_other+self.num_frames_short),:,:] = pattern1
            pattern1 = pattern1_base
        else:
            raise ValueError('Length should be either short or long.')

        similarity = self.similarities[i]

        return pattern0, pattern1, similarity


class NeuralNetwork(torch.nn.Module):
    def __init__(self, network_option, num_frames):
        super(NeuralNetwork, self).__init__()
        self.network_option = network_option

        self.layers_individual = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(8, 16, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
        )

        if self.network_option == 'embedding':
            self.layers_individual_embedding = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(16 * round(num_frames / 2) * 6 * 4, 100),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(100, 25),
            )
            self.pairwise_distance = torch.nn.PairwiseDistance()
        elif self.network_option == 'image_compare':
            self.layers_combined = torch.nn.Sequential(
                torch.nn.Conv3d(16, 32, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * round(num_frames / 4) * 6 * 4, 25),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(25, 10),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, 1),
                torch.nn.Sigmoid()
            )
        else:
            raise ValueError('Network option does not exist.')

    def forward(self, x0, x1):
        x0 = self.layers_individual(x0)
        x1 = self.layers_individual(x1)

        if self.network_option == 'image_compare':
            x = torch.sub(x0, x1)

            # To ensure the commutative property of the input.
            x = torch.abs(x)
            x = self.layers_combined(x)
            x = torch.squeeze(x)
        elif self.network_option == 'embedding':
            x0 = self.layers_individual_embedding(x0)
            x1 = self.layers_individual_embedding(x1)

            x0 = torch.nn.functional.normalize(x0)
            x1 = torch.nn.functional.normalize(x1)

            distance = self.pairwise_distance(x0, x1)

            # Maximum Euclidean distance of two normalized vectors is two, therefore the distance is divided by two.
            x = 1 - distance / 2
        else:
            raise ValueError('Network option does not exist.')

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
    # Totally random patterns.
    num_frames_long = 80
    num_frames_short = 72

    if (num_frames_long % 8 != 0) or (num_frames_short % 8 != 0):
        raise ValueError('Number of frames should be evenly divisible be 8.')

    pattern0 = np.random.randint(0, 255, size=(1, num_frames_long, 6, 4)) / 255
    pattern1 = np.random.randint(0, 255, size=(1, num_frames_short, 6, 4)) / 255
    pattern2 = pattern0
    pattern3 = pattern0
    pattern4 = pattern1

    np.save('data/p0.npy', pattern0)
    np.save('data/p1.npy', pattern1)
    np.save('data/p2.npy', pattern2)
    np.save('data/p3.npy', pattern3)
    np.save('data/p4.npy', pattern4)

    combinations = [(0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (1, 3), (1, 4),
                    (2, 3), (2, 4),
                    (3, 4)]
    similarities = [0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    1.0, 0.0,
                    0.0]

    pattern_dir = 'data'

    dataset = CustomImageDataset(combinations, similarities, pattern_dir, num_frames_long, num_frames_short)

    train_val_test_split = [0.6, 0.2, 0.2]

    if np.sum(train_val_test_split) != 1:
        raise ValueError('Train/validation/test split incorrect.')

    num_data_points = len(dataset)
    test_size = int(train_val_test_split[2] * num_data_points)
    validation_size = int(train_val_test_split[1] * num_data_points)
    train_size = num_data_points - test_size - validation_size

    # TODO: Maybe not completely independent set because of the switching of lengths.
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

    network_option = 'image_compare'

    model = NeuralNetwork(network_option=network_option, num_frames=num_frames_long).to(device)

    torchinfo.summary(model, input_size=[(batch_size, 1, num_frames_long, 6, 4), (batch_size, 1, num_frames_long, 6, 4)],
                      verbose=2)

    # More robust to outliers than mean squared error.
    loss_fn = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 50
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

    model_name = f'model_{network_option}.pth'
    torch.save(model.state_dict(), model_name)
    model = NeuralNetwork(network_option=network_option, num_frames=num_frames_long)
    model.load_state_dict(torch.load(model_name))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    errors = []
    with torch.no_grad():
        for x0, x1, true_value in test_dataloader:
            prediction = model(x0.float(), x1.float())
            prediction_item = prediction.item()
            true_value_item = true_value.item()
            print(f'Prediction: {prediction_item:.2f}, true value: {true_value_item:.2f}')
            errors.append(prediction_item - true_value_item)

    fig, ax = plt.subplots()
    ax.hist(errors, bins=200, range=(-1, 1))
    ax.set_xlabel('Errors')
    ax.set_ylabel('Number of occurences')
    plt.show()


if __name__ == '__main__':
    main()
