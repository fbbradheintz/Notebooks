import argparse
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms



# constants
MODEL_SAVEFILE = 'seedling'


# helpers
def tlog(msg):
    print('{}   {}'.format(time.asctime(), msg))


def save_model(model, epoch, model_dir):
    tlog('Saving model')
    savefile = "{}-e{}-{}.pt".format(MODEL_SAVEFILE, epoch, int(time.time()))
    path = os.path.join(model_dir, savefile)
    # recommended way from https://pytorch.org/docs/stable/notes/serialization.html
    torch.save(model.state_dict(), path)
    return savefile


def get_transforms(target_size=100, normalize=False):
    t = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    if normalize: # for imagenet-trained models specifically
        t = transforms.Compose([
            t,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return t



class SeedlingModelV1_1(nn.Module):
    def __init__(self):
        super(SeedlingModelV1_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(model, epochs, lr, model_dir, device):
    tlog('Training the model...')
    tlog('working on {}'.format(device))
    
    best_accuracy = 0. # determines whether we save a copy of the model
    saved_model_filename = None
    
    model = model.to(device) # move to GPU if available
    loss_fn = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss() for classification tasks
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):
        tlog('BEGIN EPOCH {} of {}'.format(epoch + 1, epochs))
        running_loss = 0. # bookkeeping
        
        tlog('Train:')
        for i, data in enumerate(train_loader):
            instances, labels = data[0], data[1]
            instances, labels = instances.to(device), labels.to(device) # move to GPU if available
            
            optimizer.zero_grad()
            guesses = model(instances)
            loss = loss_fn(guesses, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 200 == 0: # log every 200 batches
                tlog('  batch {}   avg loss: {}'.format(i + 1, running_loss / (200)))
                running_loss = 0.
        
        tlog('Validate:')
        with torch.no_grad(): # no need to do expensive gradient computation for validation
            total_loss = 0.
            correct = 0
            
            for i, data in enumerate(validate_loader):
                instance, label = data[0], data[1]
                instance, label = instance.to(device), label.to(device) # move to GPU if available
                
                guess = model(instance)
                loss = loss_fn(guess, label)
                total_loss += loss.item()
                
                prediction = torch.argmax(guess, 1)
                if prediction.item() == label.item(): # assuming batch size of 1
                    correct += 1

            avg_loss = total_loss / len(validate_loader)
            accuracy = correct / len(validate_loader)
            tlog('  Avg loss for epoch: {}   accuracy: {}'.format(avg_loss, accuracy))
            
            if accuracy >= best_accuracy:
                tlog( '  New accuracy peak, saving model')
                best_accuracy = accuracy
                saved_model_filename = save_model(model, epoch + 1, model_dir)
                
    return (saved_model_filename, best_accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--use-cuda', type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()
    print(args)

    if args.use_cuda and torch.cuda.is_available():
        dev = torch.device('cuda')
        print('GPU ready to go!')
    else:
        dev = torch.device('cpu')
        print('*** GPU not available - running on CPU. ***')

    full_dataset = torchvision.datasets.ImageFolder(args.train, transform=get_transforms())
    train_len = int(0.8 * len(full_dataset))
    validate_len = len(full_dataset) - train_len
    train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, (train_len, validate_len))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1)

    model_1_1 = SeedlingModelV1_1()
    best_model_1_1, acc_1_1 = train(model_1_1, args.epochs, args.learning_rate, args.output_data_dir, dev)
