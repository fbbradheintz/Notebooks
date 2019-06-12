import argparse
import os
import time
import random

import torch
from torch import nn
from torchtext import data       # NLP data utilities



# helpers
def tlog(msg):
    print('{}   {}'.format(time.asctime(), msg))

def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_correct(guesses, labels):
    return (guesses == labels).float().sum()

def save_model(model, epoch, model_dir):
    savefile = "{}-e{}-{}.pt".format('pytorch-sentiment', epoch, int(time.time()))
    tlog('Saving model {}'.format(savefile))
    path = os.path.join(model_dir, savefile)
    # recommended way from https://pytorch.org/docs/stable/notes/serialization.html
    torch.save(model.state_dict(), path)
    return savefile


# model constants
EMBEDDING_SIZE = 100 # must match dimensions in vocab vectors above
HIDDEN_SIZE = 100
OUTPUT_SIZE = 5 # 0 to 4
NUM_LAYERS = 2



class SentimentAnalyzer(nn.Module):
    def __init__(self, embedding_size, pretrained_embedding, hidden_size, output_size, batch_size):
        super(SentimentAnalyzer, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            num_layers=NUM_LAYERS,
            hidden_size=hidden_size
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, phrases, hidden):
        x = self.embedding(phrases)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(hidden[0][-1]) # remove [0] for RNN or GRU, [-1] is for layers
        return x.squeeze(0), hidden
    
    def init_hidden(self, device):
        return (torch.zeros(NUM_LAYERS, self.batch_size, self.hidden_size, dtype=torch.float).to(device),
                torch.zeros(NUM_LAYERS, self.batch_size, self.hidden_size, dtype=torch.float).to(device))



# dataset constants
VOCAB_SIZE = 15000 # max size of vocabulary
VOCAB_VECTORS = "glove.6B.100d" # Stanford NLP GloVe (global vectors) for word rep



def get_data(batch_size, device):
    tlog('Preparing data...')
    phrases_fieldspec = data.Field(include_lengths=True, tokenize='spacy')
    labels_fieldspec = data.LabelField(dtype=torch.int64, sequential=False)

    fields = [
        ('SKIP_phrase_id', None),
        ('SKIP_sentence_id', None),
        ('phrases', phrases_fieldspec),
        ('labels', labels_fieldspec)
    ]


    train_data = data.TabularDataset(
        '/opt/ml/input/data/training/train.tsv', # path to file
        'TSV', # file format
        fields,
        skip_header = True # we have a header row
    )
    
    train_data, eval_data = train_data.split()
    phrases_fieldspec.build_vocab(train_data, max_size=VOCAB_SIZE, vectors=VOCAB_VECTORS)
    labels_fieldspec.build_vocab(train_data)
    vocab_size = len(phrases_fieldspec.vocab)
    output_size = len(labels_fieldspec.vocab)
    
    train_iter, eval_iter = data.BucketIterator.splits(
        (train_data, eval_data), 
        batch_size=batch_size,
        device=device, sort=False, shuffle=True)
    
    tlog('Data prepared')
    return train_iter, eval_iter, vocab_size, output_size


def get_model(output_size, vectors, batch_size, device):
    tlog('Creating model...')
    sa = SentimentAnalyzer(EMBEDDING_SIZE, vectors, HIDDEN_SIZE, output_size, batch_size)
    sa = sa.to(device)
    tlog('The model has {} trainable parameters'.format(count_model_params(sa)))
    tlog(sa)
    return sa

    
def train(model, iterator, loss_fn, optimizer, batch_size, device): # one epoch
    curr_loss = 0.
    curr_correct = 0.
    model.train() # makes sure that training-only fns, like dropout, are active
    
    for batch in iterator:
        # get the data
        phrases, lengths = batch.phrases
        hidden = model.init_hidden(device)
        
        if phrases.shape[1] == batch_size:        
            # predict and learn
            optimizer.zero_grad()
            guesses, hidden = model(phrases, hidden)
            loss = loss_fn(guesses, batch.labels)
            loss.backward()
            optimizer.step()
            
            hidden[0].detach_() # or we get double-backward errors
            hidden[1].detach_() # or we get double-backward errors

            # measure
            curr_loss += loss.item()
            curr_correct += count_correct(torch.argmax(guesses, 1), batch.labels)
        
    return curr_loss / len(iterator), curr_correct / (len(iterator) * batch_size)


def evaluate(model, iterator, loss_fn, batch_size, device):
    curr_loss = 0.
    curr_correct = 0.
    model.eval() # makes sure that training-only fns, like dropout, are inactive
    
    with torch.no_grad(): # not training
        for batch in iterator:
            # get the data
            phrases, lengths = batch.phrases
            hidden = model.init_hidden(device)
            
            if phrases.shape[1] == batch_size:        
                # predict
                guesses, hidden = model(phrases, hidden) # .squeeze(1)
                loss = loss_fn(guesses, batch.labels)

                # measure
                curr_loss += loss.item()
                curr_correct += count_correct(torch.argmax(guesses, 1), batch.labels)

    return curr_loss / len(iterator), curr_correct / (len(iterator) * batch_size) 


def learn(model, train_iter, eval_iter, epochs, lr, batch_size, output_dir, device):
    eval_losses = []
    eval_accs = []
    best_eval_acc = 0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    for epoch in range(epochs):
        tlog('EPOCH {} of {}'.format(epoch + 1, epochs))
        
        train_loss, train_acc = train(model, train_iter, loss_fn, optimizer, batch_size, device)
        tlog('  Training loss {}   acc {}'.format(train_loss, train_acc))
        
        eval_loss, eval_acc = evaluate(model, eval_iter, loss_fn, batch_size, device)
        tlog('  Validation loss {}   acc {}'.format(eval_loss, eval_acc))
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        if eval_acc > best_eval_acc:
            tlog('  *** New accuracy peak, saving model')
            best_eval_acc = eval_acc
            saved_model_filename = save_model(model, epoch + 1, output_dir)
    
    tlog('DONE')
    tlog('Best accuracy: {}'.format(best_eval_acc))
    return eval_losses, eval_accs



if __name__ == '__main__':
    os.system('python -m spacy download en')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--cuda', dest='use_cuda', action='store_true', default=True)
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false')

    # Data, model, and output directories
    if 'SM_CHANNEL_TRAINING' in os.environ: # are we in sagemaker?
        parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    else:
        parser.add_argument('--output-data-dir', type=str, default='models')
        parser.add_argument('--model-dir', type=str, default='models')
        parser.add_argument('--train', type=str, default='.')

    args, _ = parser.parse_known_args()

    if args.use_cuda and torch.cuda.is_available():
        target_device = torch.device('cuda')
        print('GPU ready to go!')
    else:
        target_device = torch.device('cpu')
        print('*** GPU not available - running on CPU. ***')

    t_iter, e_iter, vocab_size, output_size = get_data(args.batch_size, target_device)
    embedding_vectors = t_iter.dataset.fields['phrases'].vocab.vectors.to(target_device)

    sa = get_model(output_size, embedding_vectors, args.batch_size, target_device)
    losses, accs = learn(sa, t_iter, e_iter, args.epochs, args.learning_rate, args.batch_size, args.output_data_dir, target_device)
