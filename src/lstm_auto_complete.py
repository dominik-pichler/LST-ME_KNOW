''''
Resources:
# https://www.kaggle.com/code/dota2player/next-word-prediction-with-lstm-pytorch/notebook
'''
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
import torchtext
import torchtext.vocab
import torch.nn.functional as F
import tqdm
from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.functional import one_hot
import torch.nn as nn
import torch.optim as optim
import re, os, argparse
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

data_path = Path('../data/The_critique_of_pure_reason_short.txt')

def load_text():
    with open(data_path, 'r') as file:
            file_content = file.read()

    # Filtering out invalid symbols
    file_content = file_content.replace('\n','')
    file_content = file_content.replace('\\n', '')
    pattern = re.compile(r'\[\d+\]')
    file_content = pattern.sub('', file_content)

    # getting rid of empty lines:
    lines = file_content.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    cleaned_text = '\n'.join(non_empty_lines)


    # Create a DataFrame from the list of sentences
    sentences = cleaned_text.split('.')
    df_sentences = pd.DataFrame(sentences, columns=["sentence"])
    return df_sentences['sentence']




def vocab_builder(tokenized_sentences):
    features_vocab = torchtext.vocab.build_vocab_from_iterator(
        tokenized_sentences,
        min_freq=2,
        specials=['<pad>', '<oov>'],
        special_first=True
    )
    target_vocab = torchtext.vocab.build_vocab_from_iterator(
        tokenized_sentences,
        min_freq=2
    )

    return features_vocab, target_vocab


def text_to_numerical_sequence(tokenized_text):
    '''
    Converts the tokenized text to numerical sequence.
    Example:
        vocab = {'my': 0, 'Ahmed': 1, 'is': 2, 'name': 3}
        data = ['my', 'name', 'is']
        numerical_sequence = [0, 3, 2]

    :param tokenized_text:
    :type tokenized_text:
    :return:
    :rtype:
    '''


    tokens_list = []
    if tokenized_text[-1] in target_vocab.get_itos():
        for token in tqdm.tqdm(tokenized_text[:-1]):
            num_token = features_vocab[token] if token in features_vocab.get_itos() else features_vocab['<oov>']
            tokens_list.append(num_token)
        num_token = target_vocab[tokenized_text[-1]]
        tokens_list.append(num_token)
        return tokens_list
    return None


def make_cumulative_ngrams(tokenized_sentences):
    '''
    Creates cumulative n-grams based on tokenized_sentences.

    Example:
        data = ['my', 'name', 'is', 'Ahmed']
        Ngram = [['my', 'name'], ['my', 'name', 'is'], ['my', 'name', 'is', 'Ahmed']]

    :param tokenized_sentences:
    :type tokenized_sentences:
    :return:
    :rtype:
    '''
    list_ngrams = []
    for i in range(1, len(tokenized_sentences)):
        ngram_sequence = tokenized_sentences[:i+1]
        list_ngrams.append(ngram_sequence)
    return list_ngrams

def add_random_oov_tokens(cumulative_ngram):
    '''
    This function adds out-of-vocabulary tokens to an cumulativ n-gram with a certain probability (default 10%) ,
    to improve the model's robustness and generalization ability.
    :param ngram:
    :type ngram:
    :return:
    :rtype:
    '''

    for idx, word in enumerate(ngram[:-1]):
        if random.uniform(0, 1) < 0.1:
            ngram[idx] = '<oov>'
    return ngram




def calculate_topk_accuracy(model, data_loader, k=3):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            output = model(batch_x)

            # Get top-k predictions
            _, predicted_indices = output.topk(k, dim=1)

            # Check if the correct label is in the top-k predictions
            correct_predictions += torch.any(predicted_indices == torch.argmax(batch_y, dim=1, keepdim=True),
                                                 dim=1).sum().item()
            total_predictions += batch_y.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


"""
input: path for folder with stored checkpoints
"""
def find_latest_checkpoint_path(checkpoint_path_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_path_dir) if f.startswith('checkpoint')]
    if not checkpoint_files:
        return None, 0

    epochs = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_checkpoint = f"checkpoint_epoch_{latest_epoch}.pt"

    return os.path.join(checkpoint_path_dir, latest_checkpoint)


"""
returns parameters from command line with set default parameters
"""
def get_params():

    parser = argparse.ArgumentParser(description="Train and evaluate LSTM model")
    parser.add_argument('--checkpoint', default='false', choices=['true', 'false'], help='Run from latest checkpoint')
    parser.add_argument('--checkpoint_path', default='../checkpoints', help='Define checkpoint path')
    parser.add_argument('--lr', default=0.0009, help='Learning rate: Hyperparameter')
    parser.add_argument('--epochs', default=10, help='Epochs: Hyperparameter')
    parser.add_argument('--hidden_dimensions', default=200, help='Hidden dimensions: Hyperparameter')
    return parser.parse_args()


class nextWord_LSTM(nn.Module):
    def __init__(self, features_vocab_total_words, target_vocab_total_words, embedding_dim, hidden_dim):
        super(nextWord_LSTM, self).__init__()
        self.embedding = nn.Embedding(features_vocab_total_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, target_vocab_total_words)

    def forward(self, x):
        x = x.to(self.embedding.weight.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output


if __name__ == '__main__':
    df = load_text()
    tokenizer = get_tokenizer('basic_english')
    tokenized_sentences = [tokenizer(title) for title in df]

    args = get_params()

    features_vocab = torchtext.vocab.build_vocab_from_iterator(
        tokenized_sentences,
        min_freq=2, # Wordlength min
        specials=['<pad>', '<oov>'], #Add two tokens to the feature vocab.
        special_first=True
    )
    target_vocab = torchtext.vocab.build_vocab_from_iterator(
        tokenized_sentences,
        min_freq=2
    )


    # An n-gram is built of size len(sentence)
    ngrams_list = []
    for tokenized_sentence in tokenized_sentences:
        ngrams_list.extend(make_cumulative_ngrams(tokenized_sentence))
    print(len(ngrams_list))


    ngrams_list_oov = []
    for ngram in ngrams_list:
        ngrams_list_oov.append(add_random_oov_tokens(ngram))
    print(any('<oov>' in cum_ngram for cum_ngram in ngrams_list_oov))


    # Translate numberics
    input_sequences = [text_to_numerical_sequence(sequence) for sequence in ngrams_list_oov if
                       text_to_numerical_sequence(sequence)]


    features_vocab_total_words = len(features_vocab)
    target_vocab_total_words = len(target_vocab)

    # Getting features and target
    X = [sequence[:-1] for sequence in input_sequences] # Features
    y = [sequence[-1] for sequence in input_sequences] # Target (last word of each sentence, if! in vocab!
    len(X[0]), y[0]

    longest_sequence_feature = max(len(sequence) for sequence in X)
    print(longest_sequence_feature)

    # padding to equal length
    padded_X = [F.pad(torch.tensor(sequence), (longest_sequence_feature - len(sequence),
                                               0), value=0) for sequence in X]
    padded_X[0], X[0], len(padded_X[0])

    # Transformation to train Metal Performance Shader (mps)
    padded_X = torch.stack(padded_X)
    y = torch.tensor(y)
    type(y), type(padded_X)

    y_one_hot = one_hot(y, num_classes=target_vocab_total_words)


    ## We gonna use Pytorch DataLoaders to load the data after spliting our data to train and test.
    data = TensorDataset(padded_X, y_one_hot)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    batch_size = 32

    train_data, test_data = random_split(data, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    embedding_dim = longest_sequence_feature
    hidden_dim = args.hidden_dimensions
    epochs = args.epochs

    model = nextWord_LSTM(features_vocab_total_words,
                    target_vocab_total_words,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # load latest checkpoint if desired
    if(args.checkpoint == 'false'):
        checkpoint_path = find_latest_checkpoint_path(args.checkpoint_path)
        print(f"Loading latest checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")
    model.to(device)


    all_accuracies = []
    all_losses = []


    for epoch in range(start_epoch, args.epoch):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad() # remove gradients from last epoch to avoid summing up prio gradients
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.argmax(dim=1)) #loss function
            loss.backward() # Calculate Gradients
            optimizer.step() # Adjust weights

        if epoch % 5 == 0:
            accuracy = calculate_topk_accuracy(model, train_loader)
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Train K-Accuracy: {accuracy * 100:.2f}%')
            all_accuracies.append(accuracy)
            all_losses.append(loss.item())

            #save checkpoint
            checkpoint_file_path = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt")

            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_file_path)
            print(f"Checkpoint saved at {checkpoint_file_path}")


    epoch_list = [i for i in range(1, epochs, 5)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].plot(epoch_list, all_accuracies, color='#5a7da9', label='Accuracy', linewidth=3)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Graph')
    axes[0].grid(True)

    axes[1].plot(epoch_list, all_losses, color='#adad3b', label='Accuracy', linewidth=3)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Graph')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("../out/Accuracy_Loss_Graph.png")
