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
import re
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

data_path = Path('../data/The_critique_of_pure_reason_full.txt')

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
    return df_sentences


def tokenizer(df_sentences):
    # Access the 'sentence' column from the DataFrame
    sentences = df_sentences['sentence'].tolist()

    # Initialize tokenizer
    basic_english_tokenizer = get_tokenizer('basic_english')

    # Tokenize each sentence
    tokenized_sentences = [basic_english_tokenizer(sentence) for sentence in sentences]
    print(len(tokenized_sentences))

    return tokenized_sentences



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

def make_ngrams(tokenized_sentences):
    '''
    Creates n-grams based on tokenized_sentences.

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

def add_random_oov_tokens(ngram):
    '''
    This function simulates adding out-of-vocabulary tokens to an n-gram with a certain probability.
    It could be used, for example, in language modeling or text generation tasks to simulate the presence of unknown
    words in the training data, which helps improve the model's robustness and generalization ability.
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
    tokenized_sentences = tokenizer(df)

    features_vocab, target_vocab = vocab_builder(tokenized_sentences)
    features_vocab_total_words = len(features_vocab)
    target_vocab_total_words = len(target_vocab)

    print(f'Total number of words in features vocabulary: {features_vocab_total_words}')
    print(f'Total number of words in target vocabulary: {target_vocab_total_words}')
    print('-'*30)
    print('Word -> ID')
    print('<pad> -> '+ str(features_vocab['<pad>']))
    print('<oov> -> '+ str(features_vocab['<oov>']))

    ngrams_list = []
    for tokenized_sentence in tokenized_sentences:
        ngrams_list.extend(make_ngrams(tokenized_sentences))
    print(len(ngrams_list))


    ngrams_list_oov = []
    for ngram in ngrams_list:
        ngrams_list_oov.append(add_random_oov_tokens(ngram))
    print(any('<oov>' in ngram for ngram in ngrams_list_oov))

    input_sequences = [text_to_numerical_sequence(sequence) for sequence in ngrams_list_oov if
                       text_to_numerical_sequence(sequence)]

    print(f'Total input sequences: {len(input_sequences)}')
    print(input_sequences[7:9])


    # Getting faetues and target
    X = [sequence[:-1] for sequence in input_sequences]
    y = [sequence[-1] for sequence in input_sequences]
    len(X[0]), y[0]

    longest_sequence_feature = max(len(sequence) for sequence in X)
    print(longest_sequence_feature)

    padded_X = [F.pad(torch.tensor(sequence), (longest_sequence_feature - len(sequence),
                                               0), value=0) for sequence in X]
    padded_X[0], X[0], len(padded_X[0])

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
    hidden_dim = 200
    epochs = 50

    model = nextWord_LSTM(features_vocab_total_words,
                    target_vocab_total_words,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0009)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    all_accuracies = []
    all_losses = []
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.argmax(dim=1))
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            accuracy = calculate_topk_accuracy(model, train_loader)
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Train K-Accuracy: {accuracy * 100:.2f}%')
            all_accuracies.append(accuracy)
            all_losses.append(loss.item())

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
    plt.show()