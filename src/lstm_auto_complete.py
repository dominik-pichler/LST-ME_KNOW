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
import mlflow
import mlflow.pytorch

import warnings
warnings.filterwarnings("ignore")

# Config
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

data_path = Path('../data/The_critique_of_pure_reason_short.txt')
data_path_similar = Path('../data/Perpetual_Peace_Kant.txt')
data_path_not_similar = Path('../data/moby_dick.txt')

def load_text(data_path_in):
    with open(data_path_in, 'r') as file:
            file_content = file.read()

    # Filtering out invalid symbols
    file_content = file_content.replace('\n','')
    file_content = file_content.replace('\\n', '')
    pattern = re.compile(r'\[\d+\]')
    file_content = pattern.sub('', file_content)

    # Getting rid of empty lines:
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
    list_ngrams = []
    for i in range(1, len(tokenized_sentences)):
        ngram_sequence = tokenized_sentences[:i+1]
        list_ngrams.append(ngram_sequence)
    return list_ngrams

def add_random_oov_tokens(cumulative_ngram):
    for idx, word in enumerate(cumulative_ngram[:-1]):
        if random.uniform(0, 1) < 0.1:
            cumulative_ngram[idx] = '<oov>'
    return cumulative_ngram

def calculate_topk_accuracy(model, data_loader, k=3):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            _, predicted_indices = output.topk(k, dim=1)
            correct_predictions += torch.any(predicted_indices == torch.argmax(batch_y, dim=1, keepdim=True), dim=1).sum().item()
            total_predictions += batch_y.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy

def find_latest_checkpoint_path(checkpoint_path_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_path_dir) if f.startswith('checkpoint')]
    if not checkpoint_files:
        return None, 0

    epochs = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_checkpoint = f"checkpoint_epoch_{latest_epoch}.pt"

    return os.path.join(checkpoint_path_dir, latest_checkpoint)

def get_params():
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM model")
    parser.add_argument('--checkpoint', default='false', choices=['true', 'false'], help='Run from latest checkpoint')
    parser.add_argument('--checkpoint_path', default='../checkpoints', help='Define checkpoint path')
    parser.add_argument('--lr', default=0.0009, help='Learning rate: Hyperparameter')
    parser.add_argument('--epochs', default=50, help='Epochs')
    parser.add_argument('--hidden_dimensions', default=200, help='Hidden dimensions: Hyperparameter')
    return parser.parse_args()

def text_to_numerical_sequence_test(tokenized_text):
    tokens_list = []
    for token in tokenized_text:
        num_token = features_vocab[token] if token in features_vocab.get_itos() else features_vocab['<oov>']
        tokens_list.append(num_token)
    return tokens_list

def use_model(input_list):
    model.eval()
    output_list = []
    for data in input_list:
        sentence = data[0]
        num_words = data[1]
        for i in range(num_words):
            output_of_model = []
            tokenized_input_test = tokenizer(sentence)
            tokenized_sequence_input_test = text_to_numerical_sequence_test(tokenized_input_test)
            padded_tokenized_sequence_input_test = F.pad(torch.tensor(tokenized_sequence_input_test),
                                                         (longest_sequence_feature - len(tokenized_sequence_input_test)-1, 0),
                                                         value=0)
            output_test_walking = torch.argmax(model(padded_tokenized_sequence_input_test.unsqueeze(0)))
            sentence = sentence + ' ' + target_vocab.lookup_token(output_test_walking.item())
        output_list.append(sentence)
    return output_list

"""
Function which evaluates a given text corpus on a model
"""
def evaluate_model_on_corpus(data_path, model, tokenizer, features_vocab, target_vocab):

    # load text
    df = load_text(data_path)

    # tokenize
    tokenized_sentences = [tokenizer(sentence) for sentence in df]

    # create ngrams
    ngrams_list = []
    for tokenized_sentence in tokenized_sentences:
        ngrams_list.extend(make_cumulative_ngrams(tokenized_sentence))

    # convert to numerical sequences
    input_sequences = [text_to_numerical_sequence_test(sequence) for sequence in ngrams_list if
                       text_to_numerical_sequence_test(sequence)]

    X = [sequence[:-1] for sequence in input_sequences]
    y = [sequence[-1] for sequence in input_sequences]

    #pad sequences
    longest_sequence_feature = max(len(sequence) for sequence in X)
    padded_X = [F.pad(torch.tensor(sequence), (longest_sequence_feature - len(sequence), 0), value=0) for sequence in X]
    padded_X = torch.stack(padded_X)
    y = torch.tensor(y)
    y_one_hot = one_hot(y, num_classes=len(target_vocab))

    data = TensorDataset(padded_X, y_one_hot)
    data_loader = DataLoader(data, batch_size=32, shuffle=False)

    accuracy = calculate_topk_accuracy(model, data_loader)
    print(f'Test K-Accuracy for {data_path.name}: {accuracy * 100:.2f}%')
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

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.start_run()

    df = load_text(data_path)
    tokenizer = get_tokenizer('basic_english')
    tokenized_sentences = [tokenizer(title) for title in df]

    args = get_params()

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

    ngrams_list = []
    for tokenized_sentence in tokenized_sentences:
        ngrams_list.extend(make_cumulative_ngrams(tokenized_sentence))
    print(len(ngrams_list))

    ngrams_list_oov = []
    for ngram in ngrams_list:
        ngrams_list_oov.append(add_random_oov_tokens(ngram))
    print(any('<oov>' in cum_ngram for cum_ngram in ngrams_list_oov))

    input_sequences = [text_to_numerical_sequence(sequence) for sequence in ngrams_list_oov if
                       text_to_numerical_sequence(sequence)]

    features_vocab_total_words = len(features_vocab)
    target_vocab_total_words = len(target_vocab)

    X = [sequence[:-1] for sequence in input_sequences]
    y = [sequence[-1] for sequence in input_sequences]
    len(X[0]), y[0]

    longest_sequence_feature = max(len(sequence) for sequence in X)
    print(longest_sequence_feature)

    padded_X = [F.pad(torch.tensor(sequence), (longest_sequence_feature - len(sequence), 0), value=0) for sequence in X]
    padded_X[0], X[0], len(padded_X[0])

    padded_X = torch.stack(padded_X)
    y = torch.tensor(y)
    type(y), type(padded_X)

    y_one_hot = one_hot(y, num_classes=target_vocab_total_words)

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
    if(args.checkpoint == 'true'):
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



    for epoch in range(start_epoch, args.epochs):
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
            mlflow.log_metric('train_accuracy', accuracy)
            mlflow.log_metric('train_loss', loss.item())

            checkpoint_file_path = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_file_path)
            print(f"Checkpoint saved at {checkpoint_file_path}")
            mlflow.log_artifact(checkpoint_file_path, 'checkpoints')

    mlflow.pytorch.log_model(model, 'models')

    checkpoint_file_path = os.path.join(args.checkpoint_path, "checkpoint_model_final.pt")

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_file_path)
    print(f"Checkpoint saved at {checkpoint_file_path}")

    epoch_list = [i for i in range(1, args.epochs, 5)]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # Plotting code

    accuracy = calculate_topk_accuracy(model, test_loader)
    print(f'Test K-Accuracy: {accuracy * 100:.2f}%')
    mlflow.log_metric('test_k_accuracy_same_corpus', accuracy)

    #input_test = [['Daniel is',5],['stand', 5], ['deep learning is', 5], ['data cleaning', 4], ['6 ways', 4], ['you did a', 2]]
    input_test = [['Daniel is', 9]]

    outputs_model = use_model(input_test)
    print(outputs_model)


    input_test = [['Dominik is', 9]]
    input_test2 = [['Daniel is', 9]]

    outputs_model = use_model(input_test)
    print(outputs_model)
    outputs_model = use_model(input_test2)
    print(outputs_model)

    mlflow.log_metric('test_k_accuracy_similar_corpus', evaluate_model_on_corpus(data_path_similar, model, tokenizer, features_vocab, target_vocab))
    mlflow.log_metric('test_k_accuracy_different_corpus', evaluate_model_on_corpus(data_path_not_similar, model, tokenizer, features_vocab, target_vocab))

    mlflow.end_run()  # End MLflow run
