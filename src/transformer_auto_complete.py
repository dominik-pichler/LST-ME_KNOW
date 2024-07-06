from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,PreTrainedModel,PreTrainedTokenizer
from typing import Tuple
from datasets import Dataset
import random
import torch
from torch.utils.data import DataLoader

'''
Resources: 
1) https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''



batch_size = 32
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Example: Using eos_token for padding


class textFileDataset(Dataset):
    def __init__(self, filepath, transform=None):
        with open(path, 'r') as file:
            file_content = file.read()

        sentences = [sentence.strip() for sentence in sentences]

        # Create a DataFrame from the list of sentences
        df = pd.DataFrame(sentences, columns=["sentence"])
        self.text = file_content

    ## Load text
    def load_text(path: str) -> str:
        with open(path, 'r') as file:
            # Read the entire content of the file into a string
            file_content = file.read()
        return file_content

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def generate_datasets(text: str) -> Tuple[Dataset, Dataset]:
        '''
        Function to generate tokenized datasets for hugging face models.

        :param text: A single text corpus that should be used for the task.
        :type text: str
        :return: Two datasets, where the first is the train_dataset and the second is the test_dataset
        :rtype: Tuple[Dataset,Dataset]
        '''

        sentences = text.split('.')
        random.shuffle(sentences)
        split = int(0.9 * len(sentences))
        train_sentences = sentences[:split]
        val_sentences = sentences[split:]

        train_dataset = Dataset.from_dict({"text": train_sentences})
        val_dataset = Dataset.from_dict({"text": val_sentences})

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_dataset, val_dataset



class autoComplete:

    def fine_tune_model(model: PreTrainedModel, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[
        PreTrainedModel, PreTrainedTokenizer]:
        '''
        Function to fine tune the pre-trained model on the given dataset

        :param path: Path to the corpus with which the model should be fine-tuned
        :type path: String
        :return: Tupel of pre-trained model and tokenizer
        :rtype: Tupel[Model,Tokenizer]
        '''

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        ## utilizing the trainer function to simplify  the training process
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        model.save_pretrained("./fine-tuned-gpt2")
        tokenizer.save_pretrained("./fine-tuned-gpt2")

        return model, tokenizer

    def predict_next_word(text, model, tokenizer, device, num_predictions=1):
        # Tokenize input text
        inputs = tokenizer.encode(text, return_tensors='pt').to(device)

        # Model inference
        with torch.no_grad():
            outputs = model(inputs)

        # Get logits and probabilities
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        probabilities = torch.softmax(last_token_logits, dim=-1)

        # Get top predicted indices
        predicted_indices = torch.topk(probabilities, num_predictions).indices.squeeze(0).tolist()

        # Convert indices to tokens
        predicted_tokens = [tokenizer.decode([index]) for index in predicted_indices]

        return predicted_tokens


if __name__ == '__main__':
    path = '../data/The_critique_of_pure_reason_full.txt'
    textFileDataset()