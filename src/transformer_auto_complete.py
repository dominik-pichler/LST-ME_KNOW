from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def predict_next_word(text, model, tokenizer, num_predictions=1):
    # Tokenize input text and get tensor input
    inputs = tokenizer.encode(text, return_tensors='pt')
    
    # Get the logits (predictions) from the model
    outputs = model(inputs)
    logits = outputs.logits
    
    # Get the logits of the last token and apply softmax to get probabilities
    last_token_logits = logits[:, -1, :]
    probabilities = torch.softmax(last_token_logits, dim=-1)
    
    # Get the indices of the top predictions
    predicted_indices = torch.topk(probabilities, num_predictions).indices.squeeze(0).tolist()
    
    # Decode the predicted tokens to get the next words
    predicted_tokens = [tokenizer.decode([index]) for index in predicted_indices]
    
    return predicted_tokens

# Example usage
text = "The quick brown fox"
predictions = predict_next_word(text, model, tokenizer, num_predictions=5)

print(f"Input text: {text}")
print(f"Predicted next words: {predictions}")

