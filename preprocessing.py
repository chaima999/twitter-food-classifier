import nltk
import pandas as pd
from nltk.corpus import stopwords
import re
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # GPU name

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sw = stopwords.words('english')


def clean_text(text):
    text = text.lower()

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ",
                  text)  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "", text)  # Removing URLs
    # text = re.sub(r"http", "",text)

    html = re.compile(r'<.*?>')

    text = html.sub(r'', text)  # Removing html tags

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')  # Removing punctuations

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    text = " ".join(text)  # removing stopwords

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Removing emojis

    return text


def get_preprocessing_objects():
    df = pd.read_csv("train.csv")
    df['text'] = df['text'].apply(lambda x: clean_text(x))

    tweets = df.text.values
    labels = df.target.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_len = max([len(tokenizer.encode(sent, add_special_tokens=True)) for sent in tweets])

    input_ids, attention_masks = [], []

    # For every tweet...
    for tweet in tweets:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            tweet,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 80-20 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.8 * len(dataset))
    # val_size = int(0.2 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # if device == "cuda:0":
    # # Tell pytorch to run this model on the GPU.
    #     model = model.cuda()
    model = model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    return train_dataloader, validation_dataloader, optimizer, model, tokenizer, max_len, batch_size

if __name__ == "__main__":
    # Optional test run
    train_dataloader, validation_dataloader, optimizer, model, tokenizer, max_len, batch_size = get_preprocessing_objects()
    print("Preprocessing completed successfully")



