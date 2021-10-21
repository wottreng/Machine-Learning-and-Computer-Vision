"""
code credit goes to Mariya:
https://github.com/MariyaSha/
https://www.youtube.com/c/PythonSimplified
"""

import os
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random

# nltk.download('punkt', download_dir=f"{os.getcwd()}")  # only run once
from nltk.tokenize import word_tokenize
import string
punct = string.punctuation

trainModel = False


class StoryTeller(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size):
        super(StoryTeller, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(batch_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, 512)
        self.linear3 = nn.Linear(512, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def plotAverageLoss(average_loss):
    loss_plot = pd.DataFrame(average_loss)
    loss_plot.plot()
    plt.show()
    pause()


def pause():
    keys = input("press 'ENTER' to continue or q to quit\n")
    if keys == "q":
        quit()


if trainModel:

    # Pre - processing Coraline
    # a function to pre process Coraline by Neil Gaiman
    def preprocess_coraline():
        '''
        param book: url od a PDF book file
        '''
        book = f"{os.getcwd()}/Coraline.pdf"
        output = ""
        data = open(book, 'rb')
        data = PyPDF2.PdfFileReader(book)
        npages = data.getNumPages()
        for i in range(npages):
            page_i = data.getPage(i).extractText()
            output += page_i
        output = output[1227:]
        output = output.lower()
        for word in output:
            for char in word:
                if char in punct:
                    word = word.replace(char, "")
        remove_punct = "".join([word for word in output if word not in punct])
        processed = word_tokenize(remove_punct)
        print(f'Coraline database includes {len(processed)} tokens, and {len(set(processed))} unique tokens after editing')
        # add pickle file here ----
        return processed


    coraline = preprocess_coraline()


    # Preprocessing Alice in Wonderland

    # a function to pre process Alice's Advantures in Wonderland by Lewis Carroll

    def load_alice(text_file, punct, not_a_word):
        '''
        param text_file: url to Project Gutenberg's text file for Alice's Advantures in Wonderland by Lewis Carroll
        param punct: a string of punctuation characters we'd like to filter
        param not_a_word: a list of words we'd like to filter
        '''
        book = open(text_file, 'r')
        book = book.read()
        book = book[715:145060]
        book_edit = re.sub('[+]', '', book)
        book_edit = re.sub(r'(CHAPTER \w+.\s)', '', book)
        words = word_tokenize(book_edit.lower())

        word_list = []

        # filtering punctuation and non-words
        for word in words:
            for char in word:
                if char in punct:
                    word = word.replace(char, "")
            if word not in punct and word not in not_a_word:
                word_list.append(word)

        print('Alice database includes {} tokens, and {} unique tokens after editing'.format(len(word_list),
                                                                                             len(set(word_list))))
        return word_list


    # alice = load_alice(alice_url, (punct.replace('-', "") + '’' + '‘'), ['s', '--', 'nt', 've', 'll', 'd'])

    # Preprocessing Grimm
    def load_fairytales(text_file):
        '''
        param text_file: url to Project Gutenberg's text file for Fairytales by The Brothers Grimm
        '''
        book = open(text_file, encoding='cp1252')
        book = book.read()
        book = book[2376:519859]
        book_edit = re.sub('[(+*)]', '', book)
        words = word_tokenize(book_edit.lower())

        # filtering punctuation inside tokens (example: didn't or wow!)
        for word in words:
            for char in word:
                if char in punct:
                    word = word.replace(char, "")

        # filtering punctuation as alone standing tokens(example: \ or ,)
        words = [word for word in words if word not in punct]

        print('Fairytales database includes {} tokens, and {} unique tokens after editing'.format(len(words),
                                                                                                  len(set(words))))
        return words


    # brothers_grimm = load_fairytales()
    # Combined database including all books
    data = coraline  # + alice + brothers_grimm
    print(data[:10])

    # Convert Data into Numeric Values
    vocab = set(data)
    vocab_size = len(data)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    data = [word_to_index[word] for word in data]

    print(data[:10])
    print(word_to_index["coraline"])
    # Batching Data
    batch_size = 5
    train_data = [([data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4]], data[i + 5]) for i in range(vocab_size - batch_size)]

    # Defining the Neural Network
    embedding_dim = 5


    # Defining Training Function
    average_loss = []


    def train(model, train_data, epochs, word_to_index):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Training on GPU")
        else:
            device = torch.device("cpu")
            print("Training on CPU")

        model.to(device)

        for i in range(epochs):
            model.train()
            steps = 0
            print_every = 100
            running_loss = 0
            for feature, target in train_data:
                feature_tensor = torch.tensor([feature], dtype=torch.long)
                feature_tensor = feature_tensor.to(device)
                target_tensor = torch.tensor([target], dtype=torch.long)
                target_tensor = target_tensor.to(device)
                model.zero_grad()
                log_probs = model(feature_tensor)
                loss = criterion(log_probs, target_tensor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps += 1

                if steps % print_every == 0:
                    model.eval()
                    average_loss.append(running_loss / print_every)
                    print(f"Epochs: {i + 1} / {epochs}")
                    print(f"Training Loss: {running_loss / print_every}")
                    running_loss = 0
                model.train()

        return model


    # Train Model
    model = StoryTeller(vocab_size, embedding_dim, batch_size)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 10
    device = 0  # gpu

    start_time = time.time()
    model = train(model, train_data, epochs, word_to_index)
    print(f"training took {round(((time.time() - start_time) / 60), 2)} minutes")

    # Save Checkpoint
    checkpointFile = f'{os.getcwd()}/checkpoint5.pt'

    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'word_to_index': word_to_index,
                  'index_to_word': {i: word for i, word in enumerate(vocab)},
                  'epochs': epochs,
                  'average_loss': average_loss,
                  'device': device,
                  'optimizer_state': optimizer.state_dict(),
                  'batch_size': batch_size}

    torch.save(checkpoint, checkpointFile)

    plotAverageLoss(average_loss)

else:  # run model

    # Load Checkpoint
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.optimizer_state = checkpoint['optimizer_state']
        model.load_state_dict(checkpoint['state_dict'])
        model.device = checkpoint['device']
        model.word_to_index = checkpoint['word_to_index']
        model.index_to_word = checkpoint['index_to_word']
        model.average_loss = checkpoint['average_loss']
        return model


    checkpointFile = f'{os.getcwd()}/checkpoint5.pt'
    model = load_checkpoint(checkpointFile)
    index_to_word = model.index_to_word
    word_to_index = model.word_to_index

    # ---------------------------------------------------------------------------

    plotAverageLoss(model.average_loss)

    # Predict Function
    def predict(model, first_words, story_len, top_k):
        '''
        param model: trained model
        param first_words: a string of 5 (n_feature) words to begin the story
        param story_len: an integer symbolizing the number of words you'd like the story to have
        param top_k: the number of top probabilities per word that the network will randomly select from
        '''
        feature = (first_words.lower()).split(" ")
        for i in feature:
            story.append(i)
            for i in range(story_len):
                feature_idx = torch.tensor([word_to_index[word] for word in feature], dtype=torch.long)
                feature_idx = feature_idx.to(device)
                with torch.no_grad():
                    output = model.double().forward(feature_idx)
                ps = torch.exp(output)
                topk_combined = ps.topk(top_k, sorted=True)
                # top kk probabilities
                topk_ps = topk_combined[0][0]
                # top kk classes
                topk_class = topk_combined[1][0]
                topk_class = [index_to_word[int(i)] for i in topk_class]
                next_word = random.choice(topk_class)
                feature = feature[1:]
                feature.append(next_word)
                story.append(next_word)
        return story


    # Predict

    batch_size = 5 # features
    #first_words = input(f'Type the first {batch_size} words to start the story:\nexample: A lovely day at the\n')
    first_words = "the first day at the"
    top_k = 3
    story_len = 20
    story = []
    device = 'cuda:0'

    # Predicting and Handling User-Input Errors
    try:
        prediction = predict(model, first_words, story_len, top_k)
    except KeyError as error:
        print('Oops, looks like you\'ve selected a word that the network does not understand yet: ', error)
        if story[0] != "":
            story = story[len(first_words):]
        first_words = input('please select a different word:\nexample: A lovely day at the\n')
        prediction = predict(model, first_words, story_len, top_k)
    except KeyError and RuntimeError:
        if story[0] != "":
            story = story[len(first_words):]
        first_words = input(f'Oops, looks like you\'ve typed {len(first_words.split(" "))} words instead of {batch_size}!\n\nType the first 5 words to start the story:\nexample: A lovely day at the\n')
        prediction = predict(model, first_words, story_len, top_k)

    print(    '-----------------------------------------------------\n The STORY \n-----------------------------------------------------')
    print(" ".join(story))
