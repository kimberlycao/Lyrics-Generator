# Lyrics Generator 🎶

## Table of Contents

- [Introduction](#introduction)
- [Model](#model)
- [Data](#data)
- [Training](#training)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [Authors](#authors)

## Introduction

The goal of this project is to create a model that generates 80's and 90's love song lyrics. This is an unsupervised learning problem that deals with sequential data, since songs are usually contextual. For this reason, we'll be doing it using **(1)** a recurrent neural network (RNN) with gated recurrent units (GRU) and **(2)** using the [OpenAI GPT-2 transformer model](https://huggingface.co/gpt2) provided by Huggingface and fine-tuning it.

Users will input the model, maximum length of the output lyric sequence and a temperature to generate lyrics using method **(1)**. To produce lyrics using method **(2)**, users will input a top_p, top_k, the number of returned lyric sequences and a maximum length for each lyric sequence.

## Model

**Method (2):** GPT-2 is a generative model that is capable of writing novels, news articles and even poetry. In order for GPT-2 to generate song lyrics, it needs to be fine-tuned using lyrical data.

## Data

### Data Source

Our data is a collection of 80’s and 90’s love song lyrics scraped from Genius.com. To find songs to use, we curated a collection of playlists from Spotify and used each song we found (disregarding duplicates among playlists). To acquire the lyrics, we used the artist and song name of each track to scrape the lyrics off of Genius.com. In total, we had lyrics for 184 songs in our dataset.

### Data Summary

In order to interpret our results, we collected a number of summary statistics on our data. We found that, on average, a song in our dataset was 1559.25 characters long, and 323.69 words long. As such, the inputs to our model are fairly long. 

We also investigated the amount of repetition that exists in our dataset. Our data had 80 unique characters and 3805 unique words. As for the frequencies of these words, we found that the top 20% of our most frequently used words made up for 87.27% of the words used in our lyrics. We also found that the top 10 words used in our songs were as follows:

| Word      | Frequency (%) |
| ----------| ----------- |
| I         | 5.36       |
| you       | 4.97        |
| the       | 2.58       |
| to        | 2.33        |
| me        | 1.77       |
| my        | 1.49        |
| love      | 1.23       |
| And       | 1.14        |
| a         | 1.05       |
| in        | 1.02        |

As for the length of each lyric, each line of our songs are, on average, 29.05 characters in length and 6.23 words long.

### Data Transformation

**Method (2):** Input strings need to be tokenized (converted into a numerical representation) before they are passed to the model for training or generation. The transformers library comes with a word-based tokenizer, which we used for this project.

```python
# Load GPT2 tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

# Necessary if we have added our own custom tokens

model.resize_token_embeddings(len(tokenizer))
```

An example of the tokenizer in use:

```python
test_sentence1 = "I love this computer science class"
input_ids1 = tokenizer.encode(test_sentence1, return_tensors='pt')

test_sentence2 = "I love puppies"
input_ids2 = tokenizer.encode(test_sentence2, return_tensors='pt')

print(input_ids1) # tensor([[  40, 1842,  428, 3644, 3783, 1398]])
print(input_ids2) # tensor([[   40,  1842, 37793]])
```

### Data Split

For both methods **(1)** and **(2)**, the dataset was not split - the entire dataset was used to train with. In this problem, we're trying to understand the underlying structure of the data instead of performing a classification task where there is a right and wrong answer. Usually, we're able to calculate accuracy for a test set to determine whether the model is good or bad, but measurements like accuracy are not applicable for a creative task. For this reason, we didn't split our data before training.

## Training

**Method (2):**

<img src="images/gpt2-training-loss.jpg" alt="GPT-2 Training Loss" width="450"/>

| Epoch | Training Loss |
| ----- | :-----------: |
| 1     |   5.159400    |
| 2     |   1.258556    |
| 3     |   1.117548    |
| 4     |   1.031602    |
| 5     |   0.985897    |
| 6     |   0.951483    |

When training the GPT-2 model, we found that using epochs equal to 5 and 6 yielded a similar training loss by the last epoch. Each time, the largest decrease in training loss happened between epochs 1 and 2. Epochs beyond 2 decreased minimally relative to this and each epoch took approximately 25 minutes to train, so epochs greater than 6 were not tested. We used a batch size of 2 because anything larger than 2 resulted in an out of memory error. We could have used accumulated gradients in order to use larger batch sizes, but did not for this project. When trying a learning rate of 0.0001, the lyrics were very repetitive and random-sounding compared to a learning rate of 0.0005. This smaller learning rate could have prevented the model from converging, since the number of epochs were kept the same in both training sessions and 0.0005 produced significantly better lyrics. We used a learning rate scheduler during training and the number of warmup steps was set to 100. The scheduler decreases the learning rate as the number of epochs increases. This prevents the optimization of the loss function from diverging. We chose a small epsilon value because we wanted the model to choose the next word with the highest probability most of the time.

## Results

## Ethical Considerations

Using this model to compose lyrics that which are publicly shared can lead to issues involving ownership and copyright. Musicians and record labels who write or record the original song usually own the copyright for it. Co-writers of a song will be joint copyright holders for it with equal rights, however, musicians can negotiate to for unequal rights. With a song co-written by this model, it becomes complicated whether credit should be given to the model, the engineers that designed the model, or the user who gave the prompt to the model.

Copyright is a serious issue that can leave a guilty party liable for payment. It's possible for this model to produce the same lyrics for multiple users and this becomes risky if the user is submitting the song to a competition or for assignment. The overlap in similarity in these cases is easily detectable if used in public. The model can also produce lyrics that already exist. Although some songs contain identical/similar phrases without issue, this can appear unoriginal and can be problematic in the case of a competition.

Copyright issues can extend to using existing songs to train the model. Being that this assignment is for educational purposes, we can use the lyrics without the risk of copyright infringement. However, this becomes a risk if the songs being created are released publicly and are being monetized. Copyright is concerned with the point of when a song is copied and in machine learning, the point would be gathering the training data.

Copyright is meant to protect a person's musical works. Copyright infringement is particularly harmful to the person found guilty. The guilty party may be ordered to pay fines/damages and can also suffer reputational damage as well. This is especially concerning because we are using Genius lyrics to train this model and potentially leaving ourselves vulnerable to copyright infringement if this model is used to produce a song for more than a momentary laugh. People who produce lyrics similar to a song that already exists can also be at risk for copyright infringement. Users of this model should be careful not to use the same instrumentals as existing songs.

Depending on the age of the user or the intended audience of a song produced by this model, some undesirable lyrics consisting of profanity/racism/etc. may be produced (e.g. I'd kill my mom for love). This can be seen as inappropriate to some and in the future, a way to filter out inappropriate lyrics would be a good feature.

## Authors and Credit

[Nameera Azim](https://github.com/NameeraAzim)

- GRU model training and overfitting

[Yousef Bulbulia](https://github.com/CometWhoosh)

- Data scraping and augmentation

[Kimberly Cao](https://github.com/kimberlycao)

- GPT-2 model fine-tuning
