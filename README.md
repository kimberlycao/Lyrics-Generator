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

The goal of this project is to create a model that generates 80's and 90's love song lyrics. This is an unsupervised learning problem that deals with sequential data, since songs for usually contextual. For this reason, we'll be doing it using **(1)** a recurrent neural network (RNN) with gated recurrent units (GRU) and **(2)** using the [GPT-2 transformer](https://huggingface.co/gpt2) provided by Huggingface.

Users will input the model, maximum length of the output lyric sequence and a temperature to generate lyrics using method **(1)**. To produce lyrics using method **(2)**, users will input a top_p, top_k, the number of returned lyric sequences and a maximum length for each lyric sequence.

## Model

## Data

## Training

## Results

## Ethical Considerations

Using this model to compose lyrics that which are publicly shared can lead to issues involving ownership and copyright. Musicians and record labels who write or record the original song usually own the copyright for it. Co-writers of a song will be joint copyright holders for it with equal rights, however, musicians can negotiate to for unequal rights. With a song co-written by this model, it becomes complicated whether credit should be given to the model, the engineers that designed the model, or the user who gave the prompt to the model.

Copyright is a serious issue that can leave a guilty party liable for payment. It's possible for this model to produce the same lyrics for multiple users and this becomes risky if the user is submitting the song to a competition or for assignment. The overlap in similarity in these cases is easily detectable if used in public. The model can also produce lyrics that already exist. Although some songs contain identical/similar phrases without issue, this can appear unoriginal and can be problematic in the case of a competition.

Copyright issues can extend to using existing songs to train the model. Being that this assignment is for educational purposes, we can use the lyrics without the risk of copyright infringement. However, this becomes a risk if the songs being created are released publicly and are being monetized. Copyright is concerned with the point of when a song is copied and in machine learning, the point would be gathering the training data.

Copyright is meant to protect a person's musical works. Copyright infringement is particularly harmful to the person found guilty. The guilty party may be ordered to pay fines/damages and can also suffer reputational damage as well. This is especially concerning because we are using Genius lyrics to train this model and potentially leaving ourselves vulnerable to copyright infringement if this model is used to produce a song for more than a momentary laugh. People who produce lyrics similar to a song that already exists can also be at risk for copyright infringement. Users of this model should be careful not to use the same instrumentals as existing songs.

Depending on the age of the user or the intended audience of a song produced by this model, some undesirable lyrics consisting of profanity/racism/etc. may be produced (e.g. I'd shoot up the school for you girl). This can be seen as inappropriate to some and in the future, a way to filter out inappropriate lyrics would be a good feature.

## Authors
