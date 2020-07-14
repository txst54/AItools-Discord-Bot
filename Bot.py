# bot.py
import os

import discord
from dotenv import load_dotenv

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import numpy as np


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client()


@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break

    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if '!replicate' in message.content.lower():
        uid = message.mentions[0]
        id = message.mentions[0].id
        #print(f"uid: {uid}")
        dataset = []
        mlen = []
        channel = message.channel
        async for message in channel.history(limit=10000):
            #print(message.author)
            if message.author == uid:
                mlen.append(len(message.content.split()))
                dataset.append(message.content.lower())
        #print(dataset)
        tokenizer = Tokenizer()

        data = dataset
        corpus = data

        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1
        print(total_words)
        #print(tokenizer.word_index)

        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)
        # Pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        # Create predictors and labels
        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        model = Sequential()
        model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
        model.add(Bidirectional(LSTM(25)))
        model.add(Dense(total_words, activation='softmax'))
        adam = Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        history = model.fit(xs, ys, epochs=15, verbose=1)
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)
        #await message.channel.send("Model: " + str(short_model_summary))
        # Testing (testing 1 2)
        seed_text = "Lol"
        next_words = 10
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        print(seed_text)
        print(id)
        newid = "<@!" + str(id) + ">"
        await message.channel.send("Replicating " + newid + ": " + seed_text)


client.run(TOKEN)