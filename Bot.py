# bot.py

from __future__ import print_function
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

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
import wsgiref.simple_server
import wsgiref.util

from io import StringIO
import sys

import time

SCOPES = ['https://www.googleapis.com/auth/classroom.student-submissions.me.readonly',
          'https://www.googleapis.com/auth/classroom.courses.readonly']
_DEFAULT_AUTH_PROMPT_MESSAGE = (
    'Please visit this URL to authorize this application: {url}')
"""str: The message to display when prompting the user for
authorization."""
_DEFAULT_AUTH_CODE_MESSAGE = (
    'Enter the authorization code: ')
"""str: The message to display when prompting the user for the
authorization code. Used only by the console strategy."""

_DEFAULT_WEB_SUCCESS_MESSAGE = (
    'The authentication flow has completed. You may close this window.')

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client()

import requests, re, urllib


class _WSGIRequestHandler(wsgiref.simple_server.WSGIRequestHandler):
    """Custom WSGIRequestHandler.

    Uses a named logger instead of printing to stderr.
    """

    def log_message(self, format, *args):
        # pylint: disable=redefined-builtin
        # (format is the argument name defined in the superclass.)
        _LOGGER.info(format, *args)


class _RedirectWSGIApp(object):
    """WSGI app to handle the authorization redirect.

    Stores the request URI and displays the given success message.
    """

    def __init__(self, success_message):
        """
        Args:
            success_message (str): The message to display in the web browser
                the authorization flow is complete.
        """
        self.last_request_uri = None
        self._success_message = success_message

    def __call__(self, environ, start_response):
        """WSGI Callable.

        Args:
            environ (Mapping[str, Any]): The WSGI environment.
            start_response (Callable[str, list]): The WSGI start_response
                callable.

        Returns:
            Iterable[bytes]: The response body.
        """
        start_response('200 OK', [('Content-type', 'text/plain')])
        self.last_request_uri = wsgiref.util.request_uri(environ)
        return [self._success_message.encode('utf-8')]


def get_definitions(word):
    html = requests.get('http://www.dictionary.com/browse/%s' % word).text
    if 'There are no results for: ' in html: return []
    definition_block = html.split('<div class="def-list">')[1].split('</section>')[0]
    definitions = definition_block.split('<div class="def-set">')[1:]
    strings = [re.sub(' +', ' ', re.sub('<[^<]+?>', '',
                                        i.split('<div class="def-content">')[1].split('</div>')[0].strip()).strip()) for
               i in definitions]
    return strings


def get_urban_definitions(word):
    url = 'http://api.urbandictionary.com/v0/define?term=%s' % urllib.parse.quote(word)
    dat = requests.get(url).json()
    return dat['list']


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
    if 'what is' in message.content.lower():
        question = message.content.lower()
        question = str(question).replace('what is', '')
        word = question
        definition = get_urban_definitions(word)[0]
        for i in range(len(definition['definition']) // 2000 + 1):
            await message.channel.send('```%s```' % definition['definition'][i * 2000:i * 2000 + 2000])
        await message.channel.send('```examples: %s```' % definition['example'])
        e = discord.Embed(
            title=str(message.content),
            colour=0xE5E242,
            url=f"https://www.urbandictionary.com/define.php?term={question.replace(' ', '+')}"
        )

        # await message.channel.send(embed=e)
    print(type(message.channel))
    if '?' in message.content and message.channel.id == 733176263212138516 and message.author.id != 350993177316032513:
        print("send message")
        await message.channel.send("<@!350993177316032513> and <@!650055963726053376>")
    if '!get classwork' in message.content.lower() or '!get classes' in message.content.lower():
        creds = None
        flow = Flow.from_client_secrets_file('credentials.json', SCOPES)
        wsgi_app = _RedirectWSGIApp(_DEFAULT_WEB_SUCCESS_MESSAGE)
        local_server = wsgiref.simple_server.make_server(
            'localhost', 8080, wsgi_app, handler_class=_WSGIRequestHandler)

        flow.redirect_uri = 'http://{}:{}/'.format(
            'localhost', local_server.server_port)
        kwargs = {'host': 'localhost', 'port': 8080,
                  'authorization_prompt_message': _DEFAULT_AUTH_PROMPT_MESSAGE,
                  'success_message': _DEFAULT_WEB_SUCCESS_MESSAGE,
                  'open_browser': False}
        auth_url, _ = flow.authorization_url(**kwargs)
        print(_DEFAULT_AUTH_PROMPT_MESSAGE.format(url=auth_url))
        e = discord.Embed(
            title="Please allow AI Tools Bot to view your classwork through this link:",
            colour=0x49FC58,
            url=auth_url)
        await message.channel.send(embed=e)
        local_server.handle_request()

        # Note: using https here because oauthlib is very picky that
        # OAuth 2.0 should only occur over https.
        authorization_response = wsgi_app.last_request_uri.replace(
            'http', 'https')
        flow.fetch_token(authorization_response=authorization_response)

        creds = flow.credentials
        service = build('classroom', 'v1', credentials=creds)

        results = service.courses().list(pageSize=10).execute()
        courses = results.get('courses', [])

        if not courses:
            print('No Courses Found.')
        else:
            print('Courses:')
            if '!get classes' in message.content.lower():
                for course in courses:
                    print(course['name'])
                    print(course[u'id'])
                    await message.channel.send(f"**Class Name: **{course['name']}"
                                               f"**Class ID (use this in the command): **{course[u'id']}")
            else:
                msg = message.content.lower()
                msg = msg.replace('!get classwork', '')
                msg = msg.split(';')
                for course in msg:
                    for i in courses:
                        if i[u'id'] == course:
                            await message.channel.send(f"**Classwork for course id: {i['name']}**")
                    try:
                        course_work_results = service.courses().courseWork().studentSubmissions().list(
                            courseId=int(course),
                            courseWorkId='-',
                            userId="me").execute()
                        print(course_work_results)
                        try:
                            for item in course_work_results['studentSubmissions']:
                                if str(item.get('state', '-')) == 'TURNED_IN' or str(
                                        item.get('state', '-')) == 'RETURNED':
                                    continue
                                for subitem in item.keys():
                                    if subitem == 'updateTime' or subitem == 'alternateLink':
                                        print(str(subitem) + str(item.get(subitem, '-')))
                                        await message.channel.send(str(subitem) + ": " + str(item.get(subitem, '-')))
                        except KeyError:
                            pass
                    except:
                        await message.channel.send("As a reminder: Please input using the correct format")
    if '!replicate' in message.content.lower():
        uid = message.mentions[0]
        id = message.mentions[0].id
        # print(f"uid: {uid}")
        dataset = []
        mlen = []
        channel = message.channel
        async for message in channel.history(limit=10000):
            # print(message.author)
            if message.author == uid:
                mlen.append(len(message.content.split()))
                dataset.append(message.content.lower())
        # print(dataset)
        tokenizer = Tokenizer()

        data = dataset
        corpus = data

        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1
        print(total_words)
        # print(tokenizer.word_index)

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
        # await message.channel.send("Model: " + str(short_model_summary))
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
