import json
import openai
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize


def hostile_or_personal(text: str) -> bool:
    """
    This tests the text to see if it is hostile
    or references a person. This test is done by
    your computer and should be done before
    testing with OpenAI.

    :param text: This is the text you want to test.
    :return: bool regarding status. If true, reject text
    """

    # Sentiment Analysis
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    negative_score = scores['neg']

    # Named Entity
    named_entities = []
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            named_entities.append(chunk.label())

    # Check for manipulation
    if (negative_score > 0.5) or ('PERSON' in named_entities):
        return True
    else:
        return False


class Translator():
    """
    The translator class translates text into latin if it
    passes some checks. The checks are not full-proof, so
    be careful with what you allow to be sent to OpenAI.

    Functions:
    flagged_by_openai(text): This will test if your provided text was flagged by openai.
    translate(text): This will translate the text you provide into Latin if it passes checks.
    """
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    api_key = None
    
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def flagged_by_openai(self, text: str) -> bool:
        """
        Tests text using OpenAI api. If it fails or is flagged, return false.

        :param text:
        :return: bool representing if the material is flagged or something else.
        A return of False means the text is good to go
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {"input": text}
            response = requests.post("https://api.openai.com/v1/moderations", headers=headers, json=data)
            return json.loads(response.text)['results'][0]['flagged']  # This is a bool

        except Exception as e:
            print(f'[X] Failed to test with OpenAI. Key might be invalid.')
            return True

    def translate(self, text) -> str:
        """
        This translates text into Latin

        :param text: Whatever text you want translated into Latin
        :return: Latin text if accepted or an error message
        """
        if not hostile_or_personal(text) and not self.flagged_by_openai(text):
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f' Translate "{text}" into Latin please.',
                temperature=0.7,
                max_tokens=3625,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return json.loads(str(response))['choices'][0]['text']
        else:
            return '[X] Text flagged, no request sent.'
