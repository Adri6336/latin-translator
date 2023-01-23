import json
import os
import openai
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize

class Translator():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    api_key = None
    
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def testText(self, text):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {"input": text}
        response = requests.post("https://api.openai.com/v1/moderations", headers=headers, json=data)
        return json.loads(response.text)


    def detect_manipulative_text(self, text):

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
        if ((negative_score > 0.5) or ('PERSON' in named_entities)):
            return True
        else:
            return False

    def translateToLatin(self, text):
        if not self.testText(text)['results'][0]['flagged'] and not self.detect_manipulative_text(text):
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
            return 'Text flagged, no request sent!'