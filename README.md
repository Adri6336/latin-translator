# latin-translator
Uses OpenAI's GPT-3 to translate text into Latin. You must have a valid GPT-3 API key to use. 
Instantiate Translator class with the as follows:

    from main import *
    t = Translator(key)
    t.translate('Wow, this is in Latin!!!')
    >>> \n\n"Vae, hoc est in Latina!"


The Translator class will try to determine if your text is in line with OpenAI's usage policies 
before sending the request to GPT-3. The safety tests are not fullproof, so be careful with what
you allow to be requested.
