import deepl
import base64
import json
import os
import requests
from PIL import Image
from io import BytesIO
from google.cloud import translate_v2 as google_translate_v2
from .api_keys import APIKEY_OPENAI, APIKEY_DEEPL

DEEPL_SUPPORTED_LANGUAGES = ["AR", "BG", "CS", "DA", "DE", "EL", "EN-GB", "EN-US", "ES", "ET", "FI", "FR", "HU", "ID", "IT", "JA", "KO", "LT", "LV", "NB", "NL", "PL", "PT-BR", "PT-PT", "RO", "RU", "SK", "SL", "SV", "TR", "UK", "ZH"]

def encode_image(image: str | Image.Image) -> str:
    if isinstance(image, str):
        with open(image, "rb") as img:
            return base64.b64encode(img.read()).decode('utf-8')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
  
def gpt4o_completion(image_path, text, system_text):
    base64_image = encode_image(image_path)
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {APIKEY_OPENAI}"
    }
    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": system_text
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": text
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"
            }
            }
        ]
        }
    ],
    "logprobs": True,
    "top_logprobs": 2,
    "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        print(f"HTTP error {response.status_code}: {response.text}")
        return {"response.text": response.text}
    else:
        return response.json()

# def google_translate(text, target_language):
#     translate_client = google_translate_v2.Client()
#     result = translate_client.translate(text, target_language=target_language)
#     return result["translatedText"]

def deepl_translate(text: str, target_lang: str, source_lang: str | None = None):
    translator = deepl.Translator(APIKEY_DEEPL)
    result = translator.translate_text(text, target_lang=target_lang, source_lang=source_lang)
    return result.text

def google_translate(text: str, target_lang: str, source_lang: str | None = None) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = google_translate_v2.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target_lang, source_language=source_lang)

    # print("Text: {}".format(result["input"]))
    # print("Translation: {}".format(result["translatedText"]))
    # print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result["translatedText"]

def google_translate_with_src_lang(text: str, target_lang: str, source_lang: str = "en") -> dict:
    """Translates text into the target language with source language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    translate_client = google_translate_v2.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target_lang, source_language=source_lang)

    # print("Text: {}".format(result["input"]))
    # print("Translation: {}".format(result["translatedText"]))
    # print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result

def query_translation(prompt, target_lang, backend="deepl", source_lang=None):
    if backend == "deepl":
        translate = deepl_translate
    elif backend == "google":
        translate = google_translate
    else:
        raise ValueError(f"Invalid backend: {backend}")
    
    if target_lang not in DEEPL_SUPPORTED_LANGUAGES:
        translate = google_translate # Fallback to Google Translate
        
    if target_lang == "EN-US": # and prompt != "other_yes" and prompt != "other_no":
        return prompt

    file_path = "cached_translation.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            translations = json.load(file)
    else:
        translations = {}
    
    if target_lang not in translations:
        translations[target_lang] = {}
        yes_tokens = ["yes", "Yes", "YES"]
        no_tokens = ["no", "No", "NO"]
        for token in yes_tokens + no_tokens:
            translations[target_lang][token] = translate(token, target_lang, source_lang)
    
    # if prompt == "other_yes" or prompt == "other_no":
    #     return translations[target_lang][prompt] if prompt in translations[target_lang] else []

    if prompt in translations[target_lang]:
        return translations[target_lang][prompt]
    else:
        translated_prompt = translate(prompt, target_lang, source_lang)
        translations[target_lang][prompt] = translated_prompt
        with open(file_path, 'w') as file:
            json.dump(translations, file, indent=4)
        return translated_prompt
    
def query_translation_back_to_en(prompt, source_lang):
    target_lang = "en"
    translate = google_translate_with_src_lang

    file_path = "cached_translation_back_to_en.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            translations = json.load(file)
    else:
        translations = {}

    if source_lang not in translations:
        translations[source_lang] = {}

    if prompt in translations[source_lang]:
        return translations[source_lang][prompt]["translatedText"]
    else:
        translated_prompt = translate(prompt, target_lang, source_lang)
        translations[source_lang][prompt] = translated_prompt
        with open(file_path, 'w') as file:
            json.dump(translations, file, indent=4)
        return translated_prompt["translatedText"]