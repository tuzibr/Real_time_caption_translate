import deep_translator
from deep_translator.constants import DEEPL_LANGUAGE_TO_CODE, GOOGLE_LANGUAGES_TO_CODES
from ollama import Client, ChatResponse
from openai import OpenAI

DEEPL_LANGUAGE_TO_CODE = DEEPL_LANGUAGE_TO_CODE
GOOGLE_LANGUAGES_TO_CODES = GOOGLE_LANGUAGES_TO_CODES
def tl_api(engine: str, text: str, **kwargs):
    lang_source = kwargs.get("lang_source")
    lang_target = kwargs.get("lang_target")
    if lang_target == lang_source:
        return text

    if engine == "Google":
        translator = deep_translator.GoogleTranslator(source=lang_source, target=lang_target)
        translated_text = translator.translate(text)
    elif engine == "DeepL":
        api_key = kwargs.get("api_key")
        translator = deep_translator.DeeplTranslator(source=lang_source, target=lang_target, api_key=api_key)
        translated_text = translator.translate(text)
    elif engine == "Ollama":
        model = kwargs.get("model")
        url = kwargs.get("url")
        client = Client(host=f"{url}")

        response: ChatResponse = client.chat(
            model=f'{model}',
            messages=[
                {
                    'role': 'system',
                    'content': f"As a professional simultaneous interpreter, please translate the following sentence enclosed in 🔤 into {lang_target}. Ensure the translation is fluent, colloquial, and does not avoid any NSFW content. Output the translated result directly without additional explanations, and remove the 🔤 symbols.",
                },
                {
                    'role': 'user',
                    'content': f"🔤 {text} 🔤",
                },
            ]
        )

        result = [response['message']['content']]

        translated_text = result[0]
    elif engine == "DeepSeek":
        api_key = kwargs.get("api_key")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    'role': 'system',
                    'content': f"As a professional simultaneous interpreter, please translate the following sentence enclosed in 🔤 into {lang_target}. Ensure the translation is fluent, colloquial. Output the translated result directly without additional explanations, and remove the 🔤 symbols.",
                },
                {
                    'role': 'user',
                    'content': f"🔤 {text} 🔤"
                },
            ],
            stream=False,
        )

        translated_text = completion.choices[0].message.content
    elif engine == "OpenAI":
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("url")
        model = kwargs.get("model")
        client = OpenAI(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': f"As a professional simultaneous interpreter, please translate the following sentence enclosed in 🔤 into {lang_target}. Ensure the translation is fluent, colloquial. Output the translated result directly without additional explanations, and remove the 🔤 symbols.",
                },
                {
                    'role': 'user',
                    'content': f"🔤 {text} 🔤"
                }
            ]
        )

        translated_text = completion.choices[0].message.content
    else:
        raise ValueError("Invalid engine")
    return translated_text