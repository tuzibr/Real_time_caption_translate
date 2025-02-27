import deep_translator
from deep_translator.constants import DEEPL_LANGUAGE_TO_CODE, GOOGLE_LANGUAGES_TO_CODES
from ollama import Client, ChatResponse

DEEPL_LANGUAGE_TO_CODE = DEEPL_LANGUAGE_TO_CODE
GOOGLE_LANGUAGES_TO_CODES = GOOGLE_LANGUAGES_TO_CODES
def tl_api(engine: str, text: str, **kwargs):
    if engine == "Google":
        lang_source = kwargs.get("lang_source")
        lang_target = kwargs.get("lang_target")
        translator = deep_translator.GoogleTranslator(source=lang_source, target=lang_target)
        translated_text = translator.translate(text)
    elif engine == "DeepL":
        api_key = kwargs.get("api_key")
        lang_source = kwargs.get("lang_source")
        lang_target = kwargs.get("lang_target")
        translator = deep_translator.DeeplTranslator(source=lang_source, target=lang_target, api_key=api_key)
        translated_text = translator.translate(text)
    elif engine == "Ollama":
        model = kwargs.get("model")
        lang_target = kwargs.get("lang_target")
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
    else:
        raise ValueError("Invalid engine")
    return translated_text