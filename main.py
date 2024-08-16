from transformers import MarianMTModel, MarianTokenizer

# Dictionary mapping language names to their corresponding MarianMT model names
language_model_mapping = {
    'french': 'Helsinki-NLP/opus-mt-it-fr'
    # 'german': 'Helsinki-NLP/opus-mt-en-de',
    # 'french': 'Helsinki-NLP/opus-mt-en-fr',
    # 'italian': 'Helsinki-NLP/opus-mt-en-it',
    # 'spanish': 'Helsinki-NLP/opus-mt-en-es',
    # 'portuguese': 'Helsinki-NLP/opus-mt-en-pt',
    # 'dutch': 'Helsinki-NLP/opus-mt-en-nl',
    # 'russian': 'Helsinki-NLP/opus-mt-en-ru',
    # 'polish': 'Helsinki-NLP/opus-mt-en-pl',
    # 'czech': 'Helsinki-NLP/opus-mt-en-cs',
    # 'swedish': 'Helsinki-NLP/opus-mt-en-sv',
    # 'danish': 'Helsinki-NLP/opus-mt-en-da',
    # 'norwegian': 'Helsinki-NLP/opus-mt-en-no',
    # 'finnish': 'Helsinki-NLP/opus-mt-en-fi',
    # 'greek': 'Helsinki-NLP/opus-mt-en-el',
    # 'hungarian': 'Helsinki-NLP/opus-mt-en-hu',
    # 'romanian': 'Helsinki-NLP/opus-mt-en-ro',
    # 'bulgarian': 'Helsinki-NLP/opus-mt-en-bg',
    # 'croatian': 'Helsinki-NLP/opus-mt-en-hr',
    # 'slovak': 'Helsinki-NLP/opus-mt-en-sk',
    # 'slovenian': 'Helsinki-NLP/opus-mt-en-sl',
    # 'estonian': 'Helsinki-NLP/opus-mt-en-et',
    # 'latvian': 'Helsinki-NLP/opus-mt-en-lv',
    # 'lithuanian': 'Helsinki-NLP/opus-mt-en-lt',
}

# Function to load the appropriate model and tokenizer
def load_model_and_tokenizer(target_language):
    model_name = language_model_mapping.get(target_language.lower())
    if not model_name:
        raise ValueError(f"Unsupported language: {target_language}")
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Function to translate text
def translate_text(text, target_language):
    model, tokenizer = load_model_and_tokenizer(target_language)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example usage
if __name__ == "__main__":
    user_preference = input("Enter your preferred language (e.g., German, Italian, French, etc.): ").strip()
    text_to_translate = input("Enter the text you want to translate: ").strip()
    
    try:
        translation = translate_text(text_to_translate, user_preference)
        print(f"Translated text in {user_preference}: {translation}")
    except ValueError as e:
        print(e)
