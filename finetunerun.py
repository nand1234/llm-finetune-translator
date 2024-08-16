import torch
from transformers import MarianMTModel, MarianTokenizer

# Dictionary mapping language names to their corresponding MarianMT model paths
language_model_mapping = {
    'french': 'smartbox-NLP-opus-mt-it-fr.pth'
    # Add other language mappings and their corresponding model paths here
}

# Function to load the appropriate model and tokenizer
def load_model_and_tokenizer(target_language):
    model_path = language_model_mapping.get(target_language.lower())
    if not model_path:
        raise ValueError(f"Unsupported language: {target_language}")

    # Initialize the MarianMT model using a compatible pre-trained model
    base_model_name = 'Helsinki-NLP/opus-mt-it-fr'  # Ensure this matches your model architecture
    model = MarianMTModel.from_pretrained(base_model_name)
    
    # Load the state dict from the .pth file with weights_only=True
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)

    # Load the tokenizer from the same pre-trained model
    tokenizer = MarianTokenizer.from_pretrained(base_model_name, clean_up_tokenization_spaces=True)
    
    return model, tokenizer

# Function to translate text
def translate_text(text, target_language):
    model, tokenizer = load_model_and_tokenizer(target_language)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example usage
if __name__ == "__main__":
    user_preference = input("Enter your preferred language (e.g., French): ").strip()
    text_to_translate = input("Enter the text you want to translate: ").strip()
    
    try:
        translation = translate_text(text_to_translate, user_preference)
        print(f"Translated text in {user_preference}: {translation}")
    except ValueError as e:
        print(e)
