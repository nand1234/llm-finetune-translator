import torch
from transformers import MarianMTModel, MarianTokenizer

# Directory where the checkpoint is saved
checkpoint_dir = "results_cpu_layers/checkpoint-5"

# Load the model and tokenizer from the checkpoint
model = MarianMTModel.from_pretrained(checkpoint_dir)
tokenizer = MarianTokenizer.from_pretrained(checkpoint_dir)

# Save the model's state dictionary to a .pth file
output_pth_file = "smartbox-NLP-opus-mt-it-fr.pth"
torch.save(model.state_dict(), output_pth_file)

print(f"Model state dictionary saved to {output_pth_file}")
