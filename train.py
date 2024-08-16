import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset


# Ensure that we are using CPU
device = torch.device('cpu')
data = pd.read_csv('data.csv')

# Initialize TensorBoard writer
writer = SummaryWriter('runs/marianmt_finetuning_cpu_layers')

# Load the MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-it-fr'  # italian to french example
tokenizer = MarianTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=True)
model = MarianMTModel.from_pretrained(model_name)

# Move model to CPU
model.to(device)
data = data.fillna('')

# Load your dataset (example data, replace with your own)
# Convert the DataFrame to a Dataset
dataset = Dataset.from_dict({
    "source": data["source"].tolist(),
    "target": data["target"].tolist()
})

# Split the dataset into training and validation
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Tokenize the dataset
def preprocess_function(examples):
    inputs = tokenizer(examples['source'], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples['target'], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_cpu_layers",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir="./logs_cpu_layers",
    logging_steps=10,
    use_cpu=True  # Force using CPU even if a GPU is available
)

# Custom Trainer to log losses, gradients, and weights to TensorBoard
class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log(self, logs):
        super().log(logs)
        if "loss" in logs:
            writer.add_scalar("Training Loss", logs["loss"], self.state.global_step)
        if "eval_loss" in logs:
            writer.add_scalar("Validation Loss", logs["eval_loss"], self.state.global_step)

    def training_step(self, model, inputs):
        """Override to log gradients and weights."""
        outputs = model(**inputs)
        loss = outputs.loss

        # Backpropagation
        loss.backward()

        # Log gradients and weights for each layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, self.state.global_step)
                writer.add_histogram(f'{name}.weight', param, self.state.global_step)

        return loss.detach()

    def evaluation_step(self, model, inputs, prediction_loss_only):
        """Override to log evaluation loss."""
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            # Log validation loss
            writer.add_scalar('Validation Loss', loss.item(), self.state.global_step)

            # Log weights for each layer during evaluation
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}.weight', param, self.state.global_step)

        return loss

# Initialize the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Close the TensorBoard writer
writer.close()
