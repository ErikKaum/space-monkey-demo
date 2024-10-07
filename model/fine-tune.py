import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader

# Your dataset
# dataset = [
#     ("Assign the string 'hello world' to a variable named hello and print it", "let hello = \"hello world\"; hello;"),
#     ("Write a program that adds 5 and 3", "5 + 3"),
#     ("Assign 10 to x and 5 to y, then add them", "let x = 10; let y = 5; x + y"),
#     ("Define a function that adds two numbers and call it with 7 and 3", "let add = fn(a, b) { a + b }; add(7, 3)"),
#     ("Use an if-else statement to check if 10 is greater than 5", "if (10 > 5) {\"Ten is greater\" } else { \"Five is greater\" }"),
#     ("Create an array with numbers 1, 2, 3 and access the second element", "let arr = [1, 2, 3]; arr[1]"),
#     ("Use the built-in 'len' function to check the length of this array [1, 2, 3]", "let arr = [1, 2, 3]; len(arr)"),
# ]

dataset = [
    "let hello = \"hello world\"; hello;",
    "5 + 3",
    "let x = 10; let y = 5; x + y",
    "let add = fn(a, b) { a + b }; add(7, 3)",
    "if (10 > 5) {\"Ten is greater\" } else { \"Five is greater\" }",
    "let arr = [1, 2, 3]; arr[1]",
    "let arr = [1, 2, 3]; len(arr)",
]

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]
        input_text = f"{code}"
        encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

# Initialize model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Prepare dataset and dataloader
custom_dataset = CustomDataset(dataset, tokenizer)
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("Fine-tuning complete. Model saved.")