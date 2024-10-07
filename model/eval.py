from transformers import GPT2LMHeadModel, GPT2Tokenizer 

def main():

    model_name = "fine_tuned_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    input_text = "let hello"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.5
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()