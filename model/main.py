from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
device = "cpu"

def main():

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    messages = [{"role": "user", "content": "What is the capital of France."}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, temperature=0.2, top_p=0.9, do_sample=True)
    
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()