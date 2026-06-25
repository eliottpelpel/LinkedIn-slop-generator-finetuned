import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

ADAPTER_PATH = "./linkedin-slop-lora-final"
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on {device}...")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map={"": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)


def main():
    if len(sys.argv) < 2:
        print('Usage: python generate.py "something mundane that happened to you"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    model, tokenizer, device = load_model()

    print(f'\nPrompt: "{prompt}"\n')

    print("─" * 60)
    print("BASE MODEL")
    print("─" * 60)
    with model.disable_adapter():
        base_out = generate(model, tokenizer, device, prompt)
    print(base_out)

    print("\n" + "─" * 60)
    print("FINE-TUNED (linkedin-slop-lora)")
    print("─" * 60)
    tuned_out = generate(model, tokenizer, device, prompt)
    print(tuned_out)


if __name__ == "__main__":
    main()
