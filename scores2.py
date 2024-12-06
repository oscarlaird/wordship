# %%
import nest_asyncio
nest_asyncio.apply()
import torch
torch.cuda.is_available()


# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import time
import numpy as np

# Global variables for model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A small Llama model from Hugging Face
tokenizer = None
model = None
cache = None

def initialize_model_and_tokenizer():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.cuda()
initialize_model_and_tokenizer()

#%%
del tokenizer, model
import gc
gc.collect()
#%%
import time
start_time = time.time()
prompt = "The capital of England London. The capital of China Beijing. The capital of"
countries = ["France", "Spain", "Italy", "Germany", "Russia"]
countries *= 100
# 500 prompts / 3 seconds
# when using the same 1 token prompt / 1 token output, we get 2000/sec
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
prompt_output = model(**inputs)
prompt_output.past_key_values
#%%
start_time = time.time()
prompt_suffixes = tokenizer(countries, return_tensors="pt", add_special_tokens=False).to("cuda")
repeated_past_key_values = [(k.repeat(len(countries), 1, 1, 1), v.repeat(len(countries), 1, 1, 1)) for k, v in prompt_output.past_key_values]
print(repeated_past_key_values[0][0].shape)
print(prompt_suffixes.input_ids.shape)
with torch.no_grad():
    output = model(**prompt_suffixes, past_key_values=repeated_past_key_values)
preds = output.logits.argmax(dim=-1)[:,0]
for i, pred in enumerate(preds):
    print(tokenizer.decode(pred, skip_special_tokens=True))
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# %%
prompts = [prompt + country for country in countries]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
print(inputs)
print(help(model.generate))
outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
print(outputs)
for i, output in enumerate(outputs):
    print(tokenizer.decode(output, skip_special_tokens=True))
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


# %%
