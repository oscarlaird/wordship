# %%
category = "animals"
x_axis = ("terrestrial", "aquatic")
y_axis = ("large", "small")
# Which words are animals?
# What are their scores on the x and y axes?

# %%
# Which words are animals?
# - compare vector similarity between words and "animals"
import numpy as np
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try: 
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
            except:
                print('error')
    return embeddings

# Load the 100-dimensional GloVe embeddings
glove_file = './glove.840B.300d.txt'
glove_embeddings = load_glove_embeddings(glove_file)

# Check the embedding for a specific word
word = 'king'
print(f'Embedding for "{word}":\n', glove_embeddings.get(word))

# %%
def embed(word):
    return glove_embeddings.get(word)
embed("king") @ embed("fence")

# %%
words = list(glove_embeddings.keys())
len(words)

# %%
import wordfreq
words = wordfreq.top_n_list('en', 100000)
print("len(words)", len(words))
words_in_glove = [word for word in words if word in glove_embeddings]
print("len(words_in_glove)", len(words_in_glove))
word_embeddings = [embed(word) for word in words_in_glove]
print("word_embeddings[1].shape", word_embeddings[1].shape)
print("np.array(word_embeddings).shape", np.array(word_embeddings).shape)

# %%
query_vector = embed(category)  # animal
word_similarities = word_embeddings @ query_vector
word_similarities.shape

#%%
# Sort words by similarity score
sorted_indices = np.argsort(-word_similarities)  # Negative to sort in descending order
sorted_words = [words_in_glove[i] for i in sorted_indices]

# %%
k = 5000
print("Most similar words of ranks ", k, "to", k+16, ":", sorted_words[k:k+16])

reference_animals = ("grasshopper", "kookaburra", "minnow", "aardvark", "crustacean", "seagull", "trilobyte", "fly", "history", "kingfisher", "seagull", "wildebeest", "pelican", "penguin", "penguins", "ostrich", "ostriches", "kangaroo", "kangaroos", "duck", "ducks", "mammoth", "mammoths", "goose", "wolf", "werewolves", "sloth", "sloths")
for word in sorted_words:
    if word in reference_animals:
        print(f"{word} is animal, with rank {sorted_words.index(word)}")

# %%
import nest_asyncio
nest_asyncio.apply()
import torch
import gc
gc.collect()
torch.cuda.is_available()


# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import time
import numpy as np

# Global variables for model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A small Llama model from Hugging Face
# model_name = "Qwen/Qwen2.5-3B"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = None
model = None
cache = None

def initialize_model_and_tokenizer():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.to("cuda")
initialize_model_and_tokenizer()

# %%
top_30k_animals = sorted_words[:30000]
top_30k_animals_output = tokenizer(top_30k_animals, return_tensors="pt", padding=True)#.to("cuda")
max_token_length = top_30k_animals_output.input_ids.shape[1]
token_counts = top_30k_animals_output.attention_mask.sum(dim=1)
two_tokens = top_30k_animals_output.input_ids[token_counts == 3][:,-2:]

# %%
batch_size = 100
prompt = """
Does this word refer to animals? starving no
Does this word refer to animals? unborn no
Does this word refer to animals? behave no
Does this word refer to animals? livers no
Does this word refer to animals? swine yes
Does this word refer to animals? chimps yes
Does this word refer to animals? iguana yes
Does this word refer to animals? beast yes
Does this word refer to animals? mothers no
Does this word refer to animals? fairies no
Does this word refer to animals? neurons no
Does this word refer to animals? humanely no
Does this word refer to animals? disease no
Does this word refer to animals? geese yes
Does this word refer to animals? fruits no
Does this word refer to animals? aviary no
Does this word refer to animals? lovable no
Does this word refer to animals? moths yes
Does this word refer to animals? abused no
Does this word refer to animals? blankets no
Does this word refer to animals? chicken yes
Does this word refer to animals? lizard yes
Does this word refer to animals? slaughtering 
Does this word refer to animals? prey no
Does this word refer to animals? mink yes
Does this word refer to animals? mortality no
Does this word refer to animals? boars yes
Does this word refer to animals? natures no
Does this word refer to animals? fertility no
Does this word refer to animals? gestation no
Does this word refer to animals? monsters no
Does this word refer to animals? hungry no
Does this word refer to animals? environmental
Does this word refer to animals? freshwater no
Does this word refer to animals? herbs no
Does this word refer to animals? colorful no
Does this word refer to animals? aquarium no
Does this word refer to animals? populations no
Does this word refer to animals? penguins yes
Does this word refer to animals? larvae yes
Does this word refer to animals? grasslands no
Does this word refer to animals? isolated no
Does this word refer to animals? caretakers no
Does this word refer to animals? drawings no
Does this word refer to animals? illness no
Does this word refer to animals? minerals no
Does this word refer to animals? infected no
Does this word refer to animals? puppies yes
Does this word refer to animals? beasts yes
Does this word refer to animals? dangerous no
Does this word refer to animals?"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
prompt_output = model(**inputs)
print(len(prompt_output.past_key_values))
repeated_past_key_values = [(k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1)) for k, v in prompt_output.past_key_values]
repeated_past_key_values[0][0].shape

# %%
from tqdm import tqdm
yes_token = tokenizer.encode("this yes")[-1]
print(yes_token)
no_token = tokenizer.encode("this no")[-1]
print(no_token)
is_animal_selector = np.zeros(len(top_30k_animals_output.input_ids), dtype=bool)
start_time = time.time()
for word_token_length in range(1, max_token_length+1):
    words_with_length_selector = token_counts == word_token_length+1
    num_words_with_length = words_with_length_selector.sum()
    words_with_length = top_30k_animals_output.input_ids[words_with_length_selector][:,-word_token_length:]
    print(f"Processing {num_words_with_length} words with {word_token_length} tokens")
    batch_animal_selector = np.zeros(num_words_with_length, dtype=bool)
    for i in tqdm(range(0, num_words_with_length, batch_size)):
        batch = words_with_length[i:i+batch_size]
        batch = batch.to("cuda")
        if len(batch) < batch_size:
            break  # since repeated_past_key_values won't have right shape and we can skip a few at the end
        with torch.no_grad():
            output = model(batch, past_key_values=repeated_past_key_values)
        for j in range(batch_size):
            yes_logits = float(output.logits[j,-1,yes_token])
            no_logits = float(output.logits[j,-1,no_token])
            batch_animal_selector[i+j] = yes_logits > no_logits - 0.2
        del batch
        del output
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    print(is_animal_selector.sum(), is_animal_selector.shape, batch_animal_selector.sum(), batch_animal_selector.shape)
    is_animal_selector[words_with_length_selector] = batch_animal_selector
print(f"Time taken: {time.time() - start_time:.2f} seconds")

# %%
animals = [a for i, a in enumerate(top_30k_animals) if is_animal_selector[i]]
print(len(animals))
for a in reference_animals:
    print(a, a in set(animals))
# if it is not there, we can try the plural too

# %%
# score the animals on the x and y axes
# at this point it may be affordable to use gpt-4o to score and write json
# we can ask for a tsv
# this is 2 tokens per animal, another 2 tokens for the two axes, 1 token for newline
# so writing about 25K tokens, which is $0.25 in write costs, realistically up to $0.50

# %%
# score with gpt-4o
prompt_template = """
Output json. You will be provided a list of 100 animal like things.
For each output its size as: tiny, small, medium, big, huge
For each output whether it is terrestrial or aquatic as: wholly_terrestrial, mostly_terrestrial, mostly_aquatic, wholly_aquatic
Output a newline separated, space_delimited file like the following example.
DO NOT OUTPUT ANY OTHER TEXT; DO NOT MAKE NOTES OR COMMENTS; THIS MUST BE A CLEAN FILE.
ONLY USE THE PROVIDED OPTIONS FOR SIZE AND AQUATICITY.
Example:
pony medium wholly_terrestrial
duck small mostly_aquatic
whale huge wholly_aquatic
crocodile big mostly_aquatic
ant tiny wholly_terrestrial
moose huge wholly_terrestrial
ferret small wholly_terrestrial
rabbit small wholly_terrestrial
cat small wholly_terrestrial
pig medium wholly_terrestrial
hippopotamus huge wholly_aquatic
minnow tiny wholly_aquatic
giraffe big wholly_terrestrial

Here are the 100 animals:
{batch_animals}
"""

# %%
from importlib import reload
import os
import nest_asyncio
nest_asyncio.apply()
import asyncio
from openai import AsyncOpenAI
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
async def get_batch_scores(batch_animals):
    try:
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt_template.format(batch_animals=str(batch_animals))
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0.0,
        )
        response_content = response.choices[0].message.content
        print("Finished processing batch")
        return response_content
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

animals = list(animals)
async def process_all_batches():
    batch_size = 100
    num_batches = len(animals) // batch_size
    
    # Create list of batch tasks
    tasks = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_animals = animals[start_idx:end_idx]
        
        print(f"Creating task for batch {i+1}/{num_batches}")
        tasks.append(get_batch_scores(batch_animals))

        # wait 0.1s
        await asyncio.sleep(0.1)

    # Run all tasks concurrently and gather results
    print("Running all batches concurrently...")
    all_responses = await asyncio.gather(*tasks)
    
    # Filter out None responses
    return [r for r in all_responses if r is not None]

# Run the async code
responses = asyncio.run(process_all_batches())

# %%
size_score_dict = {
    'tiny': -14,
    'small': -5,
    'medium': 0,
    'big': 5,
    'large': 5,
    'huge': 14
}
aquatic_score_dict = {
    'wholly_terrestrial': -10,
    'mostly_terrestrial': -3,
    'mostly_aquatic': 3,
    'mainly_aquatic': 3,
    'wholly_aquatic': 10,
}
coordinates = {}
for batch_response in responses:
    for line in batch_response.splitlines():
        args = line.split()
        if len(args) != 3:
            print(f"Invalid line: {line}")
            continue
        animal_name, size, aquatic = args
        if size not in size_score_dict or aquatic not in aquatic_score_dict:
            print(f"Unknown size or aquatic: {animal_name} {size} {aquatic}")
            continue
        size_score = size_score_dict[size]
        aquatic_score = aquatic_score_dict[aquatic]
        radius = (size_score**2 + aquatic_score**2)**0.5
        size_score = size_score / radius * 10.0
        aquatic_score = aquatic_score / radius * 10.0
        size_score = int(size_score * 100) / 100
        aquatic_score = int(aquatic_score * 100) / 100
        coordinates[animal_name] = (-aquatic_score/100, -size_score/100)
        if animal_name[-1] == 's':
            coordinates[animal_name[:-1]] = coordinates[animal_name]

# %%
import json
print(f"Created coordinates for {len(coordinates)} animals")

# write coordinates to animals_coordinates.json
with open('animals_coordinates.json', 'w') as f:
    json.dump({k: [round(v[0], 2), round(v[1], 2)] for k, v in coordinates.items()}, f, indent=2)