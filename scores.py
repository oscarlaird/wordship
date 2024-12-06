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

#%%
top100_animals = sorted_words[:200]
top100_animals

# %%
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

# axis_string = "terrestrial to aquatic"
# axis_examples = """
# penguin: -1
# dog: -10
# hippopotamus: 5
# fish: 10
# mammoth: -10
# stapler: 0
# crocodile: 10
# """
axis_string = "large to small"
axis_examples = """
human: 1
dog: -2
elephant: 10
whale: 10
moose: 7
deer: 5
cat: -4
rat: -6
ant: -10
"""

prompt_template = """
Output json. You will be provided a list of 100 animal like things.
For each output where it is on a scale of -10 to +10 on a scale of {axis_string}.
Assign a score of 0 to irrelevant things.
Do not assign a score of 0 to things that are relevant; instead assign 1 or -1.
Output valid json. Do not include any other text e.g. notes, explanations, etc.
e.g. {axis_examples}
Here are the 100 animals:
{batch_animals}
"""

# %%
import nest_asyncio
nest_asyncio.apply()
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
async def get_batch_scores(batch_animals):
    try:
        stream = await client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt_template.format(axis_string=axis_string, axis_examples=axis_examples, batch_animals=str(batch_animals))
                }
            ],
            model="gpt-4",
            stream=True,
        )
        
        response_content = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
                
        return response_content
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

async def process_all_batches():
    batch_size = 100
    num_batches = 50
    
    # Create list of batch tasks
    tasks = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = sorted_words[start_idx:end_idx]
        
        print(f"Creating task for batch {i+1}/{num_batches}")
        tasks.append(get_batch_scores(batch))
        
        # Still add a small delay between creating tasks to avoid rate limits
        await asyncio.sleep(0.1)
    
    # Run all tasks concurrently and gather results
    print("Running all batches concurrently...")
    all_responses = await asyncio.gather(*tasks)
    
    # Filter out None responses
    return [r for r in all_responses if r is not None]

# Run the async code
responses = asyncio.run(process_all_batches())

# %%
for i, r in enumerate(responses):
    try:
        r_json = json.loads(r)
        print(f"Batch {i+1}")
        print(len(r_json))
    except:
        print(f"Error parsing batch {i+1}")

# %%
# Join all responses and parse into one JSON array
file_name = "animals_size.json"
import json
all_scores = {}
for response in responses:
    try:
        scores = json.loads(response)
        all_scores.update(scores)
    except:
        print(f"Error parsing response")
        continue

# Write combined results to file
with open(file_name, 'w') as f:
    json.dump(all_scores, f, indent=2)

print(f"Wrote {len(all_scores)} scores to {file_name}")

# %%
# Read both JSON files
with open('animals_size.json', 'r') as f:
    size_scores = json.load(f)

with open('animals_aquatic.json', 'r') as f:
    aquatic_scores = json.load(f)

print(f"Read {len(size_scores)} size scores and {len(aquatic_scores)} aquatic scores")

# Create coordinates for each animal that appears in both files
coordinates = {}
for animal in set(size_scores.keys()) & set(aquatic_scores.keys()):
    size_score = size_scores[animal]
    aquatic_score = aquatic_scores[animal]
    coordinates[animal] = (-aquatic_score/100.0, -size_score/100.0)

print(f"Created coordinates for {len(coordinates)} animals")

# write coordinates to animals_coordinates.json
with open('animals_coordinates.json', 'w') as f:
    json.dump(coordinates, f, indent=2)

# %%
