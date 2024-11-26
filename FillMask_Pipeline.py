from transformers import pipeline

# Initialize the fill-mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Example input with a [MASK] token
sentence = "The Transformers library is [MASK] for natural language processing."

# Get predictions for the masked token
results = fill_mask(sentence)

# Print the results
for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']:.4f}")
