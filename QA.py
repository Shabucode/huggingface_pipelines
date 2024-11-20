from transformers import pipeline

# Initialize the question-answering pipeline
qa_pipeline = pipeline(
    'question-answering',
    model='distilbert-base-cased-distilled-squad',
    tokenizer='distilbert-base-cased-distilled-squad'
)

# Define context and question
context = """
The Transformers library by Hugging Face provides thousands of pre-trained models for tasks like text classification, question answering, summarization, and translation.
These models are easy to use and integrate into applications.
"""
question = "What does the Transformers library provide?"

# Get the answer from the pipeline
result = qa_pipeline(question=question, context=context)

# Print the result
print(result)
