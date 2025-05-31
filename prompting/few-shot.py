import google.generativeai as genai
import os

#few-shot prompting is a technique where a model is given a few examples of a task to perform before being asked to generate a response for a new, similar task.

#when few-shot prompting is used ?
#       it is particularly useful when the task is complex or requires specific formatting, as it helps guide the model's response based on the provided examples.

# Configure the API key (replace with your actual key or load from environment)
api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key) # It's best to load from environment variables

model = genai.GenerativeModel('gemini-1.5-flash') # gemini-1.5-flash is good for multi-turn conversations

# Define the few-shot prompt with examples
prompt_parts = [
    "Identify the primary emotion in the following sentences and output it as 'Emotion: [emotion]'.",
    "---",
    "Example 1:",
    "Sentence: I am so excited about the trip!",
    "Emotion: Joy",
    "---",
    "Example 2:",
    "Sentence: This news makes me feel incredibly sad.",
    "Emotion: Sadness",
    "---",
    "Example 3:",
    "Sentence: I can't believe they did that, it makes me furious.",
    "Emotion: Anger",
    "---",
    "Now, identify the emotion for the following sentence:",
    "Sentence: I feel really calm and peaceful right now."
]

response = model.generate_content(prompt_parts)
print(response.text)