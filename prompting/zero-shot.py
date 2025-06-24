from dotenv import load_dotenv
import os
import google.generativeai as genai

# Zero-shot prompting is a technique where a model is asked to perform a task without any prior examples or training specific to that task.

#when zero-shot prompting is used ?
#       the model relies on its pre-existing knowledge and understanding of language to generate a response based on the prompt provided.

class ZeroShotPrompting:
    def __init__(self):
        self.api_key = os.getenv("GENAI_API_KEY")
        genai.configure(api_key=self.api_key)
    
    def generate_response(self, prompt, model="gemini-2.0-flash"):
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text

#Example usage:
if __name__ == "__main__":
    load_dotenv()
    zero_shot_prompting = ZeroShotPrompting()

    prompt = input('> ')
    response = zero_shot_prompting.generate_response(prompt)

    print(f"Response: {response}")
    
    # Output:
    # Response: The capital of France is Paris.


def generatePassword(length=12):
    import random
    import string

    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for i in range(length))
    return password