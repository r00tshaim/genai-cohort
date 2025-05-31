import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
import re

# --- API Key Setup ---
try:
    API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the GOOGLE_API_KEY environment variable (e.g., export GOOGLE_API_KEY='your_key_here').")
    exit()

# --- Initialize the Gemini Client ---
client = genai.Client(api_key=API_KEY)

# --- Chain-of-Thought Process Prompt ---
# This defines the step-by-step reasoning process.
COT_PROCESS_PROMPT = """
For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{ "step": "string", "content": "string" }

Example:
Input: What is 2 + 2.
Output: { "step": "analyse", "content": "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }
Output: { "step": "think", "content"": "To perform the addition i must go from left to right and add all the operands" }
Output: { "step": "output", "content"": "4" }
Output: { "step": "validate", "content"": "seems like 4 is correct ans for 2 + 2" }
Output: { "step": "result", "content"": "2 + 2 = 4 and that is calculated by adding all numbers" }
"""

# --- Initialize Chat History ---
# We use the COT_PROCESS_PROMPT as the first user message,
# and the model's acknowledgment as the first model message.
messages = [
    {"role": "user", "content": COT_PROCESS_PROMPT}
]

print("Type 'exit' to quit.")


while True:
    user_query = input("\nEnter your query: ")
    if user_query.lower() == 'exit':
        break

    # Add the current user query to messages
    messages.append({"role": "user", "content": user_query})

    # Loop to get Chain-of-Thought steps
    while True:
        try:
            # Call the model using the client.
            # Use 'system_instruction' for the persona (Neko the cat).
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=COT_PROCESS_PROMPT,
                    response_mime_type="application/json",
                ),
                contents=user_query
            )

            # Parse the JSON response
            parsed_response = json.loads(response.text)
            
            # Add the model's response to messages for context
            messages.append({"role": "model", "content": json.dumps(parsed_response)})

            step = parsed_response.get("step")
            content = parsed_response.get("content")

            if step == "output":
                print(f"🤖 Meow! Output: {content}")
                break # Break the inner loop, move to next user query
            elif step == "result":
                print(f"✅ Meow! Result: {content}")
                break
            elif step == "error":
                print(f"❌ Meow! Error: {content}")
                break
            else:
                print(f"🧠 Meow! Thinking: {content}")
                # The chat history itself guides the model to the next step.
                
        except json.JSONDecodeError:
            print(f"⚠️ Meow! Warning: Model did not return valid JSON. Raw response: {response.text}")
            # If JSON is invalid, try to tell the model to fix it and retry this step.
            messages.append({"role": "user", "parts": [types.Part(text="Meow! The previous response was not valid JSON. Please re-generate adhering strictly to the Output Format.")]})
            continue # Try this step again
        except Exception as e:
            print(f"An API error occurred: {e}. Meow.")
            break # Break out if there's an unrecoverable API error

print("\nThank you for using the Chain-of-Thought Assistant! Purrrr.")