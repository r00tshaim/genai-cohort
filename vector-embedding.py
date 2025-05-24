from dotenv import load_dotenv
import os
import google.generativeai as genai

#Vector embedding is phaseII of LLM pipeline
#It converts tokens generated in phaseI to vector embeddings
    #but this code is not using tokens, it is directly using text as embedding API keys are designed 
    # to accpt text input for user in LLM pipeline they follow proper sequence of phases


class VectorEmbedding:
    def __init__(self):
        self.api_key = os.getenv("GENAI_API_KEY")
        genai.configure(api_key=self.api_key)

    def embed_content(self, content, model="embedding-001", task_type="RETRIEVAL_DOCUMENT"):
        return genai.embed_content(model=model, content=content, task_type=task_type)

    def print_embedding(self, embedding):
        print(f"Embedding vector length: {len(embedding)}")
        print(f"First 5 elements: {embedding[:5]}")
    
#Example usage:
if __name__ == "__main__":
    load_dotenv()
    vector_embedding = VectorEmbedding()

    text = "The cat sat on the mat."
    response = vector_embedding.embed_content(text)

    embedding = response["embedding"]
    
    vector_embedding.print_embedding(embedding)
    
    # Output:
    #Embedding vector length: 768
    #First 5 elements: [0.017315384, -0.025808778, -0.05304493, -0.030395865, 0.038529325]