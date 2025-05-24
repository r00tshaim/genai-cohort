import tiktoken

#Tokenization is phaseI of the LLM pipeline.
#It converts text to tokens and vice versa.

class Tokenizer:
    def __init__(self, model_name='gpt-4o'):
        self.encoder = tiktoken.encoding_for_model(model_name)

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    def vocab_size(self):
        return self.encoder.n_vocab
    

#Example usage:
if __name__ == "__main__":
    model = 'gpt-4o'  # Specify the model name
    tokenizer = Tokenizer(model)

    tokenizer_size = tokenizer.vocab_size()
    print(f"For model: {model}\nTokenizer size: {tokenizer_size}\n\n")  # 2,00,019 (200K)

    text = "The cat sat on the mat."
    tokens = tokenizer.encode(text)
    print(f"Input Text: {text} \nEncoded tokens: {tokens}\n\n")  # Tokens [976, 9059, 10139, 402, 290, 2450]

    my_tokens = [976, 9059, 10139, 402, 290, 2450]
    decoded_text = tokenizer.decode(my_tokens)
    print(f"Input Tokens: {my_tokens} \nDecoded text: {decoded_text}\n\n")  # "The cat sat on the mat."