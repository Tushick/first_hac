from nltk import MWETokenizer

tokenizer = MWETokenizer()

def tokenize_text(text: str, type_value: type=float) -> list:
    data = tokenizer.tokenize(text)
    int_data = []
    for i in data:
        int_data.append(type_value(ord(i)))
    
    return int_data
