import hashlib

def get_hash(sentence):
    """Возвращает хэш предложения"""
    bytes_sentence = sentence.encode('utf-8')
    hash_value = hashlib.md5(bytes_sentence).hexdigest()

    number = int(hash_value, 16)
    return number

print(get_hash('hello'))
