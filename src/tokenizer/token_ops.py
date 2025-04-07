def tokenize(element, tokenizer, block_size, padding="longest", truncate=True):
    return tokenizer(
        [tokenizer.bos_token + text.strip() + tokenizer.eos_token for text in element["text"]],
        truncation=truncate,
        max_length=block_size,
        padding=padding,
    )

def filter_long_sequences(example, tokenizer, block_size):
    tokenized = tokenizer(
        tokenizer.bos_token + example["text"].strip() + tokenizer.eos_token,
        truncation=False,
        max_length=None
    )
    return len(tokenized["input_ids"]) <= block_size
