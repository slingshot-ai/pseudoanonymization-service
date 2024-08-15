import re

import tiktoken


def count_openai_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Calculate the number of tokens a model will use for a given text.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


def flatten_replacement_dict(replacement_dict):
    flat_dict = {}
    for key, value in replacement_dict.items():
        if isinstance(key, tuple):
            for sub_key in key:
                flat_dict[sub_key] = value
        else:
            flat_dict[key] = value
    return flat_dict


def get_chunks(
    text: str, max_tokens: int, splitter, connector: str, token_count_based_on: str = "gpt-3.5-turbo"
) -> list[str]:
    """
    Splits the input text into chunks where each chunk does not exceed the specified maximum token limit,
    using the provided splitter function to determine how to split the text.

    args:
        text: the input text to split
        max_tokens: the maximum number of tokens allowed in each chunk
        splitter: a function that takes a string and returns a list of segments
        connector: a string to add between segments when joining them
        token_count_based_on: the model to use for counting tokens
    """
    chunks = []
    current_chunk = ""
    for segment in splitter(text.strip()):
        token_count = count_openai_tokens(current_chunk + connector + segment, model_name=token_count_based_on)
        if token_count > max_tokens:
            chunks.append(current_chunk.strip())

            if count_openai_tokens(segment, model_name=token_count_based_on) > max_tokens:
                # if the segment itself is too long, split it into smaller chunks
                # TODO: handle rare case where a even after splitting by words the segment contains more tokens than max_tokens
                # this can only happen if a single word is longer than max_tokens
                chunks.extend(chunk_by_word(segment, max_tokens))
            else:
                # start a new chunk
                current_chunk = segment
        else:
            # enough space to add the segment to the current chunk
            current_chunk += (connector if current_chunk else "") + segment

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_by_word(text: str, max_tokens: int) -> list[str]:
    def split_by_word(text: str) -> list[str]:
        return text.split()

    return get_chunks(text, max_tokens, split_by_word, connector=' ')


def chunk_by_terminal_punctuation(text: str, max_tokens: int) -> list[str]:
    def split_by_terminal_punctuation(text: str) -> list[str]:
        return re.split(r'(?<=[.!?])\s+', text)

    return get_chunks(text, max_tokens, split_by_terminal_punctuation, connector='.')


def chunk_by_line(text: str, max_tokens: int) -> list[str]:
    def split_by_line(text: str) -> list[str]:
        return text.split('\n')

    return get_chunks(text, max_tokens, split_by_line, connector='\n')
