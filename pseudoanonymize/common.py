from joblib import Memory

cachedir = "cache"
memory = Memory(cachedir, verbose=0)


def make_chat_completion(
    client,
    model: str,
    system_prompt: str,
    user_msg: str,
    temperature: float = 0.1,
    context_length: int = 4096,
    json_format: bool = False,
):
    @memory.cache
    def call_client(
        model: str,
        system_prompt: str,
        user_msg: str,
        temperature: float = 0.1,
        context_length: int = 4096,
        json_format: bool = False,
    ):
        openai_kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            temperature=temperature,
            max_tokens=context_length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        if json_format:
            openai_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**openai_kwargs)
        return response.choices[0].message.content

    # Call the cached function. The internal function call_client was needed because we cannot cache client because it's an object.
    # The caching is done based on the parameter level.
    return call_client(model, system_prompt, user_msg, temperature, context_length, json_format)
