import os

from dotenv import load_dotenv
from joblib import Memory
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

cachedir = "cache"
memory = Memory(cachedir, verbose=0)


def make_chat_completion(client, model, system_prompt, user_msg, temperature=0.1, context_length=4096):
    @memory.cache
    def call_client(
        model, system_prompt, user_msg, temperature=0.1, context_length=4096
    ):  # TODO: should an argument to indicate who is calling to avoid weird situations where two different models recieve the same input file.
        print("test")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            temperature=temperature,
            max_tokens=context_length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content

    return call_client(model, system_prompt, user_msg, temperature, context_length)
