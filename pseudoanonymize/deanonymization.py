from typing import Any, Dict, List

from faker import Faker
from openai import OpenAI

from pseudoanonymize.base import BaseProcessor
from pseudoanonymize.common import make_chat_completion

fake = Faker()

prompt = """
You will be given a redacted text file and a list of PII placeholders. For each placeholder, generate a corresponding fake value for deanonymization purposes, ensuring diversity and realism in names and addresses. Use the corresponding faker function for each placeholder from the list provided. If no faker function exists for a given PII description, generate a valid placeholder. Output the fake values as a dictionary in the format required for parsing.

Available faker functions:
from faker import Faker
fake = Faker()
fake.first_name()
fake.last_name()
fake.name()
fake.address()
fake.phone_number()
fake.email()
fake.company()
fake.job()
fake.date_of_birth()
fake.ssn()
fake.credit_card_number()

Example Input 1:
[NAME2]: Hey [NAME1], have you been watching any new anime lately?

[NAME1]: Yeah, actually! I just finished "Attack on Titan" last week. It's been really intense. What about you, [NAME2]?

[NAME2]: Oh, I love "Attack on Titan"! I'm currently watching "My Hero Academia". By the way, did I tell you about the time I met the creator of "Naruto" at the anime convention downtown?

[NAME1]: That sounds amazing! What was it like meeting [NAME5]?

[NAME2]: It was surreal! He was signing autographs at the convention center on [STREET1]. I got a signed poster and even a picture with him. I have it framed in my room next to my bed.
PII placeholders:
['[NAME1]', '[NAME2]', '[NAME5]', '[STREET1]']
Expected output:
{
    "[NAME1]": fake.first_name(),
    "[NAME2]": fake.first_name(),
    "[NAME5]": fake.first_name(),
    "[STREET1]": fake.address()
}

Example Input 2:
[NAME1]: No, I missed that one. But it sounds like blockchain could decentralize and democratize funding, which is really exciting. By the way, did you know that [NAME6], the [UNIQUE ACHIEVEMENT] is also a blockchain enthusiast?

[NAME2]: I had no idea! It's incredible how blockchain is drawing interest from so many different fields and industries.
PII placeholders:
['[NAME1]', '[NAME2]', '[NAME6]', '[UNIQUE ACHIEVEMENT]']
Expected output:
{
    "[NAME1]": fake.name(),
    "[NAME2]": fake.name(),
    "[NAME6]": fake.name(),
    "[UNIQUE ACHIEVEMENT]": "Nobel Prize Winner"
}
"""


class Deanonymizer(BaseProcessor):
    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.system_prompt = prompt
        self.context_length = 4096
        self.model = "gpt-4o"
        self.temperature = 0.1

    def _deanon_text_llm_output(self, text: str, replacement_keys: List[str]) -> str:
        user_msg = f"""input text:\n```\n{text}\n```\npii placeholders:\n```\n{replacement_keys}\n'''"""
        return make_chat_completion(
            self.client,
            self.model,
            system_prompt=self.system_prompt,
            user_msg=user_msg,
            context_length=self.context_length,
        )

    def predict(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        text = input_["text"]
        anon_replacement_dict = input_["replacement_dict"]
        anon_replacement_keys = list(set(anon_replacement_dict.values()))

        combined_output = self._deanon_text_llm_output(text, anon_replacement_keys)

        replacement_dict = self._extract_and_parse_replacement_dict(combined_output)
        deanonymized_text = self._parse_replacements(text, replacement_dict)
        return {"deanonymized_text": deanonymized_text}
