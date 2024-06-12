from typing import Any, Dict

from .base import BaseProcessor
from .common import make_chat_completion
from .exceptions import UnparsableLLMOutputException

with open('./prompts_examples/cot_example.txt', 'r') as file:
    cot_demo = file.read()

CoT_prompt = f"""
You are tasked with identifying and replacing Personally Identifiable Information (PII) in a conversation. PII includes names, addresses, dates, and any other specific information that can be used to identify an individual. Follow these steps:

Steps:

1. Identify PII Candidates:
   - Identify any names, streets, cities, dates, or other specific details that can be used to identify an individual. This includes:
        - Names (e.g., first name, last name).
        - Geographical locations smaller than a state (e.g., street address, city, ZIP code).
        - PII Dates (e.g., birth date, admission date).
        - Personal web Universal Resource Locators (URLs).
        - Ages over 89.
        - Personal phone numbers and fax numbers.
        - Personal email addresses.
        - Social Security numbers.
        - Medical record numbers and health plan beneficiary numbers.
        - Account numbers (e.g., credit card).
        - License numbers.
        - Vehicle identifiers (e.g., license plate numbers).
        - Device identifiers (e.g., serial numbers).
        - Personal Internet Protocol (IP) address numbers.
   - Err on the side of caution by over-identifying rather than under-identifying PII.
2. Determine Reasons for PII:
   - For each identified PII candidate, provide a brief reason why it is considered PII or not PII.

3. Generate Replacement Dictionary:
   - Create a dictionary where each key is a tuple containing all identified references for a single piece of PII.
   - Group multiple references of the same PII into a single tuple.
   - Make sure not to include overly detailed or sentence-specific references. Only use concise references like names, street names, city names, etc.
   - The dictionary values should be placeholders indicating the type of PII (e.g., "[NAME1]", "[STREET1]", "[CITY1]", "[DATE1]").

Your output should strictly follow this format:
1. PII Candidates and Reasons:
- <PII Candidate 1>
  - References: "<Reference 1>", "<Reference 2>", ...
  - Reason: <Reason>

- <PII Candidate 2>
  - References: "<Reference 1>", "<Reference 2>", ...
  - Reason: <Reason>

...

2. Replacement Dictionary:
{{
    ("<Reference 1>", "<Reference 2>"): "[Placeholder]",
    ("<Reference 3>",): "[Placeholder]",
    ...
}}

Keep the references concise and avoid overly detailed or context-specific entries.

Example:
{cot_demo}
"""


class Anonymizer(BaseProcessor):
    def __init__(self, client: Any):
        super().__init__(client)
        self.system_prompt = CoT_prompt
        self.context_length = 4096
        self.model = "gpt-4o"
        self.temperature = 0.1

    def _identify_and_anonymize_pii(self, text: str) -> str:
        return make_chat_completion(
            self.client, self.model, system_prompt=self.system_prompt, user_msg=text, context_length=self.context_length
        )

    def predict(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        text = input_["text"]
        combined_output = self._identify_and_anonymize_pii(text)

        parts = combined_output.split('2. Replacement Dictionary:')
        if len(parts) < 2:
            error_message = "LLM Produced unparsable output."
            self._log_error(error_message, combined_output)
            raise UnparsableLLMOutputException(error_message)

        replacement_dict = self._extract_and_parse_replacement_dict(parts[1])
        anonymized_text = self._parse_replacements(text, replacement_dict)
        return {"anonymized_text": anonymized_text, "replacement_dict": replacement_dict}
