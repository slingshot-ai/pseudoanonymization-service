import re

from pseudoanonymize.base import BaseProcessor


class RegexAnon(BaseProcessor):
    def __init__(self):
        # Initialize regex patterns for anonymization
        self.patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b",
            "ipv4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\b",
            "ipv6": r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b",
        }

    def predict(self, input_: dict) -> dict:
        text = input_["text"]
        anon_text_safe = text
        replacement_dictionary_all_potential = {}

        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                placeholder = f"<{key}>"
                anon_text_safe = anon_text_safe.replace(match, placeholder)
                replacement_dictionary_all_potential[match] = placeholder

        return {"anon_text_safe": anon_text_safe, "replacement_dict": replacement_dictionary_all_potential}
