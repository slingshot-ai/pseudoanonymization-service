import dspy
from openai import OpenAI

from pseudoanonymize.base import BaseProcessor


class AnonSig(dspy.Signature):
    """
    Identify Personally Identifiable Information (PIIs) in the text. PIIs include:
        - Names (e.g., first name, last name)
        - Geographical locations smaller than a state (e.g., street address, city, ZIP code)
        - Dates (e.g., birth date, admission date)
        - Personal web URLs
        - Ages
        - Personal phone numbers and fax numbers
        - Personal email addresses
        - Social Security numbers
        - Medical record numbers and health plan beneficiary numbers
        - Account numbers (e.g., credit card)
        - License numbers
        - Vehicle identifiers (e.g., license plate numbers)
        - Device identifiers (e.g., serial numbers)
        - Personal IP address numbers

    Or anything else uniquely identifying, such as:
        - Rare disease information
        - Specific topic names at a conference
        - Unique research grants
        - Niche specialties (e.g., a rare research project)
        - Unique facial features (e.g., scars)
        - Obscure travel destinations
        - Unusual transportation methods
        - Unique achievements (e.g., state chess champion)
        - Unusual hobbies (e.g., competitive eating)
    """

    unanonymized_text = dspy.InputField()
    PII_candidates = dspy.OutputField(desc="All potential PIIs identified in the text")


class ReplacementDictSig(dspy.Signature):
    """
    Generate a dictionary to replace identified PIIs in the text. Given the PII candidates and the text,
    create a dictionary mapping each PII to a placeholder. Ensure relevant keys are included even if there are typos.
    """

    unanonymized_text = dspy.InputField()
    PII_candidates = dspy.InputField()
    replacement_dictionary_all_potential = dspy.OutputField(
        desc="A dictionary mapping all potential PIIs to placeholders enclosed in brackets.",
        prefix="Valid Python Dict (will run eval):",
    )
    replacement_dictionary_after_examination = dspy.OutputField(
        desc="A dictionary mapping examined and validated PIIs to placeholders enclosed in brackets",
        prefix="Valid Python Dict (will run eval):",
    )


class CoTAnon(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(AnonSig)
        self.replacement_dict_pred = dspy.ChainOfThought(ReplacementDictSig)

    def forward(self, unanonymized_text):
        cot_pred = self.cot(unanonymized_text=unanonymized_text)
        replacement_dict_all_potential = self.replacement_dict_pred(
            unanonymized_text=unanonymized_text, PII_candidates=cot_pred.PII_candidates
        ).replacement_dictionary_all_potential

        replacement_dict_after_examination = self.replacement_dict_pred(
            unanonymized_text=unanonymized_text, PII_candidates=cot_pred.PII_candidates
        ).replacement_dictionary_after_examination

        replacement_dict_all_potential = self._parse_dict(replacement_dict_all_potential)
        replacement_dict_after_examination = self._parse_dict(replacement_dict_after_examination)
        self._remove_ash_from_replacement_dict(replacement_dict_all_potential)
        self._remove_ash_from_replacement_dict(replacement_dict_after_examination)
        anon_text_safe = self._parse_replacements(unanonymized_text, replacement_dict_all_potential)
        anon_text_smart = self._parse_replacements(unanonymized_text, replacement_dict_after_examination)
        return dspy.Prediction(
            **{
                "PII_candidates": cot_pred.PII_candidates,
                "replacement_dictionary_all_potential": replacement_dict_all_potential,
                "replacement_dictionary_after_examination": replacement_dict_after_examination,
                "anon_text_safe": anon_text_safe,
                "anon_text_smart": anon_text_smart,
            }
        )

    def _remove_ash_from_replacement_dict(self, replacement_dict_all_potential: dict):
        keys = list(replacement_dict_all_potential.keys())
        for key in keys:
            if "ash" in key.lower():
                del replacement_dict_all_potential[key]

    def _parse_dict(self, str_dict):
        parse_att = str_dict
        parse_att = parse_att[: parse_att.find("}") + 1]
        parse_att = parse_att[parse_att.find("{") :]
        parse_att = eval(parse_att)  # TODO: double check how DSPy handles exceptions.
        return parse_att

    def _parse_replacements(self, text, replacement_dict):
        cleaned_dict = dict()
        for key, replacement in replacement_dict.items():
            if isinstance(key, str):
                cleaned_dict[key] = replacement
            else:
                for key_instance in key:
                    cleaned_dict[key_instance] = replacement

        sorted_keys = sorted(cleaned_dict.keys(), key=len, reverse=True)
        for key in sorted_keys:
            text = text.replace(key, str(cleaned_dict[key]))
        return text


# %%
class DspyAnon(BaseProcessor):
    def __init__(self, client: OpenAI = None):
        self.model = CoTAnon()
        self.model.load("models/dspy_optimized_cot.json")

    def predict(self, input_: dspy.Dict[str, dspy.Any]) -> dspy.Dict[str, dspy.Any]:
        text = input_["text"]
        prediction = self.model(text)
        return prediction.anon_text_safe, prediction.replacement_dictionary_all_potential
