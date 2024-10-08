from pseudoanonymize.base import BaseProcessor


class PiplelineAnon(BaseProcessor):
    """
    A pipeline that takes a list of anonymizers and chains them together.

    Example:
    ```python
    from pseudoanonymize import PiplelineAnon
    from pseudoanonymize.dspy_anonmization import DspyAnon
    from pseudoanonymize.regex_anonymization import RegexAnon
    from pseudoanonymize.pipeline import PiplelineAnon
    model = PiplelineAnon([DspyAnon(), RegexAnon()])
    output = model.predict(inp)
    """

    def __init__(self, processors: list[BaseProcessor]):
        self.processors = processors

    def predict(self, input_, keep_ash=True):
        output_dicts = []
        for processor in self.processors:
            output = processor.predict(input_)
            replacement_dict = output["replacement_dict"]
            output_dicts.append(replacement_dict)

        final_replace_dict = {}
        for replace_dict in output_dicts:
            final_replace_dict.update(replace_dict)

        # Don't anonymize keys that contain "ash" if keep_ash is True

        print(final_replace_dict.keys())
        if keep_ash:
            keys = list(final_replace_dict.keys())
            for key in keys:
                if "ash" in key.lower():
                    del final_replace_dict[key]

        anonymized_text = self._parse_replacements(input_["text"], final_replace_dict)
        # return anonymized_text, output_dicts
        # return anonymized_text, final_replace_dict
        return {"anonymized_text": anonymized_text, "replacement_dict": final_replace_dict}
