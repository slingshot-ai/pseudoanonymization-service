import os

import dspy
from dotenv import load_dotenv
from openai import OpenAI

from pseudoanonymize.anonymization import Anonymizer
from pseudoanonymize.deanonymization import Deanonymizer
from pseudoanonymize.direct_json_anonymization import JSON_FEW_SHOT_PROMPT, JSON_SYSTEM_PROMPT, JsonDirectAnonymizer
from pseudoanonymize.dspy_anonmization import DspyAnon
from pseudoanonymize.pipeline import PiplelineAnon
from pseudoanonymize.regex_anonymization import RegexAnon

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
deanonymizer = Deanonymizer(client=client)


def get_pipeline(option: str) -> PiplelineAnon:
    print("v2")
    turbo = dspy.OpenAI(model='gpt-4o', max_tokens=4096, api_key=openai_api_key)
    dspy.settings.configure(lm=turbo)

    anonymizer = Anonymizer(client=client)

    model_gpt4o = "gpt-4o"
    prompt_gpt4o = JSON_SYSTEM_PROMPT + JSON_FEW_SHOT_PROMPT
    direct_anonymizer_gpt4o = JsonDirectAnonymizer(client=client, model=model_gpt4o, system_prompt=prompt_gpt4o)

    model_gpt35ft = "ft:gpt-3.5-turbo-0125:slingshot-daniel:pii-dist-processed:9jcKs3kP"
    prompt_gpt35ft = JSON_SYSTEM_PROMPT
    direct_anonymizer_gpt35ft = JsonDirectAnonymizer(client=client, model=model_gpt35ft, system_prompt=prompt_gpt35ft)

    dspy_anonymizer = DspyAnon()

    pipeline_options = {
        "DSPyCOT": PiplelineAnon([dspy_anonymizer, RegexAnon()]),
        "GPT-4o": PiplelineAnon([direct_anonymizer_gpt4o, RegexAnon()]),
        "GPT-3.5-FT": PiplelineAnon([direct_anonymizer_gpt35ft, RegexAnon()]),
    }

    return pipeline_options[option]
