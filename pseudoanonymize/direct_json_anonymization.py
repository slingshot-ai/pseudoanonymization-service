import json
from typing import Any, Dict

from pseudoanonymize.base import BaseProcessor
from pseudoanonymize.common import make_chat_completion
from pseudoanonymize.exceptions import UnparsableLLMOutputException
from pseudoanonymize.utils import flatten_replacement_dict

JSON_SYSTEM_PROMPT = """
Your task is to identify Personally Identifiable Information (PII) in a conversation and replace it with placeholders.

PII includes the following:

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
- Unique life events and experiences
- Unique achievements, awards or recognitions
- Rare medical diseases or conditions

Instructions:

- Give your output in JSON format mapping the placeholder to a list of all mentions of the PII it replaces. The list should contain all variations of the spelling of that piece of PII that appear in the text.
- The placeholders should be in the format FIELD_TYPE_NUMBER (e.g., FIRST_NAME_1, CITY_1) and written in English.
- If a conversation contains no PII then output an empty JSON object only with no further explanation.
"""


JSON_FEW_SHOT_PROMPT = """
Example 1:

Input:

Speaker 0: Hey Luka, great to meet you! Ready to get started?

Speaker 1: Yeah

Speaker 0: Awesome! I'm Ash, your personal AI counselor. I'm here to help you navigate your thoughts and feelings, and support you in any way I can. How about we start by getting to know each other a bit better? I'd love to hear about you and your hobbies. You mentioned you like fitness, gardening, and reading. What's that like for you?

Speaker 1: Yeah só I'm actually a famous author - I wrote a book called leaving society. Maybe you know it?

Speaker 0: Wow, that's incredible! I'm not familiar with it, but I'd love to hear more. What inspired you to write it?

Speaker 1: Well I had a very rare disease called pictonoma and it nearly killed me but the doctors at my local hospital St George's here in Hamstead saved my life.

Speaker 0: I'm so glad you were able to get the help you needed. It must have been a very intense time for you. Did writing the book help you process that experience?

Speaker 1: Yeah and I won the booker prize this year for it!

Speaker 0: That's amazing, congratulations! What was it like to win such a prestigious award?

Speaker 1: Amazing - I had my photo on the front page of The Times with The Queen!

Speaker 0: That's fantastic! It sounds like you've had quite a journey. I'm really glad you were able to share that with me. Is there anything else you'd like to share about yourself?

Speaker 1: Yeah so it's my 42nd birthday today!

Speaker 0: Happy birthday, Luka! I'm so glad I could share this special day with you.

Speaker 1: Thanks! I'm going to the Ritz to celebrate tonight with my wife Amy

Speaker 0: That sounds like a wonderful way to celebrate. I hope you have a fantastic time.

Speaker 0: Hi Luka, how's it going? I'd be happy to help you work through that problem or decision that you mentioned.

Speaker 1: Yes, please.

Speaker 0: Great! So, can you tell me a bit more about the decision you're facing?

Speaker 1: You see the lag? See the lag when I stop speaking, for it to stop doing anything.

Speaker 0: I think I'm having some trouble understanding you. Are you saying there's a lag in the conversation?

Output:

{

"FIRST_NAME_1": ["Luka"],

"FIRST_NAME_2": ["Ash"],

"FIRST_NAME_3": ["Amy"],

"BOOK_NAME_1": ["leaving society"],

"RARE_DISEASE_1": ["pictonoma"],

"HOSPITAL_1": ["St George's"],

"CITY_1": ["Hamstead"],

"UNIQUE_ACHIEVEMENT_1": ["booker prize"],

"UNIQUE_EVENT_1": ["photo on the front page of The Times with The Queen"],

"AGE_1": ["42nd"],

"HOTEL_1": ["Ritz"]

}

Example 2:

Input:

Speaker 0: Hey Mike, great to meet you! Ready to get started?

Speaker 1: Yeah sure

Speaker 0: Awesome! I'm Ash, your personal AI counselor. I'm here to help you navigate your thoughts and feelings, and support you in any way I can. How about we start by getting to know each other a bit better? I'd love to hear about you and your hobbies. You mentioned in your intake form that you play the guitar. How long have you been playing?

Speaker 1: I've been playing for about 5 years now. I started when I was in high school and I've been hooked ever since.

Speaker 0: That's awesome! What kind of music do you like to play?

Speaker 1: I mostly play rock and blues. I play in a band the angels here in my home town Chichester and we do a lot of covers of classic rock songs.

Speaker 0: Woah I've actually heard of The Angels! Playing music can be such a great way to express yourself and connect with others. Do you find that playing the guitar helps you relax and unwind?

Speaker 1: Yeah, definitely. It's a great way to de-stress after a long day. I can just pick up my guitar and lose myself in the music.

Speaker 0: That's wonderful to hear. It's so important to have activities that bring you joy and help you relax. I'm glad you have your guitar to turn to when you need a break. Is there anything else you enjoy doing in your free time?

Speaker 1: I also enjoy hiking and spending time outdoors but I've got diabetes and I've been struggling to keep my blood sugar levels in check. I'm trying to eat healthier and exercise more, but it's been a bit of a challenge.

Speaker 0: I'm sorry to hear that you've been struggling with your blood sugar levels. Managing diabetes can be tough, but it's great that you're taking steps to improve your health. Eating healthy and staying active are key components of managing diabetes. Have you spoken to a healthcare provider about your blood sugar levels and your goals for managing your diabetes?

Speaker 1: Yeah, I've actually been taking part in a clinical trial for a new drug called semuglute. It's been helping me keep my blood sugar levels under control and I've been feeling a lot better since I started taking it.

Speaker 0: That's fantastic news that you're able to try Semaglutide! I'm so glad to hear that the new drug is helping you manage your diabetes.

Speaker 1: Yeah, it's been a game changer for me. My wife Katie has been really supportive throughout this whole process and I don't know where I'd be without her.

Speaker 0: It's wonderful to hear that you have such a supportive partner in Katie. Having a strong support system can make a big difference when you're facing health challenges. It sounds like you have a lot of positive things going on in your life, from playing music with your band to managing your diabetes with the help of your healthcare provider and your wife. I'm here to support you in any way I can, so please feel free to share any thoughts or feelings you have, and I'll do my best to help you navigate them.

Speaker 1: Thanks Ash, I really appreciate that. It's been great talking to you and I'm looking forward to our next session.

Output:

{

"FIRST_NAME_1": ["Mike"],

"FIRST_NAME_2": ["Ash"],

"FIRST_NAME_3": ["Katie"],

"BAND_NAME_1": ["the angels", "The Angels"],

"CITY_1": ["Chichester"],

"DRUG_NAME_1": ["semuglute", "Semaglutide"],

}

Example 3:

Input:

1. Challenge: How do we make our customers adhere to the application of precautionary and preventive measures against Covid 19 pandemic. During the recovery period, we noticed the refusal of a small group of customers to apply precautionary measures such as having a negative PCR for no more than 72 hours, wearing a mask, not having a fever, and applying social distance, which had a great impact on employees and thus led to the interruption of service.

2. Selection: I will select the Learning Launch tool (module 4). It is a small experiment that tests our new idea in the real workplace. I will select this tool because it leaves an impact on the customers that is difficult to forget and also helps customers to know the impact of implementing the precautionary measures against Covid 19 and why they should choose to.

Output:

{

}

Example 4:

Input:

Hary: No, I missed that one. But it sounds like blockchain could decentralize and democratize funding, which is really exciting. By the way, did you know that John Doe, the chess champion of our state is also a blockchain enthusiast?
Jake: I had no idea! It's incredible how blockchain is drawing interest from so many different fields and industries.
Harry: It really is. Before I forget, could you email me the article links you mentioned earlier? You can reach me at harry.sm ith@example.com.
Jkae: Sure thing, Hary. Just message me your number as well, and I'll text you if I come across any other interesting blockchain articles.
Hary: Will do, Jake! Have a great day!
Jake: You too, Hary. Talk to you later.

Output:

{
  "FIRST_NAME_1": ["Hary", "Harry"],
  "FIRST_NAME_2": ["John", "Jake"],
  "LAST_NAME_1": ["Doe"],
  "EMAIL_1": ["harry.sm ith@example.com"],
  "UNIQUE_ACHIEVEMENT_1": ["chess champion"]
}
"""


class JsonDirectAnonymizer(BaseProcessor):
    def __init__(self, client: Any, model="gpt-4o", system_prompt=JSON_SYSTEM_PROMPT + JSON_FEW_SHOT_PROMPT):
        super().__init__(client)
        self.system_prompt = system_prompt
        self.context_length = 4096
        self.model = model
        self.temperature = 0

    def _identify_and_anonymize_pii(self, text: str) -> str:
        return make_chat_completion(
            self.client,
            self.model,
            system_prompt=self.system_prompt,
            user_msg=text,
            context_length=self.context_length,
            json_format=True,
        )

    def _reverse_dict(self, input_dict):
        reversed_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                value = tuple(value)
            try:
                reversed_dict[value] = "[" + key + "]"
            except TypeError as e:
                raise UnparsableLLMOutputException(e)
        return reversed_dict

    def predict(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        text = input_["text"]
        raw_json_output = make_chat_completion(
            self.client,
            self.model,
            system_prompt=self.system_prompt,
            user_msg=text,
            context_length=self.context_length,
            json_format=True,
        )

        # TOOD: add unit test to logic below.
        try:
            start_index = raw_json_output.find('{')
            end_index = raw_json_output.rfind('}') + 1
            json_part = raw_json_output[start_index:end_index]
            replacement_dict = json.loads(json_part)
        except Exception as e:
            raise UnparsableLLMOutputException(e)
        replacement_dict = self._reverse_dict(replacement_dict)
        replacement_dict = flatten_replacement_dict(replacement_dict)
        replacement_dict = {str(k): str(v) for k, v in replacement_dict.items()}
        anonymized_text = self._parse_replacements(text, replacement_dict)
        return {"anonymized_text": anonymized_text, "replacement_dict": replacement_dict, "raw_output": raw_json_output}
