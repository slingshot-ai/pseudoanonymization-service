from typing import Any, Dict

from pseudoanonymize.base import BaseProcessor
from pseudoanonymize.common import make_chat_completion
from pseudoanonymize.exceptions import UnparsableLLMOutputException

chain_of_thought_prompt = f"""
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
```
Input Example:
'''
Alex: Hey Samantha, how's your beer brewing going lately?

Samantha: It's going great, Alex! I just brewed a new batch last weekend at my home on Maple Street. How about you?

Alex: Nice! I haven't had much time since moving to Springfield, but I did manage to make an IPA last month.

samantha: That's awesome Alx. Do you use any particular recipe?

Alex: Yeah, I usually follow a recipe I got from a beer brewing forum. By the way, I recently found a new hops supplier near Elm Street. You might wanna check them out.
'''

Expected Output:
1. PII Candidates and Reasons:
- Samantha
  - References: "Hey Samantha", "going great, Alex!", "That's awesome Alx"
  - Reason: Names are directly associated with individuals and should be considered PII.
  - Decision: PII
- Alex
  - References: "Hey Samantha", "going great, Alex!", "That's awesome Alx", "Yeah, I"
  - Reason: Names are directly associated with individuals and should be considered PII.
  - Decision: PII
- Maple Street
  - Reference: "my home on Maple Street"
  - Reason: Specific street names can identify an individual's location and thus are considered PII.
  - Decision: PII
- Springfield
  - Reference: "since moving to Springfield"
  - Reason: Cities can identify an individual's broader location and thus are considered PII.
  - Decision: PII
- Elm Street
  - Reference: "new hops supplier near Elm Street"
  - Reason: Specific street names can identify an individual's location and thus are considered PII.
  - Decision: PII

2. Replacement Dictionary:
{{
    ("Samantha", "samantha"): "[NAME1]",
    ("Alex", "Alx"): "[NAME2]",
    ("Maple Street",): "[STREET1]",
    ("Springfield",): "[CITY1]",
    ("Elm Street",): "[STREET2]"
}}

Example 2:
Input Example:
'''
1. Challenge: How do we make our customers adhere to the application of precautionary and preventive measures against Covid 19 pandemic. During the recovery period, we noticed the refusal of a small group of customers to apply precautionary measures such as having a negative PCR for no more than 72 hours, wearing a mask, not having a fever, and applying social distance, which had a great impact on employees and thus led to the interruption of service.

2. Selection: I will select the Learning Launch tool (module 4). It is a small experiment that tests our new idea in the real workplace. I will select this tool because it leaves an impact on the customers that is difficult to forget and also helps customers to know the impact of implementing the precautionary measures against Covid 19 and why they should choose to.

3. Application: There will be two scenarios. The first scenario shows that X customer enters the customer service center ignoring the precautionary measures and by doing that it helped spread the virus among employees and the rest of the customers who applied the precautionary measures, which led to the injury of some, disruption of work and an increase in expenses as a result of the sterilization of the center.

4. Insight: By applying the selected tools we can convince the target group of the need to adhere to the application of precautionary measures because everyone’s safety is safe. This theory helped to discover the fault lines and work to correct them through updating and developing the proposed idea to meet the challenge.

5. Approach: Next time I will select a different tool which is Story Telling (Module 2) because this...
'''

Expected Output:
1. PII Candidates and Reasons:
- X customer
  - References: "X customer"
  - Reason: This cannot be used in a malicous way to identify a person, so this is not a PII.
  - Decision: Not PII


2. Replacement Dictionary:
{{}}


Example 3:
```
Input Example:
Challenge & Selection     Youssef Schneider, who has muscular discomfort, is one of the biggest fans of Trabzonspor. His  biggest dream was to score a goal for Trabzonspor one day. I wanted to realize Youssef's dreams  to show how 5G technology will remove the limits. In order to introduce such technology, I  wanted to make a little kid's dreams come true. Then I planned all our work and infrastructure  accordingly.    Application    For this, I used storytelling tool which I believe is the most appropriate one for this case. I  used VR glasses and microphones so that Youssef can direct the football player instantly. Youssef  instantly saw the camera attached to the football player and directed him and scored his dream  goal. Thus, I realized Youssef's dream of scoring beautiful goals like Trabzonspor players, one of  his biggest dreams.   Before the 3 December Disabled Day, all work was done in the Trabzonspor training field,  preparations were completed. Thanks to 5G technology, Youssef saw the field with VR glasses  at a delay rate of only 1-10 milliseconds through Sörloth's eyes, and he scored Erce Kardeşler  with the commands he gave!    Insight & Approach    Beginning from the moment it was released, my ad film got supported by every football club  and their fans alongside Trabzonspor. On TV and digital we reached 10,3 million people in  total, with 689 thousand impressions and 2,5 million views. Our #PeopleHaveNoBoundaries  hashtag became TT on Twitter and shared for 3.6 thousand times.  Turkey’s most prominent  TV channels, 19 newspapers and 144 news sites announced Youssef’s goal on their news  broadcasts. What I might do differently next time is that I would plan my mind mapping in a  more specific way. But I am sure I’d use storytelling tool again.
```

1. PII Candidates and Reasons:
- Youssef Schneider
  - References: "Youssef Schneider", "Youssef", "I wanted to realize Youssef's dreams", "I realized Youssef's dream", "announced Youssef’s goal"
  - Reason: The full name can directly be used to identify a specific individual, therefore it is considered as a PII.
  - Decision: PII
- Trabzonspor
  - References: "Trabzonspor", "Trabzonspor players"
  - Reason: This is the name of a football club and does not directly indicate an individual's identity, therefore it is not PII.
  - Decision: Not PII
- Sörloth
  - References: "through Sörloth's eyes"
  - Reason: The context indicates that Sörloth is an individual's name, thus this is considered PII as it can be used to identify a specific person.
  - Decision: PII
- Erce Kardeşler
  - References: "he scored Erce Kardeşler"
  - Reason: An individual's name which can be used to identify a specific person. Thus, it's considered PII.
  - Decision: PII

2. Replacement Dictionary:
{{
    ("Youssef Schneider", "Youssef)": "[NAME1]",
    ("Sörloth",): "[NAME2]",
    ("Erce Kardeşler",): "[NAME3]"
}}
"""


class Anonymizer(BaseProcessor):
    def __init__(self, client: Any):
        super().__init__(client)
        self.system_prompt = chain_of_thought_prompt
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

        parts = combined_output.split("2. Replacement Dictionary:")
        if len(parts) < 2:
            error_message = "LLM Produced unparsable output."
            self._log_error(error_message, combined_output)
            raise UnparsableLLMOutputException(error_message)

        replacement_dict = self._extract_and_parse_replacement_dict(parts[1])
        anonymized_text = self._parse_replacements(text, replacement_dict)
        return {"anonymized_text": anonymized_text, "replacement_dict": replacement_dict}
