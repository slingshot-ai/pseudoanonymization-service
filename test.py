import requests
import os

example = """
assistant: Hi Alex, nice to meet you. I've just had a look at your intake form. Can you tell me a bit more about yourself, and maybe why you're here?
user: I'm thirty five years old and I I'm here because I am done with depression and don't have a job right now.
"""

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

def write_to_file(filename, content):
    with open(os.path.join(output_dir, filename), 'w') as file:
        file.write(content)

# Test the /anonymize endpoint
anonymize_response = requests.post("http://127.0.0.1:8000/anonymize", json={"text": example})
if anonymize_response.status_code == 200:
    anonymized_data = anonymize_response.json()
    write_to_file('anonymized_text.txt', anonymized_data["anonymized_text"])
    write_to_file('replacement_dict.txt', str(anonymized_data["replacement_dict"]))
else:
    print("Anonymize request failed:", anonymize_response.status_code, anonymize_response.text)

# Test the /deanonymize endpoint
if anonymize_response.status_code == 200:
    deanonymize_response = requests.post("http://127.0.0.1:8000/deanonymize", json={
        "text": anonymized_data["anonymized_text"],
        "replacement_dict": anonymized_data["replacement_dict"]
    })
    if deanonymize_response.status_code == 200:
        deanonymized_data = deanonymize_response.json()
        write_to_file('deanonymized_text.txt', deanonymized_data["deanonymized_text"])
    else:
        print("Deanonymize request failed:", deanonymize_response.status_code, deanonymize_response.text)

# Test the /pseudoanonymize endpoint
pseudoanonymize_response = requests.post("http://127.0.0.1:8000/pseudoanonymize", json={"text": example})
if pseudoanonymize_response.status_code == 200:
    pseudoanonymize_data = pseudoanonymize_response.json()
    write_to_file('pseudoanonymized_anonymized_text.txt', pseudoanonymize_data["anonymized_text"])
    write_to_file('pseudoanonymized_deanonymized_text.txt', pseudoanonymize_data["deanonymized_text"])
else:
    print("Pseudoanonymize request failed:", pseudoanonymize_response.status_code, pseudoanonymize_response.text)
