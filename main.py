import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict

import dspy
import humps
from ashley_protos.care.ashley.contracts.common.v1 import *
from ashley_protos.care.ashley.contracts.internal.v1 import *
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from openai import OpenAI
from pydantic import BaseModel

from pseudoanonymize.anonymization import Anonymizer
from pseudoanonymize.deanonymization import Deanonymizer
from pseudoanonymize.direct_json_anonymization import JSON_FEW_SHOT_PROMPT, JSON_SYSTEM_PROMPT, JsonDirectAnonymizer
from pseudoanonymize.dspy_anonmization import DspyAnon
from pseudoanonymize.exceptions import MaxRetriesExceededException
from pseudoanonymize.pipeline import PiplelineAnon
from pseudoanonymize.regex_anonymization import RegexAnon
from pseudoanonymize.utils import chunk_by_line, flatten_replacement_dict

app = FastAPI()

# set up
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
turbo = dspy.OpenAI(model='gpt-4o', max_tokens=4096, api_key=openai_api_key)
dspy.settings.configure(lm=turbo)

# models
anonymizer = Anonymizer(client=client)
model = "gpt-4o"
prompt = JSON_SYSTEM_PROMPT + JSON_FEW_SHOT_PROMPT
dierct_anonymizer_gpt4o = JsonDirectAnonymizer(client=client, model=model, system_prompt=prompt)

# model = "ft:gpt-3.5-turbo-0125:slingshot-daniel:pii-dist-gpt3:9ekTDaTH"
model = "ft:gpt-3.5-turbo-0125:slingshot-daniel:pii-dist-processed:9jcKs3kP"
prompt = JSON_SYSTEM_PROMPT
direct_anonymizer_gpt35ft = JsonDirectAnonymizer(client=client, model=model, system_prompt=prompt)


deanonymizer = Deanonymizer(client=client)
dspy_anonymizer = DspyAnon()


class CamelModel(BaseModel):
    id: str | None = None

    class Config:
        alias_generator = humps.camelize
        populate_by_name = True


class AnonymizeRequest(CamelModel):
    text: str
    retries: int = 5


class DeanonymizeRequest(CamelModel):
    text: str
    replacement_dict: Dict[Any, str]
    retries: int = 5


class AnonymizeResponse(CamelModel):
    anonymized_text: str
    replacement_dict: Dict[str, str]


class DeanonymizeResponse(CamelModel):
    deanonymized_text: str


class PseudoanonymizeResponse(CamelModel):
    anonymized_text: str
    replacement_dict: Dict[str, str]
    deanonymized_text: str


pipeline_model = PiplelineAnon([DspyAnon(), RegexAnon()])


@app.post("/anonymize_event_log")
async def anonymize_event_log(request: Request):
    def extract_conv_from_event_log(event_log: EventLog) -> str:
        """obtain the conversation as a string from the event log"""
        conv = ""
        for event in event_log.events:
            if event.therapist_message_completed:
                content = event.therapist_message_completed.message.content
                content = content.replace("\n", " ")
                conv += content + "\n"
            elif event.user_message_completed:
                content = event.user_message_completed.message.content
                content = content.replace("\n", " ")
                conv += content + "\n"
            # TODO: add user message amended
            # TODO: add therpsiedamended
            # TODO: add therpisted interupted (has ameneded message)
            # TODO: add critic finished thinking (thoughts -> thoughts)
            # TODO: session summaryGenerated -> summary -> message
            # TODO: session feedback

        return conv

    def copy_event_log(events: EventLog):
        # a simple deepcopy causes unexpected behavior, like data loss when returning the new object
        return EventLog().FromString(events.SerializeToString())

    def update_message_contents(events: EventLog, anonymized_conversation: str) -> EventLog:
        """
        update the message contents in the new event log with the anonymized text
        """
        new_event = copy_event_log(events)

        anonymized_text_turns = anonymized_conversation.split(
            "\n"
        )  # was formatted with newlines per turn, so split on newlines to get each turn.

        j = 0
        for i, event in enumerate(events.events):
            if event.therapist_message_completed:
                new_event.events[i].therapist_message_completed.message.content = anonymized_text_turns[j]
                j += 1
            elif event.user_message_completed:
                new_event.events[i].user_message_completed.message.content = anonymized_text_turns[j]
                j += 1
        return new_event

    try:
        request_body = await request.body()
        events = EventLog.FromString(request_body)
        print("debug: ")
        print(events)
        print("-----------------")
        conv = extract_conv_from_event_log(events)
        anonymized_conversation, replacement_dict = pipeline_model.predict({"text": conv})

        anonymized_event_log = update_message_contents(events, anonymized_conversation)
        return Response(content=anonymized_event_log.SerializeToString(), media_type="application/protobuf")
    except Exception as e:
        raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize(request: AnonymizeRequest):
    try:
        prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        flattened_dict = flatten_replacement_dict(prediction["replacement_dict"])
        return AnonymizeResponse(anonymized_text=prediction["anonymized_text"], replacement_dict=flattened_dict)
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anonymize_v2", response_model=AnonymizeResponse)
async def anonymize_v2(request: AnonymizeRequest):
    try:
        text_chunks = chunk_by_line(request.text, 2000)
        print(len(text_chunks))
        anonymized_text_list = []
        replacement_dict = {}

        with ThreadPoolExecutor() as executor:
            futures = []
            for chunk in text_chunks:
                future = executor.submit(direct_anonymizer_gpt35ft.retry_prediction, {"text": chunk}, request.retries)
                futures.append(future)

            for future in futures:
                try:
                    prediction = future.result()
                except MaxRetriesExceededException as e:
                    raise HTTPException(status_code=500, detail=str(e))

                anonymized_text_list.append(prediction["anonymized_text"])
                replacement_dict.update(prediction["replacement_dict"])

        anonymized_text = "\n".join(anonymized_text_list)
        return AnonymizeResponse(anonymized_text=anonymized_text, replacement_dict=replacement_dict)
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize(request: DeanonymizeRequest):
    try:
        prediction = deanonymizer.retry_prediction(
            {"text": request.text, "replacement_dict": request.replacement_dict}, request.retries
        )
        return DeanonymizeResponse(deanonymized_text=prediction["deanonymized_text"])
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pseudoanonymize", response_model=PseudoanonymizeResponse)
async def pseudoanonymize(request: AnonymizeRequest):
    try:
        anonymized_prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        anonymized_text = anonymized_prediction["anonymized_text"]
        replacement_dict = flatten_replacement_dict(anonymized_prediction["replacement_dict"])

        deanon_prediction = deanonymizer.retry_prediction(
            {"text": anonymized_text, "replacement_dict": replacement_dict}, request.retries
        )
        return PseudoanonymizeResponse(
            anonymized_text=anonymized_text,
            replacement_dict=replacement_dict,
            deanonymized_text=deanon_prediction["deanonymized_text"],
        )
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anonymize_ash_conversation")
async def anonymize_ash_conversation(request: Request):
    def gather_text(entries, max_length, sep_seq):
        """
        Gather text from user and therapist messages within the given length limit.

        Args:
            entries (list): List of conversation entries.
            max_length (int): Maximum length of the gathered text.
            sep_seq (str): Separator sequence used between messages.

        Returns:
            tuple: A tuple containing gathered text, indices, and types.
        """
        gathered_text = ""
        indices = []
        types = []

        for j in range(len(entries)):
            user_msg = entries[j].user_message.content
            ther_msg = entries[j].therapist_message.content

            if user_msg and len(gathered_text) + len(user_msg) <= max_length:
                gathered_text += user_msg + sep_seq
                indices.append(j)
                types.append("user")

            if ther_msg and len(gathered_text) + len(ther_msg) <= max_length:
                gathered_text += ther_msg + sep_seq
                indices.append(j)
                types.append("therapist")

        return gathered_text, indices, types

    def update_entries_with_anonymized_text(entries, indices, types, anon_messages):
        """
        Update the conversation entries with the anonymized text.

        Args:
            entries (list): List of conversation entries.
            indices (list): Indices of the entries to be updated.
            types (list): Types of the messages (user or therapist).
            anon_messages (list): Anonymized messages.
        """
        for j_idx, msg_type in zip(indices, types):
            if msg_type == "user":
                entries[j_idx].user_message.content = anon_messages.pop(0)
            else:
                entries[j_idx].therapist_message.content = anon_messages.pop(0)

    request_body = await request.body()
    conv = ReadConversationForUserResponse.FromString(request_body)
    new_sessions = deepcopy(conv.sessions)

    max_length = 2000
    sep_seq = "-\n-"

    for i in range(len(new_sessions)):
        entries = new_sessions[i].entries
        gathered_text, indices, types = gather_text(entries, max_length, sep_seq)

        if gathered_text:
            anon_messages, _ = dspy_anonymizer.predict(dict(text=gathered_text.strip()))
            anon_messages.split(sep_seq)
            update_entries_with_anonymized_text(entries, indices, types, anon_messages)

    conv.sessions[:] = new_sessions
    return Response(content=conv.SerializeToString(), media_type="application/protobuf")


# TODO: change endpoint called from the kotlin side to /anonymize_ash_conversation. This is just a temporary solution.
@app.post("/read_conversation_for_user")
async def anonymize_ash_conversation_v1(request: Request):
    return await anonymize_ash_conversation(request)
