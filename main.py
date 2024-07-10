import os
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
from pseudoanonymize.dspy_anonmization import DspyAnon
from pseudoanonymize.exceptions import MaxRetriesExceededException
from pseudoanonymize.utils import flatten_replacement_dict

app = FastAPI()

# set up
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
turbo = dspy.OpenAI(model='gpt-4o', max_tokens=4096, api_key=openai_api_key)
dspy.settings.configure(lm=turbo)

# models
anonymizer = Anonymizer(client=client)
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


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize(request: AnonymizeRequest):
    try:
        prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        flattened_dict = flatten_replacement_dict(prediction["replacement_dict"])
        return AnonymizeResponse(anonymized_text=prediction["anonymized_text"], replacement_dict=flattened_dict)
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
