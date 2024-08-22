from ashley_protos.care.ashley.contracts.common.v1 import *
from ashley_protos.care.ashley.contracts.internal.v1 import *


def extract_conv_from_event_log(event_log: EventLog) -> str:
    """obtain the conversation as a string from the event log"""
    conv = ""
    for event in event_log.events:
        if hasattr(event, "therapist_message_completed"):
            if event.therapist_message_completed.message.content:
                content = event.therapist_message_completed.message.content
                content = content.replace("\n", " ")
                conv += content + "\n"
        if hasattr(event, "user_message_completed"):
            if event.user_message_completed.message.content:
                content = event.user_message_completed.message.content
                content = content.replace("\n", " ")
                conv += content + "\n"

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
        if hasattr(event, "therapist_message_completed"):
            if event.therapist_message_completed.message.content:
                new_event.events[i].therapist_message_completed.message.content = anonymized_text_turns[j]
                j += 1
        if hasattr(event, "user_message_completed"):
            if event.user_message_completed.message.content:
                new_event.events[i].user_message_completed.message.content = anonymized_text_turns[j]
                j += 1
    return new_event
