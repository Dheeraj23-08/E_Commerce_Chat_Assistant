from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()


small_talk_groq=Groq()

def small_talk(question):
    prompt = f"""
    You are a friendly and intelligent AI assistant designed to engage in natural small talk.

    When a user asks a question or starts a conversation that falls under the 'small_talk' category 
    (such as greetings, personal questions, jokes, or casual chat),
    respond in a warm, conversational, and human-like manner.

    Keep your tone positive, polite, and engaging.
    If the question is unclear or unusual, respond gracefully and keep the conversation flowing.

    Keep answers brief and polite (max 15 words).

    Avoid giving robotic or repetitive answers.

    QUESTION: {question}
    """


    completion = small_talk_groq.chat.completions.create(
        model=st.secrets['GROQ_MODEL'],
        messages=[{"role": "user","content":prompt}]
    )
    return completion.choices[0].message.content


if __name__=="__main__":
    print(small_talk('how are you'))










