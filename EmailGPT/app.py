"""Python file to serve as the frontend"""
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
import os

with open('.env', 'r') as f:
    env_file = f.readlines()
envs_dict = {key.strip("'"): value.strip("\n") for key, value in [(i.split('=')) for i in env_file]}
os.environ['OPENAI_API_KEY'] = envs_dict['OPENAI_API_KEY']

template = """
    Below is an email that may be poorly worded.
    Your goal is to:
    - Properly format the email
    - Convert the input text to a specified tone
    - Convert the input text to a specified dialect

    Here are some examples different Tones:
    - Formal: We went to Barcelona for the weekend. We have a lot of things to tell you.
    - Informal: Went to Barcelona for the weekend. Lots to tell you.  

    Here are some examples of words in different dialects:
    - American English: French Fries, cotton candy, apartment, garbage, cookie, green thumb, parking lot, pants, windshield
    - British English: chips, candyfloss, flag, rubbish, biscuit, green fingers, car park, trousers, windscreen
    - Deutsch: Pommes, Zuckerwatte, Flagge, Müll, Keks, Grüne Finger, Parkplatz, Hose, Windschutzscheibe

    Below is the email, tone, and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}

    YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["tone", "dialect", "email"],
    template=template,
)


def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    return llm


llm = load_LLM()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Formalize Email", page_icon=":robot:")
st.header("Formalize Text")
st.markdown("Often professionals would like to improve their emails, but don't have the skills to do so. This tool \
                will help you improve your email skills by converting your emails into a more professional format.")

st.markdown("## Enter Your Email To Convert")

col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        'Which tone would you like your email to have?',
        ('Formal', 'Informal'))

with col2:
    option_dialect = st.selectbox(
        'Which Language would you like?',
        ('American English', 'British English', 'Deutsch', 'Kolsch', 'Schwabisch'))


def get_text():
    input_text = st.text_area("", placeholder="Your Email...", key="email_input")
    return input_text


email_input = get_text()

st.markdown("### Your Converted Email:")

if email_input:
    output = llm(prompt.format(tone=option_tone, dialect=option_dialect, email=email_input))

    st.write(output)