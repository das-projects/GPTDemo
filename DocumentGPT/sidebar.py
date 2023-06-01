import streamlit as st


def faq():
    st.markdown(
        """
# FAQ
## How does DocumentGPT work?
When you upload a document, it will be divided into smaller chunks 
and stored in a special type of database called a vector index 
that allows for semantic search and retrieval.

When you ask a question, DocumentGPT will search through the
document chunks and find the most relevant ones using the vector index.
Then, it will use GPT-3.5 to generate a final answer.

## Why does it take so long to index my document?
If you are using a free OpenAI API key, it will take a while to index
your document. This is because the free API key has strict [rate limits](https://platform.openai.com/docs/guides/rate-limits/overview).
To speed up the indexing process, you can use a paid API key.

## What do the numbers mean under each source?
For a PDF document, you will see a citation number like this: 3-12. 
The first number is the page number and the second number is 
the chunk number on that page. For DOCS and TXT documents, 
the first number is set to 1 and the second number is the chunk number.

## Are the answers 100% accurate?
No, the answers are not 100% accurate. DocumentGPT uses GPT-3.5 to generate
answers. GPT-3.5 is a powerful language model, but it sometimes makes mistakes 
and is prone to hallucinations. Also, DocumentGPT uses semantic search
to find the most relevant chunks and does not see the entire document,
which means that it may not be able to find all the relevant information and
may not be able to answer all questions (especially summary-type questions
or questions that require a lot of context from the document).

But for most use cases, DocumentGPT is very accurate and can answer
most questions. Always check with the sources to make sure that the answers
are correct.
"""
    )


def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
            "2. Upload a pdf, docx, or txt file📄\n"
            "3. Ask a question about the document💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )

        if api_key_input:
            set_openai_api_key(api_key_input)

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖DocumentGPT allows you to ask questions about your "
            "documents and get accurate answers with instant citations. "
        )
        st.markdown(
            "This tool is a work in progress. "
            "All feedback and suggestions💡 are appreciated!"
        )
        st.markdown("---")

        faq()
