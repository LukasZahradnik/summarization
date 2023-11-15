import os

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
import streamlit as st
import tiktoken

client = None
model = "gpt-3.5-turbo"
max_tokens = 3800


def init_session():
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("api_key", st.secrets["api_key"])


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    return len(encoding.encode(string))


def run_summarize_llm(prompt_template, text):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name=model)
    texts = text_splitter.split_text(text)

    docs = [Document(page_content=t) for t in texts]
    llm = ChatOpenAI(temperature=0, openai_api_key=st.session_state.api_key, model_name=model)

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    num_tokens = num_tokens_from_string(text, model)

    if num_tokens < max_tokens:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
    return chain.run(docs)


def run_qa_llm(prompt_template, query, context):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name=model)
    texts = text_splitter.split_text(context)

    docs = [Document(page_content=t) for t in texts]
    llm = ChatOpenAI(temperature=0, openai_api_key=st.session_state.api_key, model_name=model)

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    num_tokens = num_tokens_from_string(context + query, model)

    if num_tokens < max_tokens:
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    else:
        chain = load_qa_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
    return chain.run(docs)


def summarize(text):
    prompt_template = """Write a summary of the following text:

    {text}

    SUMMARY IN ENGLISH:"""

    summary = run_summarize_llm(prompt_template, text)
    return summary


def answer(question, context):
    prompt_template = """Answer question about the following text:

    {context}
    """

    prompt_template += f"""

    The question:
    {question}

    ANSWER IN ENGLISH:"""

    summary = run_qa_llm(prompt_template, question, context)
    return summary


def draw_file_upload():
    audio_file = st.file_uploader("Upload Meeting Recording", type=["m4a"])

    if st.button("Summarize") and audio_file:
        with st.spinner('Processing...'):
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, audio_file.name)

            with open(file_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.session_state.audio_file_path = file_path

            with open(st.session_state.audio_file_path, 'rb') as f:
                st.session_state.transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                ).text

            st.session_state.summary = summarize(st.session_state.transcript)


def draw_tabs():
    transcription_tab, summarization_tab = st.tabs(["Transcript", "Summary"])
    with transcription_tab:
        if st.session_state.transcript:
            st.write(st.session_state.transcript)

    with summarization_tab:
        if st.session_state.summary:
            st.write(st.session_state.summary)


def draw_chat():
    st.subheader("Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("Thinking...")

            full_response = answer(prompt, st.session_state.transcript)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    global client

    init_session()
    client = OpenAI(api_key=st.session_state.api_key)

    st.title("Meeting Summarization")

    draw_file_upload()
    st.divider()

    draw_tabs()
    st.divider()

    if st.session_state.transcript:
        draw_chat()


if __name__ == "__main__":
    main()


