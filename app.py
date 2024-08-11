###Summarize the data from yt or any other links provided
##use validator install==0.28.1 
##use youtube transcript_api 
##install pytube as well

import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
st.secrets["GROQ_API_KEY"]

#streamlit app
st.set_page_config(page_title="Data Summarizer", page_icon=":tada:", layout="wide")
st.title("YT Video Summarizer")
st.subheader("Summarize data from YouTube videos.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-IT")

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
    Summarize the following text:
    {text}
    """
)

url = st.text_input("Enter the URL..", label_visibility="collapsed")
if st.button("Summarize"):
    try:
        with st.spinner("Summarizing..."):
            if validators.url(url):
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                docs = loader.load()
            else:
                loader = UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
                docs = loader.load()
            summary_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
            summary = summary_chain.invoke(docs)
            output = summary["output_text"]
            st.write(output)

    except Exception as e:
        st.error(f"An error occurred: {e}")