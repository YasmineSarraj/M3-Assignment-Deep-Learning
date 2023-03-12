import streamlit as st
from streamlit.components.v1 import html
import os
import PyPDF2
import requests

def get_pdf_text(pdf_path):
    # creating a pdf file object
    pdfFileObj = open(pdf_path, 'rb')
        
    # creating a pdf reader object
    pdf_reader = PyPDF2.PdfReader(pdfFileObj)

    # extract text
    total_text_list = []

    for i in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[i].extract_text()
        total_text_list.append(page_text)

    pdf_text = " ".join(total_text_list)
    pdfFileObj.close()

    return pdf_text

headers = {"Authorization": st.secrets["HF_AUTH"]}

def create_tags(payload):
    API_URL_TAGS = "https://api-inference.huggingface.co/models/fabiochiu/t5-base-tag-generation"
    
    response = requests.post(API_URL_TAGS, headers=headers, json=payload)
    return response.json()

def summarize_text(payload):
    API_URL = "https://api-inference.huggingface.co/models/yasminesarraj/flan-t5-small-samsum"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# Start of the app code

tab_general_topics, tab_your_paper = st.tabs(["Research topics", "Summarize your paper(s)"])

with tab_general_topics:
    html("", height=10)

    st.header("See the status of a research topic through a summary of the most cited papers")

    st.selectbox("Select a research topic", ["Artificial Intelligence", "Sustainability", "Cooking"])

with tab_your_paper:
    html("", height=10)

    st.markdown("""
### Simply upload one or multiple PDFs and we summarize the content for you!
    """)

    pdf_files = st.file_uploader("Upload your paper as a pdf", type=[".pdf"], accept_multiple_files=True, help="You can summarize one or also multiple papers at once. The file format needs to be a pdf.")
    if pdf_files:
        recently_added = []
        for pdf in pdf_files:
            # Saving the files
            pdf_data = pdf.getvalue()
            pdf_path = os.path.join("pdfs", pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_data)
                recently_added.append(pdf_path)

        pdfs_content_list = []
        for recent_pdf in recently_added:
            # Reading the pdf files
            pdf_content = get_pdf_text(recent_pdf)
            print("**", pdf_content)
            pdfs_content_list.append(pdf_content)

            # Delete the files
            os.remove(recent_pdf)

        all_text_together = " ".join(pdfs_content_list)

        try:
            tags = create_tags({
                "inputs": all_text_together,
            })[0]["generated_text"]
            tags_available = True
        except:
            tags_available = False
        	
        summary = summarize_text({
            "inputs": all_text_together
        })[0]["summary_text"]

        col1, col2 = st.columns(2)
        with col1:
            if len(recently_added) > 1:
                st.markdown("#### Summary of your paper(s):")
            else:
                st.markdown("#### Summary of your paper:")
            st.write(summary)

        if tags_available == True:
            with col2:
                if len(recently_added) > 1:
                    st.markdown("#### Identified topics of your paper(s):")
                else:
                    st.markdown("#### Identified topics of your paper:")
                st.write(tags)

        with st.expander("See your total text"):
            st.write(all_text_together)