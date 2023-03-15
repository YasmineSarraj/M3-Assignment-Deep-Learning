from flask import Flask, jsonify, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from flask_wtf.file import FileAllowed, FileRequired, FileField
from wtforms.fields import MultipleFileField
from wtforms.validators import DataRequired, Length, ValidationError, EqualTo, Email
import os
import PyPDF2
import requests
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import math
import time

if not os.path.exists(os.path.join("pdfs")):
    os.mkdir("pdfs")

app = Flask(__name__)

load_dotenv(find_dotenv())
app.config["SECRET_KEY"] = os.getenv("APP_PWD")

# Forms
class FileUploadForm(FlaskForm):
    files = MultipleFileField('Bild dieser Station', validators=[FileAllowed(["pdf"], 'PDFs only!')])
    submit = SubmitField("Speichern")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

headers = {"Authorization": f"Bearer {os.getenv('HF_AUTH')}"}

def create_tags(payload):
    API_URL_TAGS = "https://api-inference.huggingface.co/models/fabiochiu/t5-base-tag-generation"
    
    response = requests.post(API_URL_TAGS, headers=headers, json=payload)
    return response.json()

def summarize_text(payload):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


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

all_text_together = ""
emb_sentences = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-your-paper", methods=["POST", "GET"])
def upload():
    form = FileUploadForm()
    if form.validate_on_submit():
        # Saving the uploaded PDFs
        recently_uploaded = []
        for every_file in form.files.data:
            every_file.save(os.path.join("pdfs", every_file.filename))
            recently_uploaded.append(os.path.join("pdfs", every_file.filename))
        
        pdfs_content_list = []
        for recent_pdf in recently_uploaded:
            # Reading the pdf files
            pdf_content = get_pdf_text(recent_pdf)
            pdfs_content_list.append(pdf_content)

            # Delete the files
            os.remove(recent_pdf)
        global all_text_together
        all_text_together = " ".join(pdfs_content_list)
        all_text_together = re.sub("\n", "", all_text_together)
        global all_text_together_sentences
        all_text_together_sentences = all_text_together.split(".")

        if len(all_text_together_sentences) > 50:
            turns = int(math.ceil(len(all_text_together_sentences) / 50))
            start_num = 0
            end_num = 50
            summary_list = []
            tags_list = []
            for iteration in range(turns):
                print(iteration, "/", turns)
                all_text_together = " ".join(all_text_together_sentences[start_num:end_num])
                tags = create_tags({"inputs": all_text_together})[0]["generated_text"]
                summary = summarize_text({"inputs": "summarize: "+all_text_together})[0]["summary_text"]
                summary_list.append(summary)
                tags_list.append(tags)
                start_num += 50
                end_num += 50
                time.sleep(.5)

            total_summaries = " ".join(summary_list)
            all_tags = " ".join(tags_list)
            tags = create_tags({"inputs": all_tags})[0]["generated_text"]
            tags = tags.split(",")
            summary = summarize_text({"inputs": "summarize: "+total_summaries})[0]["summary_text"]

        else:
            tags = create_tags({"inputs": all_text_together})[0]["generated_text"]
            tags = tags.split(",")
            summary = summarize_text({"inputs": "summarize: "+all_text_together})[0]["summary_text"]

        return render_template("custom_paper.html", total_text = all_text_together, tags=tags, summary=summary)
    return render_template("upload.html", form=form)

topics = ["Artificial Intelligence", "Sustainability", "Medicine", "Marketing", "Finance", "Accounting", "Mechanical Engineering"]
papers = ["Attention is all you need", "Machine learning in medicine", "The power of linear regression"]
topics_and_papers = topics + papers

def find_n_largest(list_item, n):
    largest_idxs =[]
    for i in range(n):
        list_item = np.asarray(list_item)
        largest_val_idx = list_item.argmax()
        largest_idxs.append(largest_val_idx)
        list_item = list_item.tolist()
        del list_item[largest_val_idx]
    return largest_idxs

@app.route("/search", methods=["POST"])
def search():
    if request.method == "POST":
        emb_search = sbert_model.encode(request.get_json().get("current_search"))
        emb_results = sbert_model.encode(topics_and_papers)

        cos_sim = util.cos_sim(emb_search, emb_results).tolist()[0]
        results_to_send = []
        for x in find_n_largest(cos_sim, 5):
            if topics_and_papers[x] in topics:
                _ = {"item": topics_and_papers[x], "type": "topic"}
            elif topics_and_papers[x] in papers:
                _ = {"item": topics_and_papers[x], "type": "paper"}
            
            if _ not in results_to_send:
                results_to_send.append(_)

        return jsonify(results_to_send)

@app.route("/subject-search", methods=["POST"])
def subject_search():
    if request.method == "POST":
        emb_subject = sbert_model.encode(request.get_json().get("subject"))
        global emb_sentences

        top_sentence = util.cos_sim(emb_subject, emb_sentences)[0].argmax().item()
        following_sentences = top_sentence + 5
        if following_sentences:
            to_send = all_text_together_sentences[top_sentence:following_sentences]
            to_send = ". ".join(to_send)
        else:
            to_send = all_text_together_sentences[top_sentence:]
            to_send = ". ".join(to_send)

        return jsonify({"text": to_send})


@app.route("/encode-text", methods=["GET"])
def encode_text():
    global all_text_together_sentences
    global emb_sentences
    emb_sentences = sbert_model.encode(all_text_together_sentences)
    print("Done encoding")
    return "okay"


if __name__ == "__main__":
    app.run()