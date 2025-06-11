
pub fn get_app_content(proj_type: &str) -> &'static str {
    match proj_type {
        "Hello World" => "print('Hello from KR!')\n",
        "API" => r#"from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello API'

if __name__ == '__main__':
    app.run()"#,
        _ => "",
    }
}

pub fn get_fastapi_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        ("main.py", r#"from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to KR - FastAPI Project"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)"#),
        ("requirements.txt", "fastapi\nuvicorn\n"),
        ("README.md", "# KR FastAPI Project\n\nGenerated using KR - Kompiler Ready"),
    ]
}

pub fn get_ml_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        ("train.py", r#"import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample ML model
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = LinearRegression()
model.fit(X, y)

print("Model trained!")"#),
        ("predict.py", r#"from train import model

# Predict example
prediction = model.predict([[4]])
print(f"Prediction: {prediction[0]}")"#),
        ("requirements.txt", "numpy\npandas\nscikit-learn\n"),
        ("README.md", "# KR ML Project\n\nGenerated using KR - Kompiler Ready"),
    ]
}

pub fn get_dl_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        ("train.py", r#"import torch
import torch.nn as nn
import torch.optim as optim

# Simple neural network
net = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    labels = torch.tensor([[2.0], [4.0], [6.0]])

    outputs = net(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("DL model trained!")"#),
        ("requirements.txt", "torch\n"),
        ("README.md", "# KR DL Project\n\nGenerated using KR - Kompiler Ready"),
    ]
}

#[allow(dead_code)]
pub fn get_streamlit_docling_langchain_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "app.py",
            r##"
import streamlit as st
from langchain_docling import DoclingLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
import os

st.set_page_config(page_title="KR - Document Q&A", layout="wide")
st.title("📄 KR - Kompiler Ready: Document Q&A")

HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:"
)

def load_and_index(file_path):
    loader = DoclingLoader(file_path=file_path)
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    vectorstore = Milvus.from_documents(documents=docs, embedding=embeddings, collection_name="docling_demo")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def setup_qa_chain(retriever):
    llm = HuggingFaceEndpoint(repo_id=GEN_MODEL_ID, huggingfacehub_api_token=HF_TOKEN)
    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, question_answer_chain)

def main():
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("Parsing document..."):
            retriever = load_and_index(temp_path)

        st.success("Document loaded and indexed!")

        question = st.text_input("Ask a question about the document:")
        if question:
            chain = setup_qa_chain(retriever)
            with st.spinner("Thinking..."):
                result = chain.invoke({"input": question})
                st.markdown("*** Answer")
                st.write(result["answer"])
                st.markdown("*** Sources")
                for i, doc in enumerate(result["context"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.code(doc.page_content[:500] + "...")

if __name__ == "__main__":
    main()
"##,
        ),
        (
            "requirements.txt",
            r#"
streamlit
langchain-docling
langchain-huggingface
langchain-milvus
langchain-core
huggingface_hub
sentence-transformers
torch
pymilvus
python-dotenv
"#,
        ),
        ("README.md", "# KR Streamlit + Docling + LangChain\n\nGenerated using KR - Kompiler Ready"),
    ]
}