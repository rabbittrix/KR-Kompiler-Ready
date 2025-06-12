// tempates.rs
use crate::utils::fs_utils;
use std::path::Path;

pub fn create_template_structure(_path: &Path, proj_type: &str) {
    let structure = match proj_type {
        "Hello World" => get_app_content(),
        "API" => get_api_structure(),
        "FastAPI" => get_fastapi_structure(),
        "FastAPI Simple" => get_fastapi_simple_structure(),
        "Modular" => get_modular_structure(),
        "Microservices" => get_microservices_structure(),
        "ML" => get_ml_structure(),
        "DL" => get_dl_structure(),
        "Django" => get_django_structure(),
        "Streamlit" => get_streamlit_structure(),
        "Streamlit + Docling + LangChain" => get_streamlit_docling_langchain_structure(),
        _ => vec![],
    };

    for (filename, content) in structure {
        fs_utils::write_file(&_path.join(filename), content).expect("Failed to write file");
    }
}

pub fn get_app_content() -> Vec<(&'static str, &'static str)> {
    vec![
        ("app.py", "print('Hello from KR!')\n"),
        (
            "README.md",
            "# Hello World Project\n\nGenerated using KR - Kompiler Ready",
        ),
        ("requirements.txt", ""),
    ]
}

pub fn get_api_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "app.py",
            r#"from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
items = {
    "1": "Item One",
    "2": "Item Two",
    "3": "Item Three",
    "4": "Item Four",
    "5": "Item Five",
    "6": "Item Six",
    "7": "Item Seven",
    "8": "Item Eight",
    "9": "Item Nine"}

# Create
@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    item_id = len(items) + 1
    items[item_id] = data.get('name')
    return jsonify({"id": item_id, "name": items[item_id]}), 201

# Read All
@app.route('/items', methods=['GET'])
def get_items():
    return jsonify([{ "id": k, "name": v } for k, v in items.items()])

# Read Single
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    name = items.get(item_id)
    if not name:
        return jsonify({"error": "Item not found"}), 404
    return jsonify({"id": item_id, "name": name})

# Update
@app.route('/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    if item_id not in items:
        return jsonify({"error": "Item not found"}), 404
    data = request.get_json()
    items[item_id] = data.get('name')
    return jsonify({"id": item_id, "name": items[item_id]})

# Delete
@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    if item_id not in items:
        return jsonify({"error": "Item not found"}), 404
    del items[item_id]
    return jsonify({"message": "Item deleted"})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')"#,
        ),
        ("requirements.txt", "flask\n"),
        (
            "README.md",
            "# KR API Project\n\nGenerated using KR - Kompiler Ready",
        ),
    ]
}

pub fn get_fastapi_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "main.py",
            r#"from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from typing import List
from contextlib import asynccontextmanager

from database import engine, SessionLocal
from models import Base, Item
from schemas import ItemCreate, ItemResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("Database tables created")
    yield

app = FastAPI(lifespan=lifespan)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/items/", response_model=ItemResponse)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/", response_model=List[ItemResponse])
def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = db.query(Item).offset(skip).limit(limit).all()
    return items

@app.get("/items/{item_id}", response_model=ItemResponse)
def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.put("/items/{item_id}", response_model=ItemResponse)
def update_item(item_id: int, item: ItemCreate, db: Session = Depends(get_db)):
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db_item.name = item.name
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)
    return db_item

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(db_item)
    db.commit()
    return {"detail": "Item deleted"}"#,
        ),
        (
            "models.py",
            r#"from sqlalchemy import Column, Integer, String
from database import Base

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))"#,
        ),
        (
            "schemas.py",
            r#"from pydantic import BaseModel

class ItemBase(BaseModel):
    name: str
    description: str | None = None

class ItemCreate(ItemBase):
    pass

class ItemResponse(ItemBase):
    id: int

    class Config:
        from_attributes = True"#,
        ),
        (
            "database.py",
            r#"from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()"#,
        ),
        (
            "requirements.txt",
            r#"fastapi
uvicorn
sqlalchemy
pydantic
python-dotenv
typing_extensions
"#,
        ),
        (
            "README.md",
            r#"# KR FastAPI Project

This is a basic CRUD API built with FastAPI and SQLAlchemy ORM.

## Features

- Create, Read, Update, Delete items
- SQLite backend by default
- Pydantic model validation
- Modular structure ready to expand

## Setup

```bash
cd your_project_folder
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
uvicorn main:app --reload
```

## Usage

- Start the server: `uvicorn main:app --reload`
- View API docs: http://localhost:8000/docs
- Test endpoints: http://localhost:8000/items/"#,
        ),
    ]
}

/// Simple FastAPI template with app.py entry point
pub fn get_fastapi_simple_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "app.py",
            r#"from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Car Brands API", description="A simple CRUD API for car brands")

# Data model
class CarBrand(BaseModel):
    id: Optional[int] = None
    name: str
    country: str
    founded_year: Optional[int] = None

# In-memory database
car_brands = [
    {"id": 1, "name": "Toyota", "country": "Japan", "founded_year": 1937},
    {"id": 2, "name": "BMW", "country": "Germany", "founded_year": 1916},
    {"id": 3, "name": "Ford", "country": "USA", "founded_year": 1903},
    {"id": 4, "name": "Mercedes-Benz", "country": "Germany", "founded_year": 1926},
    {"id": 5, "name": "Honda", "country": "Japan", "founded_year": 1948},
]

# CRUD Operations
@app.get("/")
def read_root():
    return {"message": "Welcome to Car Brands API"}

@app.get("/brands/", response_model=List[CarBrand])
def get_brands():
    return car_brands

@app.get("/brands/{brand_id}", response_model=CarBrand)
def get_brand(brand_id: int):
    brand = next((brand for brand in car_brands if brand["id"] == brand_id), None)
    if brand is None:
        raise HTTPException(status_code=404, detail="Brand not found")
    return brand

@app.post("/brands/", response_model=CarBrand)
def create_brand(brand: CarBrand):
    new_id = max(brand["id"] for brand in car_brands) + 1
    new_brand = {"id": new_id, "name": brand.name, "country": brand.country, "founded_year": brand.founded_year}
    car_brands.append(new_brand)
    return new_brand

@app.put("/brands/{brand_id}", response_model=CarBrand)
def update_brand(brand_id: int, brand: CarBrand):
    brand_index = next((i for i, b in enumerate(car_brands) if b["id"] == brand_id), None)
    if brand_index is None:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    car_brands[brand_index] = {"id": brand_id, "name": brand.name, "country": brand.country, "founded_year": brand.founded_year}
    return car_brands[brand_index]

@app.delete("/brands/{brand_id}")
def delete_brand(brand_id: int):
    brand_index = next((i for i, b in enumerate(car_brands) if b["id"] == brand_id), None)
    if brand_index is None:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    deleted_brand = car_brands.pop(brand_index)
    return {"message": f"Brand {deleted_brand['name']} deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)"#,
        ),
        (
            "requirements.txt",
            r#"fastapi
uvicorn
pydantic
"#,
        ),
        (
            "README.md",
            r#"# Car Brands API

A simple CRUD API for managing car brands built with FastAPI.

## Features

- Create, Read, Update, Delete car brands
- In-memory database
- Pydantic model validation
- Automatic API documentation

## Setup

```bash
cd your_project_folder
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Usage

### Start the server
```bash
python app.py
# or
uvicorn app:app --reload
```

### API Endpoints

- **GET /** - Welcome message
- **GET /brands/** - List all car brands
- **GET /brands/{id}** - Get specific car brand
- **POST /brands/** - Create new car brand
- **PUT /brands/{id}** - Update car brand
- **DELETE /brands/{id}** - Delete car brand

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example Usage

```bash
# Get all brands
curl http://localhost:8000/brands/

# Create a new brand
curl -X POST "http://localhost:8000/brands/" \
     -H "Content-Type: application/json" \
     -d '{"name": "Tesla", "country": "USA", "founded_year": 2003}'
```"#,
        ),
    ]
}

pub fn get_ml_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "train.py",
            r#"import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample ML model
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = LinearRegression()
model.fit(X, y)

print("Model trained!")"#,
        ),
        (
            "predict.py",
            r#"from train import model

# Predict example
prediction = model.predict([[4]])
print(f"Prediction: {prediction[0]}")"#,
        ),
        ("requirements.txt", "numpy\npandas\nscikit-learn\n"),
        (
            "README.md",
            "# KR ML Project\n\nGenerated using KR - Kompiler Ready",
        ),
    ]
}

pub fn get_dl_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "train.py",
            r#"import torch
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

print("DL model trained!")"#,
        ),
        ("requirements.txt", "torch\n"),
        (
            "README.md",
            "# KR DL Project\n\nGenerated using KR - Kompiler Ready",
        ),
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
        (
            "README.md",
            "# KR Streamlit + Docling + LangChain\n\nGenerated using KR - Kompiler Ready",
        ),
    ]
}

pub fn get_modular_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        // core/calculator.py
        (
            "core/calculator.py",
            r#"def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)"#,
        ),
        // core/__init__.py
        ("core/__init__.py", r#""#),
        // utils/logger.py
        (
            "utils/logger.py",
            r#"import logging

logging.basicConfig(level=logging.INFO)

def log_info(message):
    logging.info(message)"#,
        ),
        // utils/__init__.py
        ("utils/__init__.py", r#""#),
        // services/external_api.py
        (
            "services/external_api.py",
            r#"def fetch_data(url):
    # Simulate fetching from an external service
    return {"data": f"Response from {url}"}"#,
        ),
        // services/__init__.py
        ("services/__init__.py", r#""#),
        // models/data_model.py
        (
            "models/data_model.py",
            r#"class User:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name

    def __repr__(self):
        return f"<User {self.name}>""#,
        ),
        // models/__init__.py
        ("models/__init__.py", r#""#),
        // api/main.py (Flask example)
        (
            "api/main.py",
            r#"from flask import Flask, jsonify
from core.calculator import calculate_average

app = Flask(__name__)

@app.route('/average', methods=['GET'])
def average():
    nums = [10, 20, 30]
    avg = calculate_average(nums)
    return jsonify({"average": avg})"#,
        ),
        // api/__init__.py
        ("api/__init__.py", r#""#),
        // config/settings.py
        (
            "config/settings.py",
            r#"APP_NAME = "Modular App"
DEBUG = True
DATABASE_URL = "sqlite:///./test.db""#,
        ),
        // config/__init__.py
        ("config/__init__.py", r#""#),
        // main.py
        (
            "main.py",
            r#"from flask import Flask
from api.main import app as api_app

if __name__ == "__main__":
    api_app.run(debug=True)"#,
        ),
        // requirements.txt
        (
            "requirements.txt",
            r#"flask
pydantic
sqlalchemy
python-dotenv
typing_extensions"#,
        ),
        // README.md
        (
            "README.md",
            r#"# KR Modular Python Project

This is a modular Python project generated using KR - Kompiler Ready.

## Features

- Modular structure for easy maintenance
- Example modules: calculator, logger, model, external service
- Flask-based API endpoint for demonstration
- Configurable settings

## Setup

```bash
cd your_project_folder
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python main.py
Visit http://localhost:5000/average to test the API."#,
        ),
    ]
}
pub fn get_django_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "manage.py",
            r#"#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line

        execute_from_command_line(sys.argv)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)"#,
        ),
        (
            "config/settings.py",
            r#"import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent   
SECRET_KEY = 'your-secret-key'
DEBUG = True
ALLOWED_HOSTS = []
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app',  # Your app name
]
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
ROOT_URLCONF = 'config.urls'
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True
STATIC_URL = '/static/'
"#,
        ),
        (
            "config/urls.py",
            r#"from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]"#,
        ),
        (
            "app/models.py",
            r#"from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __repr__(self):
        return f"<User {self.username}>"#,
        ),
        (
            "app/views.py",
            r#"from django.http import JsonResponse

def user_average_view(request):
    users = User.objects.all()
    average_age = sum(user.age for user in users) / len(users) if users else 0
    return JsonResponse({"average_age": average_age})"#,
        ),
        (
            "app/admin.py",
            r#"from django.contrib import admin
from .models import User

admin.site.register(User)"#,
        ),
        (
            "app/apps.py",
            r#"from django.apps import AppConfig

class UserConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'"#,
        ),
        ("app/migrations/__init__.py", r#""#),
        (
            "app/tests.py",
            r#"from django.test import TestCase

class UserModelTest(TestCase):
    def setUp(self):
        User.objects.create(username="testuser", email="test@example.com")

    def test_user_creation(self):
        user = User.objects.get(username="testuser")
        self.assertEqual(user.email, "test@example.com")"#,
        ),
        (
            "app/urls.py",
            r#"from django.urls import path

urlpatterns = [
    path('users/average/', user_average_view, name='user_average'),
]"#,
        ),
        (
            "requirements.txt",
            r#"Django>=3.2,<4.0 
# Add any additional dependencies here
"#,
        ),
        (
            "README.md",
            r#"# KR Django Project
This is a Django project generated by KR - Python Project Manager.
"#,
        ),
    ]
}
pub fn get_streamlit_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "app.py",
            r#"import streamlit as st

st.title("Hello, Streamlit!")
st.write("Welcome to your Streamlit app.")"#,
        ),
        (
            "requirements.txt",
            r#"streamlit
            "#,
        ),
        (
            "README.md",
            r#"# KR Streamlit Project
This is a Streamlit project generated by KR - Python Project Manager.
            "#,
        ),
    ]
}

pub fn get_microservices_structure() -> Vec<(&'static str, &'static str)> {
    vec![
        // Shared module
        (
            "services/shared/utils.py",
            r#"def log_info(message):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(message)"#,
        ),
        ("services/shared/__init__.py", r#""#),
        // User Service
        (
            "services/user-service/app/models.py",
            r#"class User:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name

    def to_dict(self):
        return {"id": self.user_id, "name": self.name}"#,
        ),
        (
            "services/user-service/app/routes.py",
            r#"from flask import Blueprint, jsonify
from .models import User

user_bp = Blueprint('user', __name__)

USERS = {1: User(1, "Alice"), 2: User(2, "Bob")}

@user_bp.route('/users')
def list_users():
    return jsonify([user.to_dict() for user in USERS.values()])

@user_bp.route('/users/<int:user_id>')
def get_user(user_id):
    user = USERS.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())"#,
        ),
        ("services/user-service/app/__init__.py", r#""#),
        (
            "services/user-service/main.py",
            r#"from flask import Flask
from app.routes import user_bp

app = Flask(__name__)
app.register_blueprint(user_bp)

if __name__ == "__main__":
    app.run(port=5001)"#,
        ),
        (
            "services/user-service/requirements.txt",
            r#"flask
pydantic
sqlalchemy"#,
        ),
        // Order Service
        (
            "services/order-service/app/models.py",
            r#"class Order:
    def __init__(self, order_id, product):
        self.order_id = order_id
        self.product = product

    def to_dict(self):
        return {"id": self.order_id, "product": self.product}"#,
        ),
        (
            "services/order-service/app/routes.py",
            r#"from flask import Blueprint, jsonify
from .models import Order

order_bp = Blueprint('order', __name__)

ORDERS = {1: Order(1, "Laptop"), 2: Order(2, "Phone")}

@order_bp.route('/orders')
def list_orders():
    return jsonify([order.to_dict() for order in ORDERS.values()])

@order_bp.route('/orders/<int:order_id>')
def get_order(order_id):
    order = ORDERS.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404
    return jsonify(order.to_dict())"#,
        ),
        ("services/order-service/app/__init__.py", r#""#),
        (
            "services/order-service/main.py",
            r#"from flask import Flask
from app.routes import order_bp

app = Flask(__name__)
app.register_blueprint(order_bp)

if __name__ == "__main__":
    app.run(port=5002)"#,
        ),
        (
            "services/order-service/requirements.txt",
            r#"flask
pydantic
sqlalchemy"#,
        ),
        // Gateway Service
        (
            "gateway/app/config.py",
            r#"SERVICE_URLS = {
    'user': 'http://localhost:5001',
    'order': 'http://localhost:5002'
}"#,
        ),
        (
            "gateway/app/routes.py",
            r#"from flask import Blueprint, jsonify
import requests

gateway_bp = Blueprint('gateway', __name__)

@gateway_bp.route('/all')
def get_all_data():
    from app.config import SERVICE_URLS
    data = {}
    for name, url in SERVICE_URLS.items():
        try:
            response = requests.get(f"{url}/{name}s")
            if response.status_code == 200:
                data[name] = response.json()
        except Exception as e:
            data[name] = {"error": str(e)}
    return jsonify(data)"#,
        ),
        ("gateway/app/__init__.py", r#""#),
        (
            "gateway/main.py",
            r#"from flask import Flask
from app.routes import gateway_bp

app = Flask(__name__)
app.register_blueprint(gateway_bp)

if __name__ == "__main__":
    app.run(port=5000)"#,
        ),
        // Config
        (
            "config/settings.py",
            r#"APP_NAME = "Microservices App"
DEBUG = True
DATABASE_URL = "sqlite:///./test.db""#,
        ),
        ("config/__init__.py", r#""#),
        // README
        (
            "README.md",
            r#"# KR - Kompiler Ready: Microservices Project

This is a microservices-based Python project generated using KR - Kompiler Ready.

## Features

- Modular microservices: user-service, order-service
- Centralized gateway
- Shared utilities
- Flask-based APIs

## Setup

```bash
# Install dependencies for each service
cd services/user-service && pip install -r requirements.txt
cd ../order-service && pip install -r requirements.txt
cd ../..

# Run services in separate terminals
python services/user-service/main.py  # Port 5001
python services/order-service/main.py  # Port 5002
python gateway/main.py                 # Port 5000

# Access combined API
curl http://localhost:5000/all
Visit http://localhost:5000/all to test the gateway."#,
        ),
        // Optional: Run All Script (Linux/macOS only)
        (
            "run_all.sh",
            r#"#!/bin/bash
echo "Starting User Service..."
(cd services/user-service && python main.py) &

echo "Starting Order Service..."
(cd services/order-service && python main.py) &

echo "Starting Gateway..."
(cd gateway && python main.py) &

wait"#,
        ),
    ]
}
