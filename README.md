# 🎓 College RAG Chatbot

An AI-powered chatbot that answers questions based on custom college data using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Demo

> Ask questions like:

* What courses are offered?
* What is the fee structure?
* Which companies visit for placements?

---

## 🧠 How It Works

1. 📄 Input data (PDF/Text) is loaded
2. ✂️ Split into chunks
3. 🔢 Converted into embeddings
4. 📦 Stored in FAISS vector database
5. ❓ User query → retrieves relevant chunks
6. 🤖 LLM generates answer based on context

---

## 🧱 Tech Stack

* ⚙️ **Backend:** FastAPI
* 🧠 **Embeddings:** HuggingFace (MiniLM)
* 📦 **Vector DB:** FAISS
* 🤖 **LLM:** FLAN-T5
* 💬 **Frontend:** HTML, CSS, JavaScript

---

## 📁 Project Structure

```
college-rag-chatbot/
│
├── src/
│   ├── ingest.py
│
├── data/
│   └── college.txt
│
├── index.html
├── main.py
├── requirements.txt
```

---

## ▶️ Run Locally

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Create vector database

```
python src/ingest.py
```

### 3️⃣ Start backend

```
uvicorn main:app --reload
```

### 4️⃣ Run frontend

Open `index.html` using Live Server

---

## 🧪 Example Queries

* What courses are offered?
* What is the fee for engineering?
* Which companies visit the campus?
* Tell me about facilities

---

## 🔥 Features

* ✅ Context-aware answers
* ✅ Fast retrieval using FAISS
* ✅ ChatGPT-style UI
* ✅ Works with custom data

---

## 🚧 Future Improvements

* 🔄 React frontend
* 🌍 Deployment (Render + Vercel)
* ⚡ Streaming responses
* 🧠 Better LLM (GPT / Llama)

---

## 💼 Use Case

This project demonstrates how to build a **full-stack AI application** using RAG architecture for domain-specific question answering.

---

## 👨‍💻 Author

Built by Vajra Teja as a hands-on project to learn AI systems, FastAPI, and full-stack development.

