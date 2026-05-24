# 🚀 Advanced RAG Pipeline with RAGAS Evaluation

A production-ready PDF-based Retrieval-Augmented Generation (RAG) system built using **Gemini**, **LangChain**, **FAISS**, and **RAGAS** for hallucination detection and retrieval evaluation.

This project focuses not only on document question-answering, but also on evaluating the quality of generated responses using modern RAG evaluation metrics.

---

# ✨ Features

## 📄 PDF Chatbot

* Upload multiple PDF documents
* Ask questions in natural language
* Context-grounded responses using Retrieval-Augmented Generation (RAG)

---

## 🧠 Advanced Retrieval Pipeline

* FAISS Vector Database
* Persistent embeddings stored on disk
* Auto-loading saved indexes after restart
* MMR (Max Marginal Relevance) retrieval
* Content-based hashing for index consistency

---

## 📊 RAGAS Evaluation Layer

Evaluate generated answers using modern RAG evaluation metrics:

| Metric            | Purpose                                  |
| ----------------- | ---------------------------------------- |
| Faithfulness      | Detect hallucinations / grounding        |
| Answer Relevancy  | Measure answer alignment with user query |
| Context Precision | Evaluate retrieval quality               |

---

## 🚨 Hallucination Detection

The system automatically flags potentially hallucinated responses based on Faithfulness scores.

---

## 🔍 Context Inspection

View retrieved chunks and source documents used to generate the answer.

---

## 🏗️ System Architecture

![RAG Architecture](https://github.com/AbdulSamad200/PDF-RAG-Evaluation-System/blob/main/Architecture.png?raw=true)


---

# 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini
* FAISS
* RAGAS
* Pandas
* PyPDF

---

# 📂 Project Structure

```text
├── app.py
├── requirements.txt
├── .env
├── faiss_indexes/
├── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/Advanced-RAG-Pipeline-with-RAGAS.git

cd Advanced-RAG-Pipeline-with-RAGAS
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 📊 Example Evaluation Output

| Metric            | Score |
| ----------------- | ----- |
| Faithfulness      | 0.88  |
| Answer Relevancy  | 0.81  |
| Context Precision | 0.32  |

---

# 📌 Key Learnings

This project demonstrates that production-grade RAG systems require more than just generation.

Important aspects include:

* Hallucination detection
* Retrieval quality analysis
* Context grounding
* Evaluation-driven development
* Retrieval optimization

---

# 🔜 Future Improvements

* Cross-Encoder Reranking
* Hybrid Search (BM25 + Vector Search)
* Conversation Memory
* Multi-query Retrieval
* Async RAGAS Evaluation
* Advanced Observability

---

# 📸 Demo

(Add screenshots/GIFs here)

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork the project and submit pull requests.


---

# 👨‍💻 Author

Abdul Samad Khaja

AI Engineer | Generative AI | RAG Systems | Agentic AI
