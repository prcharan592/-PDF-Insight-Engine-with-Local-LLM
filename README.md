📄 PDF Insight Engine with Local LLM

A Streamlit-based AI application that enables users to upload PDFs, process their content, and interact with them through conversational AI powered by locally running LLMs (Ollama). This project eliminates the need for an internet connection while ensuring efficient document analysis.

# 🚀 Features
✅ Upload and process multiple PDFs
✅ Conversational Q&A using local LLMs (Ollama)
✅ Embeddings & Vector Search with FAISS and HuggingFace
✅ Chunking & Retrieval using LangChain
✅ Secure & Offline document analysis

# 🛠️ Tech Stack
	•	Python 3.9+
	•	Streamlit (for the web interface)
	•	LangChain (for document processing & retrieval)
	•	FAISS (for efficient vector storage & retrieval)
	•	HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
	•	Ollama (for running local LLMs)

# Model	Used
Mistral 7B	4.1 GB	
Llama3 8B	4.7 GB	
Llama3.2 3B	2.0 GB
DeepSeek-R1 7B	4.7 GB

# 🏗️ Installation & Setup

# 1️⃣ Install Dependencies

# Clone the repository:
git clone https://github.com/yourusername/PDF-Insight-Engine.git
cd PDF-Insight-Engine

pip install -r requirements.txt


# 2️⃣ Install & Run Ollama

Ollama is required to run the local LLM. Download and install it from:
🔗 https://ollama.com

# Then, pull the required models:
ollama pull mistral
ollama pull llama3
ollama pull llama3:8b
ollama pull deepseek-r1


# ▶️ Running the Application

Start the Streamlit app:
streamlit run app.py


# 🏗️ How It Works

1️⃣ Upload PDFs
2️⃣ Extract & Process Content using LangChain
3️⃣ Convert text into embeddings (HuggingFace)
4️⃣ Store & Retrieve vectors using FAISS
5️⃣ Interact with PDFs through local LLM (Ollama)


# ⚡ Example Usage

Upload a PDF 📄 → Ask “Summarize this document.” → Get AI-powered response in seconds!
