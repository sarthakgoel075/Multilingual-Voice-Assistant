# 🌐 Multilingual Voice Assistant

This project is a **Multilingual Voice Assistant** built to support **Indian regional languages** like **Hindi, Tamil, Bengali, Punjabi, and English**. It performs speech-to-text (ASR), generates smart responses using a local LLM and RAG (Retrieval Augmented Generation), and converts the output back to speech (TTS).

---

## 📌 Key Features

- 🎤 **Multilingual Speech Input** (e.g., Hindi, Tamil)
- 🧠 **Smart Response Generation** (LLM + SQL/PDF-based RAG)
- 🔊 **Text-to-Speech in Multiple Languages**
- 📦 Modular and expandable pipeline

---

## 🛠 Technologies Used

| Component          | Technology Used              | Reason                                                  |
|--------------------|------------------------------|---------------------------------------------------------|
| Speech-to-Text     | [Whisper (OpenAI)](https://github.com/openai/whisper) or Citrinet (NVIDIA NeMo) | High-accuracy transcription, multilingual support       |
| Text Generation    | Local LLM + RAG              | Fast responses, contextual and document-aware answers   |
| Text-to-Speech     | [AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS) | Support for Indian languages, low latency                |


---


---

## 🚀 How It Works

1. 🎙 **User speaks** in a supported Indian language.
2. 🧠 **ASR (e.g., Whisper)** converts audio to text.
3. 📚 **RAG module** retrieves relevant answer from SQL database or PDF/text.
4. 🗣 **TTS (Indic-TTS)** speaks the answer in the same language.
5. 🔁 The pipeline is continuous and low-latency.

---

## 🧪 Setup & Installation

### Prerequisites
- Python 3.8+
- PyTorch (with GPU support recommended)
- ffmpeg installed and added to PATH

### Installation
```bash
git clone https://github.com/sarthakgoel075/Multilingual-Voice-Assistant.git
cd Multilingual-Voice-Assistant
Install Whisper:
pip install git+https://github.com/openai/whisper.git
Install TTS (AI4Bharat):
pip install TTS
````
💡 Future Improvements
✅ Add Citrinet ASR fine-tuned for Hindi (NeMo)

🌐 Deploy backend using FastAPI + WebSockets

🔀 Handle code-switching (e.g., Hinglish) more robustly

🧾 Summarize scraped FAQ content via LLM

📱 React Native app version for mobile voice assistant

👨‍💻 Author
Sarthak Goel
Internship Project | LG Voice Assistant + PowerBI Dashboard
With integration of Whisper / Citrinet, RAG, and Indic-TTS




