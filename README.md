# Corrective-Rag-Offensive-Qwen
- Implemented Local Corrective-Rag with Offensive-Qwen2.5-Coder-7B using paper Corrective Retrieval Augmented Generation [https://arxiv.org/abs/2401.15884]
- This project is part of and the first version of Offensive-GenAI system
## Offensive-Engine
- Offensive-Engine is a collection of open-source fine-tuned models used for code reviews, static analysis and pentesting in general.
- Here is my huggingface hub: huggingface.co/doss1232
## Technology used
- Langchain
- Streamlit
- Tavily
- Chromadb
- Ollama

## Guide to using
1. Clone the repo
2. Go into requirements folder
3. use this command: pip install -r requirements.txt
4. Meanwhile, install Ollama: https://ollama.com
5. Run this command: ollama start
6. pull these 2 models using ollama
   + ollama pull Offensive-Qwen:latest
   + ollama pull nomic-embed-text:latest
7. Set up LangSmith api key (optional):
![image](https://github.com/user-attachments/assets/de756f1d-d089-4fac-bd11-341509f98d5c)

8. In the project repo, run: streamlit run app.py
9. This will open interface in the browser
