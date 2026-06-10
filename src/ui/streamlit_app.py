import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as strl
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.rag import ingest_and_index
from src.pipeline.routing import run_orchestrator
from src.pipeline.cache import check_cache, save_to_cache

strl.set_page_config(page_title="Linux Technical Assistant", page_icon="🐧")
strl.title("🐧 Assistente Especialista: The Linux Command Line")
strl.markdown("Respostas com citações exatas de páginas técnicas direto da fonte.")

with strl.sidebar:
    strl.header("Gestão do Corpus")
    if strl.button("🚀 Ingerir e Indexar Livro Técnico"):
        with strl.spinner("Extraindo e processando PDF..."):
            ingest_and_index()
            strl.success("Concluído com sucesso!")

if "messages" not in strl.session_state:
    strl.session_state.messages = []

for msg in strl.session_state.messages:
    with strl.chat_message(msg["role"]):
        strl.markdown(msg["content"])

if user_prompt := strl.chat_input("Pergunte algo sobre o terminal Linux..."):
    strl.session_state.messages.append({"role": "user", "content": user_prompt})
    with strl.chat_message("user"):
        strl.markdown(user_prompt)
    
    with strl.chat_message("assistant"):
        resposta_cache = check_cache(user_prompt)
        
        if resposta_cache:
            strl.markdown(f"*(Resposta do Cache)*\n\n{resposta_cache}")
            strl.session_state.messages.append({"role": "assistant", "content": resposta_cache})
        else:
            with strl.spinner("Analisando documentos..."):
                stream = run_orchestrator(user_prompt)
                
            def stream_generator():
                texto_completo = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        pedaco = chunk.choices[0].delta.content
                        texto_completo += pedaco
                        yield pedaco
                save_to_cache(user_prompt, texto_completo)

            resposta_final = strl.write_stream(stream_generator)
            strl.session_state.messages.append({"role": "assistant", "content": resposta_final})