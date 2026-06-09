import streamlit as strl
from dotenv import load_dotenv
# Garante o carregamento das chaves de api salvas no .env local
load_dotenv()

from src.pipeline.rag import ingest_and_index
from src.pipeline.routing import run_orchestrator

strl.set_page_config(page_title="Linux Technical Assistant", page_icon="🐧")
strl.title("🐧 Assistente Especialista: The Linux Command Line")
strl.markdown("Respostas com citações exatas de páginas técnicas direto da fonte.")

# Botão utilitário na barra lateral para realizar o primeiro processamento do livro
with strl.sidebar:
    strl.header("Gestão do Corpus")
    if strl.button("🚀 Ingerir e Indexar Livro Técnico"):
        with strl.spinner("Extraindo e processando PDF..."):
            ingest_and_index()
            strl.success("Concluído com sucesso!")

# Inicialização do estado de memória do chat do Streamlit
if "messages" not in strl.session_state:
    strl.session_state.messages = []

# Exibe o histórico de conversas antigas
for msg in strl.session_state.messages:
    with strl.chat_message(msg["role"]):
        strl.markdown(msg["content"])

# Captura de novas entradas do usuário
if user_prompt := strl.chat_input("Pergunte algo sobre o terminal Linux ou peça sumários de capítulos..."):
    with strl.chat_message("user"):
        strl.markdown(user_prompt)
    strl.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Processamento através do Orquestrador Unificado
    with strl.chat_message("assistant"):
        with strl.spinner("Analisando documentos..."):
            resposta = run_orchestrator(user_prompt)
            strl.markdown(resposta)
            
    strl.session_state.messages.append({"role": "assistant", "content": resposta})