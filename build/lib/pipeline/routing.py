# import os
# import json
# from openai import OpenAI
# from src.pipeline.rag import retrieve
# from src.pipeline.tools import lookup_chapter, tool_schema

# def run_orchestrator(user_query: str):
#     api_key = os.getenv("GEMINI_API_KEY")
#     client = OpenAI(
#         api_key=api_key,
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     )
    
#     messages = [{"role": "user", "content": user_query}]
#     response = client.chat.completions.create(
#         model="gemini-2.5-flash-lite",
#         messages=messages,
#         tools=[tool_schema],
#         tool_choice="auto",
#         temperature=0.0
#     )
    
#     response_message = response.choices[0].message
    
#     if response_message.tool_calls:
#         for tool_call in response_message.tool_calls:
#             if tool_call.function.name == "lookup_chapter":
#                 args = json.loads(tool_call.function.arguments)
#                 tool_result = lookup_chapter(chapter_number=args.get("chapter_number"))
                
#                 messages.append(response_message)
#                 messages.append({
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "name": "lookup_chapter",
#                     "content": tool_result
#                 })
                
#                 final_response = client.chat.completions.create(
#                     model="gemini-2.5-flash-lite",
#                     messages=messages,
#                     temperature=0.0,
#                     stream=True
#                 )
#                 return final_response

#     hits = retrieve(user_query, k=4)
#     contexto = "\n\n---\n\n".join([f"[{h['source']}:p{h['page']}]\n{h['text']}" for h in hits])
    
#     RAG_PROMPT = f"""Você é um assistente técnico especialista no livro 'The Linux Command Line'.
# Responda à pergunta do usuário utilizando APENAS o contexto abaixo fornecido. 
# Se a informação necessária não constar explicitamente no contexto, responda estritamente 'Não encontrado no corpus'.
# Sempre cite obrigatoriamente a página do arquivo original usando o formato [arquivo:pagina].

# CONTEXTO:
# {contexto}

# PERGUNTA: {user_query}

# RESPOSTA:"""

#     final_response = client.chat.completions.create(
#         model="gemini-3.1-flash-lite",
#         messages=[{"role": "user", "content": RAG_PROMPT}],
#         temperature=0.0,
#         stream=True
#     )
#     return final_response

import os
import json
from openai import OpenAI
from src.pipeline.rag import retrieve
from src.pipeline.tools import lookup_chapter, tool_schema

def run_orchestrator(user_query: str):
    api_key = os.getenv("GEMINI_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    
    # SYSTEM PROMPT para alinhar o modelo logo na entrada
    SYSTEM_INSTRUCTION = (
        "Você é um assistente especialista OBRIGATORIAMENTE focado no livro 'The Linux Command Line'. "
        "Você deve SEMPRE responder citando a fonte no formato [arquivo:pagina]. "
        "Nunca diga que não sabe qual é o livro."
    )

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_query}
    ]
    
    # 1. Tenta identificar se a pergunta pede a estrutura de um capítulo (Tool)
    response = client.chat.completions.create(
        model="gemini-2.5-flash-lite",
        messages=messages,
        tools=[tool_schema],
        tool_choice="auto",
        temperature=0.0
    )
    
    response_message = response.choices[0].message
    
    # SE O MODELO DECIDIR USAR A FERRAMENTA (Ex: Perguntas diretas sobre "Capítulo X")
    if response_message.tool_calls:
        print("\n🚀 [ORQUESTRADOR] >>> Usando RAG + TOOL (Sumário Dinâmico) <<< 🚀")
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "lookup_chapter":
                args = json.loads(tool_call.function.arguments)
                tool_result = lookup_chapter(chapter_number=args.get("chapter_number"))
                
                messages.append(response_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "lookup_chapter",
                    "content": tool_result
                })
                
                # Força a citação mesmo vindo da ferramenta
                messages.append({
                    "role": "system", 
                    "content": "Formate a resposta final de forma clara e garanta que a citação de página/fonte trazida pela ferramenta seja exibida no formato [arquivo:pagina]."
                })
                
                final_response = client.chat.completions.create(
                    model="gemini-2.5-flash-lite",
                    messages=messages,
                    temperature=0.0,
                    stream=True
                )
                return final_response
            
    print("\n🔍 [ORQUESTRADOR] >>> Usando PURE RAG (Busca Vetorial no PDF) <<< 🔍")
    # SE NÃO USAR A FERRAMENTA, VAI PARA O RAG SEGURO (Busca no PDF)
    hits = retrieve(user_query, k=10)
    
    # Se o RAG falhar em achar documentos relevantes no banco
    if not hits:
        contexto = "Nenhum trecho encontrado no banco de dados para esta busca."
    else:
        # contexto = "\n\n---\n\n".join([f"[{h['source']}:p{h['page']}]\n{h['text']}" for h in hits])
        # Altere a linha do contexto no seu routing.py para:
        contexto = "\n\n---\n\n".join([f"[{h['source']}:p{h['page']}] {h['text']}" for h in hits])
        
#     RAG_PROMPT = f"""Você é um assistente técnico especialista no livro 'The Linux Command Line'.
# Responda à pergunta do usuário utilizando APENAS o contexto abaixo fornecido. 
# Se a informação necessária não constar explicitamente no contexto, responda estritamente 'Não encontrado no corpus'.

# Você DEVE incluir a citação exata da página no formato [arquivo:pagina] exatamente como consta na tag de cabeçalho de cada trecho do contexto.

# CONTEXTO:
# {contexto}

# PERGUNTA: {user_query}

# RESPOSTA:"""
    
    RAG_PROMPT = f"""Você é um assistente técnico especialista no livro 'The Linux Command Line'.
Responda à pergunta do usuário utilizando APENAS o contexto abaixo fornecido. 
Se a informação necessária não constar explicitamente no contexto, responda estritamente 'Não encontrado no corpus'.

ATENÇÃO: Você DEVE incluir a citação completa da fonte no final da resposta exatamente no formato [arquivo:pÁGINA] (exemplo: [TLCL-25.12.pdf:p27]), extraindo-a do cabeçalho de cada trecho do contexto fornecido. Não omita o número da página.

CONTEXTO:
{contexto}

PERGUNTA: {user_query}

RESPOSTA:"""

    final_response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": RAG_PROMPT}],
        temperature=0.0,
        stream=True
    )
    return final_response