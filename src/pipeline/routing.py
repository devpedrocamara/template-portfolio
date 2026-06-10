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
    
    messages = [{"role": "user", "content": user_query}]
    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=messages,
        tools=[tool_schema],
        tool_choice="auto",
        temperature=0.0
    )
    
    response_message = response.choices[0].message
    
    if response_message.tool_calls:
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
                
                final_response = client.chat.completions.create(
                    model="gemini-3.1-flash-lite",
                    messages=messages,
                    temperature=0.0,
                    stream=True
                )
                return final_response

    hits = retrieve(user_query, k=4)
    contexto = "\n\n---\n\n".join([f"[{h['source']}:p{h['page']}]\n{h['text']}" for h in hits])
    
    RAG_PROMPT = f"""Você é um assistente técnico especialista no livro 'The Linux Command Line'.
Responda à pergunta do usuário utilizando APENAS o contexto abaixo fornecido. 
Se a informação necessária não constar explicitamente no contexto, responda estritamente 'Não encontrado no corpus'.
Sempre cite obrigatoriamente a página do arquivo original usando o formato [arquivo:pagina].

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