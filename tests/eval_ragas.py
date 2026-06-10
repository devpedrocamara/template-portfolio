import os
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

from dotenv import load_dotenv
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
os.environ["CHROMA_OPENAI_API_KEY"] = gemini_key or ""
os.environ["OPENAI_API_KEY"] = gemini_key or ""

from datasets import Dataset
from ragas import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.pipeline.rag import retrieve
from src.pipeline.routing import run_orchestrator

def run_evaluation():
    print("Iniciando avaliação RAGAS...")
    
    perguntas = [
        "O que é o shell?",
        "Para que serve o comando pwd?"
    ]
    respostas_esperadas = [
        "O shell é um programa que recebe comandos do teclado e os passa para o sistema operacional executar.",
        "O comando pwd (print working directory) serve para mostrar o diretório atual em que você está trabalhando."
    ]
    
    respostas_geradas = []
    contextos_recuperados = []

    for pergunta in perguntas:
        print(f"Avaliando pergunta: {pergunta}")
        
        hits = retrieve(pergunta, k=4)
        contextos_recuperados.append([h["text"] for h in hits])
        
        stream = run_orchestrator(pergunta)
        resposta_completa = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                resposta_completa += chunk.choices[0].delta.content
        respostas_geradas.append(resposta_completa)

    dataset = Dataset.from_dict({
        "question": perguntas,
        "answer": respostas_geradas,
        "contexts": contextos_recuperados,
        "ground_truth": respostas_esperadas
    })

    llm_ragas = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=gemini_key)
    embeddings_ragas = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_key)

    print("Calculando métricas...")
    resultados = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm_ragas,
        embeddings=embeddings_ragas
    )

    print("\n=== RESULTADOS DA AVALIAÇÃO ===")
    print(resultados)
    
    df = resultados.to_pandas()
    df.to_csv("ragas_results.csv", index=False)
    print("\nResultados detalhados salvos em 'ragas_results.csv'.")

if __name__ == "__main__":
    run_evaluation()