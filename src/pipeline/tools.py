from pydantic import BaseModel, Field

CHAPTERS_MAP = {
    "1": "What Is The Shell? - Introdução ao prompt, bash e comandos básicos de navegação inicial.",
    "2": "Navigation - Comandos essenciais para navegar em diretórios: pwd, cd, ls.",
    "3": "Exploring The System - Investigando o sistema operacional com menos, file e o mapeamento Linux.",
    "4": "Manipulating Files And Directories - Criação e destruição com cp, mv, mkdir, rm.",
    "5": "Working With Commands - Diferenciação de tipos usando type, which, help e man."
}

class LookupChapterInput(BaseModel):
    chapter_number: str = Field(description="O número do capítulo em string (ex: '1', '2', '3') para buscar o resumo descritivo.")

def lookup_chapter(chapter_number: str) -> str:
    chapter_clean = str(chapter_number).strip()
    if chapter_clean in CHAPTERS_MAP:
        return f"Capítulo {chapter_clean}: {CHAPTERS_MAP[chapter_clean]}"
    return f"Erro: Capítulo {chapter_clean} não mapeado no sumário dinâmico. Escolha de 1 a 5."

tool_schema = {
    "type": "function",
    "function": {
        "name": "lookup_chapter",
        "description": "Busca informações e escopos de cobertura de um capítulo específico do livro técnico por número.",
        "parameters": {
            "type": "object",
            "properties": {
                "chapter_number": {
                    "type": "string",
                    "description": "Número do capítulo desejado (ex: '1', '2', etc.)"
                }
            },
            "required": ["chapter_number"]
        }
    }
}