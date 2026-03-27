import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 1. Carga manual del modelo y tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def obtener_respuesta(pregunta, contexto):
    # Preprocesar la entrada
    inputs = tokenizer(pregunta, contexto, return_tensors="pt")
    
    # Obtener las puntuaciones del modelo
    with torch.no_grad():
        outputs = model(**inputs)
    
    # El modelo nos da las probabilidades de donde empieza y termina la respuesta
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Buscamos los índices con la puntuación más alta
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    
    # Convertimos esos números (IDs) de vuelta a texto
    respuesta = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return respuesta

# --- DATOS DE PRUEBA ---
contexto = """
ShadowRoot07 is a developer who works exclusively from a mobile device using Termux. 
He uses NeoVim as his primary text editor and GitHub Actions for heavy computing tasks.
"""

preguntas = [
    "What is ShadowRoot07's primary text editor?",
    "What device does he use for work?"
]

print("\n--- SISTEMA DE RESPUESTAS MANUAL (QA) ---")
for p in preguntas:
    res = obtener_respuesta(p, contexto)
    print(f"\nPregunta: {p}")
    print(f"Respuesta: {res}")

