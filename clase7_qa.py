from transformers import pipeline

# 1. Usamos el pipeline de QA (este suele ser más estable con los nombres)
# Si falla, ya sabes que podemos usar el método manual que aprendiste
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# 2. El "Contexto" (La base de conocimientos)
contexto = """
ShadowRoot07 is a developer who works exclusively from a mobile device using Termux. 
He is currently learning Artificial Intelligence and has mastered Scikit-learn and Hugging Face. 
His setup includes NeoVim as his primary text editor and he uses GitHub Actions for heavy computing tasks.
"""

# 3. Las preguntas
preguntas = [
    "What is ShadowRoot07's primary text editor?",
    "What device does he use for work?",
    "Which technologies has he mastered?"
]

print("\n--- SISTEMA DE RESPUESTAS (QA) ---")
for pregunta in preguntas:
    resultado = qa_model(question=pregunta, context=contexto)
    print(f"\nPregunta: {pregunta}")
    print(f"Respuesta: {resultado['answer']} (Confianza: {resultado['score']:.4f})")

