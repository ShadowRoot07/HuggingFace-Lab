from transformers import pipeline

# 1. Instanciamos el pipeline de análisis de sentimiento
# Esto descarga automáticamente un modelo pre-entrenado (aprox 260MB)
classifier = pipeline("sentiment-analysis")

# 2. Lista de frases para analizar
frases = [
    "I love learning AI with ShadowRoot07, it's amazing!",
    "The weather is terrible today, I'm very sad.",
    "This is just a neutral sentence."
]

# 3. Ejecutar la IA
print("--- Iniciando Análisis de Sentimiento ---")
resultados = classifier(frases)

for frase, res in zip(frases, resultados):
    print(f"\nFrase: {frase}")
    print(f"Resultado: {res['label']} | Score: {res['score']:.4f}")

