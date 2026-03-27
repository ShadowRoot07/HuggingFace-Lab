from transformers import pipeline

# 1. Pipeline de Traducción (Corregido)
# Cambiamos "translation_en_to_es" por "translation"
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

# 2. Pipeline de Resumen (Este estaba bien, pero lo mantenemos)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- PRUEBA DE TRADUCCIÓN ---
texto_ingles = "Artificial Intelligence is transforming the world. Learning it from a mobile device using Termux is a superpower."
traduccion = translator(texto_ingles)

print("\n--- TRADUCCIÓN ---")
print(f"Original: {texto_ingles}")
# El resultado de 'translation' a veces viene en una lista diferente, así es más seguro:
print(f"Español: {traduccion[0]['translation_text']}")

# --- PRUEBA DE RESUMEN ---
texto_largo = """
The Transformer model was first introduced in the paper 'Attention is All You Need' in 2017. 
Before Transformers, sequence modeling mainly relied on recurrent neural networks (RNNs). 
However, Transformers introduced the self-attention mechanism, which allows the model 
to weigh the importance of different words regardless of their distance.
"""

resumen = summarizer(texto_largo, max_length=40, min_length=10, do_sample=False)

print("\n--- RESUMEN ---")
print(f"Texto corto: {resumen[0]['summary_text']}")

