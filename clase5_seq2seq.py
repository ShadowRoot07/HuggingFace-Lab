from transformers import pipeline

# 1. Pipeline de Traducción (Inglés a Español)
# Usamos un modelo específico de Helsinki-NLP
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

# 2. Pipeline de Resumen
# Usamos 'sshleifer/distilbart-cnn-12-6', que es una versión optimizada y rápida
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- PRUEBA DE TRADUCCIÓN ---
texto_ingles = "Artificial Intelligence is transforming the world. Learning it from a mobile device using Termux is a superpower."
traduccion = translator(texto_ingles)

print("\n--- TRADUCCIÓN ---")
print(f"Original: {texto_ingles}")
print(f"Español: {traduccion[0]['translation_text']}")

# --- PRUEBA DE RESUMEN ---
# Un texto largo sobre la historia de los Transformers
texto_largo = """
The Transformer model was first introduced in the paper 'Attention is All You Need' in 2017. 
Before Transformers, sequence modeling mainly relied on recurrent neural networks (RNNs) or 
convolutional neural networks. However, Transformers introduced the self-attention mechanism, 
which allows the model to weigh the importance of different words in a sentence regardless 
of their distance. This led to massive improvements in NLP tasks and paved the way for 
models like BERT, GPT, and Llama.
"""

resumen = summarizer(texto_largo, max_length=40, min_length=10, do_sample=False)

print("\n--- RESUMEN ---")
print(f"Texto corto: {resumen[0]['summary_text']}")

