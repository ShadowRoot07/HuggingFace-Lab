from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Cargamos el modelo y el tokenizer manualmente
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Creamos el pipeline pasando el modelo y el tokenizer directamente
# Al pasar el objeto 'model', ya no necesitamos el nombre de la tarea
translator = pipeline("translation", model=model, tokenizer=tokenizer)

# 3. Lo mismo para el resumidor (Usamos AutoModel y AutoTokenizer)
sum_name = "sshleifer/distilbart-cnn-12-6"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_name)
summarizer = pipeline("summarization", model=sum_model, tokenizer=sum_tokenizer)

# --- EJECUCIÓN ---
texto_ingles = "Artificial Intelligence is transforming the world. Learning it from a mobile device using Termux is a superpower."
traduccion = translator(texto_ingles)

print("\n--- TRADUCCIÓN ---")
print(f"Original: {texto_ingles}")
print(f"Español: {traduccion[0]['translation_text']}")

# --- RESUMEN ---
texto_largo = """
The Transformer model was first introduced in the paper 'Attention is All You Need' in 2017. 
Before Transformers, sequence modeling mainly relied on recurrent neural networks (RNNs). 
However, Transformers introduced the self-attention mechanism, which allows the model 
to weigh the importance of different words regardless of their distance.
"""
resumen = summarizer(texto_largo, max_length=40, min_length=10, do_sample=False)

print("\n--- RESUMEN ---")
print(f"Texto corto: {resumen[0]['summary_text']}")

