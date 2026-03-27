from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Función manual para traducir
def traducir_texto(texto):
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Convertir texto a números (Tokens)
    inputs = tokenizer(texto, return_tensors="pt")
    
    # Generar la traducción (el modelo crea los nuevos tokens)
    outputs = model.generate(**inputs)
    
    # Convertir números de vuelta a palabras (Decode)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 2. Función manual para resumir
def resumir_texto(texto):
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(texto, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=40, min_length=10)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- EJECUCIÓN ---
texto_ingles = "Artificial Intelligence is transforming the world. Learning it from a mobile device using Termux is a superpower."
print("\n--- TRADUCCIÓN MANUAL ---")
print(f"Original: {texto_ingles}")
print(f"Español: {traducir_texto(texto_ingles)}")

texto_largo = """
The Transformer model was first introduced in the paper 'Attention is All You Need' in 2017. 
Before Transformers, sequence modeling mainly relied on recurrent neural networks (RNNs). 
However, Transformers introduced the self-attention mechanism, which allows the model 
to weigh the importance of different words regardless of their distance.
"""
print("\n--- RESUMEN MANUAL ---")
print(f"Texto corto: {resumir_texto(texto_largo)}")

