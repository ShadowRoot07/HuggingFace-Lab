from transformers import AutoTokenizer

# Usamos el tokenizer de un modelo famoso: BERT
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

frase = "I am ShadowRoot07 and I love Neovim."

# 1. Convertir texto a tokens (palabras cortadas)
tokens = tokenizer.tokenize(frase)
print(f"\nTokens: {tokens}")

# 2. Convertir tokens a IDs (números que van a la CPU/GPU)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"IDs numéricos: {ids}")

# 3. El proceso completo (lo que hace el modelo internamente)
inputs = tokenizer(frase)
print(f"\nInput completo para la IA: {inputs}")

