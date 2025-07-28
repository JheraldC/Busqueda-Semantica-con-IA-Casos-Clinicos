from transformers import AutoTokenizer, AutoModelForCausalLM

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")

print("Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")

print("Modelo cargado correctamente")
