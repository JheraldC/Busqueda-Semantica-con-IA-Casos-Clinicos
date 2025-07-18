import csv

filename = 'dataset_casos_clinicos_OK.csv'

# 1. Detectar filas con diferente número de columnas
with open(filename, encoding='utf-8') as f:
    reader = csv.reader(f)
    row_lengths = []
    for idx, row in enumerate(reader):
        row_lengths.append((idx+1, len(row)))
        if len(row) != row_lengths[0][1]:
            print(f'Fila {idx+1} tiene {len(row)} columnas (esperado {row_lengths[0][1]})')

# 2. Revisar comas fuera de comillas
with open(filename, encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        # Cuenta comillas dobles
        quote_count = line.count('"')
        # Si es impar, probablemente hay un problema
        if quote_count % 2 != 0:
            print(f"¡Posible error de comillas impares en la fila {i+1}!")
        # Busca comas fuera de comillas
        if ',' in line and '"' not in line:
            print(f"¡Coma fuera de comillas en la fila {i+1}!")

# 3. Buscar saltos de línea dentro de celdas
with open(filename, encoding='utf-8') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        for col in row:
            if '\n' in col:
                print(f"¡Salto de línea dentro de la celda en fila {idx+1}!")

print("Chequeo terminado.")
