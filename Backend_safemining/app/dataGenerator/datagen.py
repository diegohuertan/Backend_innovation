from transformers import BertForMaskedLM, BertTokenizer
import torch
import random
import csv
import re

# Cargar el modelo y el tokenizador de BERT
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Función de limpieza de texto
def limpiar_texto(texto):
    # Eliminar tokens duplicados
    texto = re.sub(r'\b(\w+)\s+\1\b', r'\1', texto)
    
    # Eliminar espacios extra
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # Capitalizar primera letra
    texto = texto.capitalize()
    
    return texto

# Función para generar texto variado usando BERT con más control
def generar_texto_riesgo_mejorado(frase_base, max_intentos=5):
    for _ in range(max_intentos):
        # Convertir la frase base en tokens
        tokens = tokenizer.tokenize(frase_base)
        
        # Crear posiciones de máscaras más estratégicas
        mask_positions = [i for i, token in enumerate(tokens) 
                          if token not in tokenizer.all_special_tokens 
                          and len(token) > 2  # Solo enmascarar tokens significativos
                          and random.random() < 0.1]  # Reducir probabilidad de máscara
        
        # Si no hay posiciones para enmascarar, saltar esta iteración
        if not mask_positions:
            continue
        
        # Insertar máscaras en las posiciones seleccionadas
        tokens_masked = tokens.copy()
        for pos in mask_positions:
            tokens_masked[pos] = '[MASK]'
        
        # Reconstruir la frase con las máscaras
        masked_input = tokenizer.convert_tokens_to_string(tokens_masked)
        
        # Codificar la entrada para el modelo
        inputs = tokenizer.encode(masked_input, return_tensors="pt")
        
        # Usar el modelo para predecir las palabras que faltan
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs[0]
        
        # Reemplazar las máscaras por las predicciones más probables
        for pos in mask_positions:
            predicted_token_id = torch.argmax(predictions[0, pos]).item()
            predicted_token = tokenizer.decode([predicted_token_id])
            tokens[pos] = predicted_token
        
        # Reconstruir la frase con las palabras predichas
        texto_generado = tokenizer.convert_tokens_to_string(tokens)
        
        # Limpiar el texto generado
        texto_generado = limpiar_texto(texto_generado)
        
        # Filtrar generaciones con errores obvios
        if not any(error in texto_generado.lower() for error in ['[unk]', '..', ',,', '  ']):
            return texto_generado
    
    # Fallback: devolver la frase original limpia si no se genera texto aceptable
    return limpiar_texto(frase_base)

# Definir categorías de riesgo con más ejemplos
sin_riesgo = [
    "Todo está funcionando correctamente, no hay problemas con los equipos.",
    "Los niveles de temperatura están dentro de los parámetros normales.",
    "Los equipos están operando sin ningún inconveniente.",
    "Las inspecciones de seguridad se han realizado correctamente.",
    "El sistema de monitoreo no reporta ninguna anomalía.",
    "Todos los protocolos de seguridad se están siguiendo adecuadamente.",
    "Las condiciones de trabajo son seguras y estables.",
    "No se han detectado riesgos potenciales en la zona de trabajo.",
    "El personal está cumpliendo con todas las normas de seguridad.",
    "Los equipos de protección personal están completos y en buen estado."
]

riesgo_medio = [
    "El horno está emitiendo ruidos extraños, pero no parece haber problemas graves.",
    "Hemos tenido una pequeña fluctuación en la energía, pero está estabilizándose.",
    "Se detectaron leves anomalías en la temperatura, estamos monitoreando.",
    "Hay un retraso en la maquinaria, pero no está afectando la producción.",
    "Se observa un ligero desgaste en algunos equipos de protección.",
    "La ventilación muestra signos de necesitar mantenimiento preventivo.",
    "Hay una pequeña fuga de aceite que requiere atención.",
    "Los niveles de ruido están por encima de lo normal, pero no son críticos.",
    "Se han identificado algunos procedimientos que necesitan revisión.",
    "El equipo de seguridad indica algunas mejoras menores necesarias."
]

riesgo_alto = [
    "Hay una fuga de gas cerca del área de fundición, necesitamos evacuar inmediatamente.",
    "Un trabajador ha sufrido una caída grave, necesitamos asistencia médica urgente.",
    "El sistema de ventilación ha fallado y hay humo en el área de trabajo.",
    "Una explosión ha ocurrido en la zona de fundición, necesitamos evacuación masiva.",
    "Se ha detectado una falla crítica en los sistemas de seguridad.",
    "Hay múltiples señales de riesgo inminente de accidente.",
    "El sistema de control ha colapsado completamente.",
    "Se han roto varios protocolos de seguridad simultáneamente.",
    "Hay un riesgo inmediato de colapso estructural.",
    "La exposición a productos químicos peligrosos es inminente."
]

# Generar frases variadas para cada categoría de riesgo
sin_riesgo_variado = [generar_texto_riesgo_mejorado(frase) for frase in sin_riesgo]
riesgo_medio_variado = [generar_texto_riesgo_mejorado(frase) for frase in riesgo_medio]
riesgo_alto_variado = [generar_texto_riesgo_mejorado(frase) for frase in riesgo_alto]

# Crear el archivo CSV con ejemplos simulados
with open('conversaciones_riesgo_mineria.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['texto', 'etiqueta'])
    
    for _ in range(1000):  # Aumenté a 10,000 ejemplos
        categoria = random.choice(['sin riesgo', 'riesgo medio', 'riesgo alto'])
        
        if categoria == 'sin riesgo':
            texto = random.choice(sin_riesgo_variado)
        elif categoria == 'riesgo medio':
            texto = random.choice(riesgo_medio_variado)
        else:
            texto = random.choice(riesgo_alto_variado)
        
        writer.writerow([texto, categoria])

print("Archivo CSV generado exitosamente.")