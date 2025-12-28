#!/usr/bin/env python3
"""
Pollard p-1 - Versión para Experimentación (Corregida según Temario)
--------------------------------------------------------------------
Algoritmo Pollard p-1 ajustado a la definición de la Diapositiva 22.
"""

import math
import time
import random 
from typing import List, Dict
import json
import csv

def mcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def pollard_p1_factorizar(n: int, timeout: float = None, max_k: int = 10000) -> Dict:
    """
    Algoritmo Pollard p-1 según Diapositiva 22 del Tema 10-1.
    """
    inicio = time.time()
    bits = n.bit_length()
    
    if n % 2 == 0:
        return {
            'n': n, 'bits': bits, 'factores': (2, n // 2),
            'tiempo_segundos': time.time() - inicio, 'iteraciones': 0,
            'exito': True, 'metodo': 'pollard_p1', 'timeout': False
        }
    
    
    a = random.randint(2, n - 1)
    
    # Paso 2: if 1 < mcd(A, n) < n then return mcd(A, n)
    d = mcd(a, n)
    if 1 < d < n:
        q = n // d
        return {
            'n': n, 'bits': bits, 'factores': (min(d, q), max(d, q)),
            'tiempo_segundos': time.time() - inicio, 'iteraciones': 0,
            'exito': True, 'metodo': 'pollard_p1', 'timeout': False,
            'motivo': 'mcd_inicial'
        }
    
    # Paso 3: k = 2
    k = 2
    
    # Paso 4: while True (limitado por max_k)
    while k <= max_k:
        if timeout and (time.time() - inicio) > timeout:
            return {
                'n': n, 'bits': bits, 'factores': None,
                'tiempo_segundos': time.time() - inicio, 'iteraciones': k - 2,
                'exito': False, 'metodo': 'pollard_p1', 'timeout': True
            }
        
        # Paso 5: A = A^k mod n
        a = pow(a, k, n)
        
        # Paso 6: d = mcd(A - 1, n)
        d = mcd(a - 1, n)
        
        # Paso 7
        if 1 < d < n:
            q = n // d
            return {
                'n': n, 'bits': bits, 'factores': (min(d, q), max(d, q)),
                'tiempo_segundos': time.time() - inicio, 'iteraciones': k - 1,
                'exito': True, 'metodo': 'pollard_p1', 'timeout': False,
                'k_final': k
            }
        
        # Paso 8
        if d == n:
            return {
                'n': n, 'bits': bits, 'factores': None,
                'tiempo_segundos': time.time() - inicio, 'iteraciones': k - 1,
                'exito': False, 'metodo': 'pollard_p1', 'timeout': False,
                'motivo': 'd_igual_n'
            }
        
        # Paso 9
        k += 1
    
    return {
        'n': n, 'bits': bits, 'factores': None,
        'tiempo_segundos': time.time() - inicio, 'iteraciones': max_k - 1,
        'exito': False, 'metodo': 'pollard_p1', 'timeout': False,
        'motivo': 'max_k_alcanzado'
    }


def calcular_estadisticas(resultados: List[Dict]) -> Dict:
    """Calcula estadísticas de un conjunto de resultados."""
    tiempos_todos = [r['tiempo_segundos'] for r in resultados]
    tiempos_exitosos = [r['tiempo_segundos'] for r in resultados if r['exito']]
    iteraciones_exitosas = [r['iteraciones'] for r in resultados if r['exito']]
    
    if not tiempos_todos:
        return {
            'num_problemas': len(resultados),
            'num_exitosos': 0,
            'tasa_exito': 0.0,
            'tiempo_medio': None,
            'tiempo_mediana': None,
            'tiempo_min': None,
            'tiempo_max': None,
            'tiempo_desv_std': None,
            'iteraciones_media': None,
        }
    
    tiempos_ordenados = sorted(tiempos_todos)
    n = len(tiempos_todos)
    
    # Mediana
    if n % 2 == 0:
        mediana = (tiempos_ordenados[n//2 - 1] + tiempos_ordenados[n//2]) / 2
    else:
        mediana = tiempos_ordenados[n//2]
    
    # Media
    media = sum(tiempos_todos) / n
    
    # Desviación estándar
    varianza = sum((t - media) ** 2 for t in tiempos_todos) / n
    desv_std = math.sqrt(varianza)
    
    return {
        'num_problemas': len(resultados),
        'num_exitosos': len(tiempos_exitosos),
        'tasa_exito': len(tiempos_exitosos) / len(resultados),
        'tiempo_medio': media,
        'tiempo_mediana': mediana,
        'tiempo_min': min(tiempos_todos),
        'tiempo_max': max(tiempos_todos),
        'tiempo_desv_std': desv_std,
        'iteraciones_media': sum(iteraciones_exitosas) / len(iteraciones_exitosas) if iteraciones_exitosas else None,
    }


def guardar_resultados(resultados: List[Dict], nombre_archivo: str):
    """Guarda los resultados en formato JSON."""
    with open(nombre_archivo, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n✓ Resultados guardados en {nombre_archivo}")


def mostrar_resumen(resultados: List[Dict]):
    """Muestra un resumen de los resultados."""
    stats = calcular_estadisticas(resultados)
    
    print(f"\n{'='*70}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*70}")
    print(f"Total de problemas:        {stats['num_problemas']}")
    print(f"Problemas resueltos:       {stats['num_exitosos']}")
    print(f"Tasa de éxito:             {stats['tasa_exito']*100:.1f}%")
    
    if stats['tiempo_medio'] is not None:
        print(f"\nTiempos (segundos):")
        print(f"  Media:                   {stats['tiempo_medio']:.6f}")
        print(f"  Mediana:                 {stats['tiempo_mediana']:.6f}")
        print(f"  Mínimo:                  {stats['tiempo_min']:.6f}")
        print(f"  Máximo:                  {stats['tiempo_max']:.6f}")
        print(f"  Desviación estándar:     {stats['tiempo_desv_std']:.6f}")
        
        if stats['iteraciones_media'] is not None:
            print(f"\nIteraciones media:         {stats['iteraciones_media']:.1f}")
    
    print(f"{'='*70}")
    
    return stats


def cargar_retos_poliformat(archivo: str = "reto_ext.txt"):
    """Carga los retos del archivo de Poliformat."""
    retos = {}
    
    with open(archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith('#'):
                continue
            
            partes = linea.split(',')
            if len(partes) == 2:
                bits = int(partes[0].strip())
                numero = int(partes[1].strip())
                
                if bits not in retos:
                    retos[bits] = []
                retos[bits].append(numero)
    
    return retos


def generar_tabla_excel(estadisticas_por_tamanio: Dict[int, Dict],
                        tamanios_deseados: List[int], 
                        archivo_salida: str = "pollard_p1_resultados.csv"):
    """
    Genera un archivo CSV con estadísticas por tamaño, listo para Excel.
    Usa comas decimales (formato europeo).
    """
    # Preparar datos para CSV
    filas = []
    
    for bits in tamanios_deseados:
        if bits not in estadisticas_por_tamanio:
            continue
        
        stats = estadisticas_por_tamanio[bits]
        
        fila = {
            'Tamaño (bits)': bits,
            'Media': f"{stats['tiempo_medio']:.6f}".replace('.', ',') if stats['tiempo_medio'] is not None else "N/A",
            'Mediana': f"{stats['tiempo_mediana']:.6f}".replace('.', ',') if stats['tiempo_mediana'] is not None else "N/A",
            'Mínimo': f"{stats['tiempo_min']:.6f}".replace('.', ',') if stats['tiempo_min'] is not None else "N/A",
            'Máximo': f"{stats['tiempo_max']:.6f}".replace('.', ',') if stats['tiempo_max'] is not None else "N/A",
            'Desviación estándar': f"{stats['tiempo_desv_std']:.6f}".replace('.', ',') if stats['tiempo_desv_std'] is not None else "N/A",
            'Iteraciones media': f"{stats['iteraciones_media']:.1f}".replace('.', ',') if stats['iteraciones_media'] is not None else "N/A",
            'Tasa éxito': f"{stats['tasa_exito']*100:.2f}%".replace('.', ',')
        }
        filas.append(fila)
    
    # Escribir CSV
    columnas = ['Tamaño (bits)', 'Media', 'Mediana', 'Mínimo', 'Máximo', 
                'Desviación estándar', 'Iteraciones media', 'Tasa éxito']
    
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas, delimiter='\t')
        writer.writeheader()
        writer.writerows(filas)
    
    print(f"\n✓ Tabla Excel guardada en: {archivo_salida}")
    
    # También mostrar en consola
    print("\n" + "="*120)
    print("TABLA DE RESULTADOS POLLARD P-1 (copiar y pegar en Excel)")
    print("="*120)
    print("\t".join(columnas))
    print("-"*120)
    for fila in filas:
        print("\t".join(str(fila[col]) for col in columnas))
    print("="*120)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" POLLARD P-1 - EXPERIMENTACIÓN")
    print("="*70)
    
    # Cargar retos de Poliformat
    retos = cargar_retos_poliformat("reto_ext.txt")
    
    # Configuración
    TAMANIOS = [24, 32, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 92, 104, 116, 128]
    TIMEOUT = 60.0
    
    MAX_K_GLOBAL = 500000  # Límite de seguridad inalcanzable en 60s normalmente
    
    estadisticas_por_tamanio = {}
    
    # Procesar cada tamaño
    for bits in TAMANIOS:
        if bits not in retos:
            print(f"\n⚠️  No hay retos de {bits} bits en el archivo")
            continue
            
        print(f"\n{'='*70}")
        print(f" Procesando {len(retos[bits])} números de {bits} bits")
        print(f" Parámetro max_k (seguridad) = {MAX_K_GLOBAL}")
        print(f"{'='*70}")
        
        resultados = []
        
        # Factorizar cada número
        for i, n in enumerate(retos[bits], 1):
            print(f"Problema {i}/{len(retos[bits])} (n={n})...", end=" ", flush=True)
            
            
            resultado = pollard_p1_factorizar(n, timeout=TIMEOUT, max_k=MAX_K_GLOBAL)
            resultados.append(resultado)
            
            # Mostrar resultado
            if resultado['exito']:
                p, q = resultado['factores']
                tiempo_ms = resultado['tiempo_segundos'] * 1000
                k_final = resultado.get('k_final', '?')
                print(f"✓ {tiempo_ms:.3f}ms (k={k_final}) → {p} × {q}")
            else:
                if resultado.get('timeout', False):
                    print(f"✗ Timeout (llegó a k={resultado['iteraciones']})")
                else:
                    motivo = resultado.get('motivo', 'max_iter')
                    print(f"✗ Fallo ({motivo})")
        
        # Resumen y guardado
        stats = mostrar_resumen(resultados)
        estadisticas_por_tamanio[bits] = stats
        
        nombre_archivo = f"pollard_p1_resultados_{bits}bits.json"
        guardar_resultados(resultados, nombre_archivo)
    
    # Generar tabla Excel
    archivo_excel = "pollard_p1_tabla_resumen.csv"
    generar_tabla_excel(estadisticas_por_tamanio, TAMANIOS, archivo_excel)

    print("\n" + "="*70)
    print(" EXPERIMENTACIÓN COMPLETADA")
    print("="*70)