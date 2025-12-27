#!/usr/bin/env python3
"""
Fermat - Versión para Experimentación (Corregida según Temario)
---------------------------------------------------------------
Versión del algoritmo de Fermat preparada para realizar experimentos.
Ajustado para usar aritmética entera de precisión arbitraria (math.isqrt)
en lugar de flotante, necesario para los tamaños de clave del trabajo.
"""

import math
import time
from typing import List, Dict
import json
import csv

def es_cuadrado_perfecto(n: int) -> bool:
    """Verifica si un número es un cuadrado perfecto usando aritmética entera."""
    if n < 0:
        return False

    raiz = math.isqrt(n)
    return raiz * raiz == n

def fermat_factorizar(n: int, timeout: float = None, max_iteraciones: int = 100000000) -> Dict:
    """
    Algoritmo de Fermat según Diapositiva 16 del Tema 10-1.
    """
    inicio = time.time()
    bits = n.bit_length()
    
    # Caso especial: n par
    if n % 2 == 0:
        return {
            'n': n,
            'bits': bits,
            'factores': (2, n // 2),
            'tiempo_segundos': time.time() - inicio,
            'iteraciones': 0,
            'exito': True,
            'metodo': 'fermat',
            'timeout': False
        }
    

    # math.isqrt(n) devuelve floor(sqrt(n))
    raiz_n = math.isqrt(n)
    if raiz_n * raiz_n == n:
        A = raiz_n
    else:
        A = raiz_n + 1
    
    for iteracion in range(max_iteraciones):
        # Verificar timeout
        if timeout and (time.time() - inicio) > timeout:
            return {
                'n': n,
                'bits': bits,
                'factores': None,
                'tiempo_segundos': time.time() - inicio,
                'iteraciones': iteracion,
                'exito': False,
                'metodo': 'fermat',
                'timeout': True
            }
        
        B = A * A - n
        
        if es_cuadrado_perfecto(B):

            raiz_B = math.isqrt(B)
            p = A + raiz_B
            q = A - raiz_B
            
            # Verificación de seguridad
            if p * q == n:
                return {
                    'n': n,
                    'bits': bits,
                    'factores': (min(p, q), max(p, q)),
                    'tiempo_segundos': time.time() - inicio,
                    'iteraciones': iteracion + 1,
                    'exito': True,
                    'metodo': 'fermat',
                    'timeout': False
                }
        
        A += 1
    
    return {
        'n': n,
        'bits': bits,
        'factores': None,
        'tiempo_segundos': time.time() - inicio,
        'iteraciones': max_iteraciones,
        'exito': False,
        'metodo': 'fermat',
        'timeout': False
    }

def calcular_estadisticas(resultados: List[Dict]) -> Dict:
    """
    Calcula estadísticas de un conjunto de resultados.
    
    CORREGIDO: Ahora calcula estadísticas incluso si no hay éxitos,
    usando todos los tiempos (éxitos + fallos).
    """
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
    
    # Calcular estadísticas con TODOS los tiempos
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
    
    if stats['num_exitosos'] > 0:
        print(f"\nTiempos (segundos):")
        print(f"  Media:                   {stats['tiempo_medio']:.6f}")
        print(f"  Mediana:                 {stats['tiempo_mediana']:.6f}")
        print(f"  Mínimo:                  {stats['tiempo_min']:.6f}")
        print(f"  Máximo:                  {stats['tiempo_max']:.6f}")
        print(f"  Desviación estándar:     {stats['tiempo_desv_std']:.6f}")
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
                        archivo_salida: str = "fermat_resultados.csv"):
    """
    Genera un archivo CSV con estadísticas por tamaño, listo para Excel.
    Usa comas decimales (formato europeo).
    
    CORREGIDO: Ahora usa estadísticas pre-calculadas del resumen,
    garantizando coherencia entre lo mostrado en consola y la tabla Excel.
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
    print(f"  Puedes abrirlo directamente en Excel")
    
    # También mostrar en consola
    print("\n" + "="*120)
    print("TABLA DE RESULTADOS (copiar y pegar en Excel)")
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
    print(" FERMAT - EXPERIMENTACIÓN")
    print("="*70)
    
    # Cargar retos de Poliformat
    retos = cargar_retos_poliformat("reto_ext.txt")
    
    # Configuración del experimento
    # TAMANIOS = [24, 32, 40, 44]  # Tamaños en bits a procesar
    TAMANIOS = [24, 32, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 92]
    # TAMANIOS = sorted(retos.keys())
    TIMEOUT = 60.0          # Timeout en segundos
    
    estadisticas_por_tamanio = {}
    
    # Iterar por cada tamaño
    for bits in TAMANIOS:
        if bits not in retos:
            print(f"\n⚠️  No hay retos de {bits} bits en el archivo")
            continue
        
        print(f"\n{'='*70}")
        print(f" Procesando {len(retos[bits])} números de {bits} bits")
        print(f"{'='*70}")
        
        resultados = []
        
        # Procesar cada número del tamaño actual
        for i, n in enumerate(retos[bits], 1):
            print(f"Problema {i}/{len(retos[bits])} (n={n})...", end=" ", flush=True)
            
            resultado = fermat_factorizar(n, timeout=TIMEOUT)
            resultados.append(resultado)
            
            # Mostrar resultado inmediato
            if resultado['exito']:
                p, q = resultado['factores']
                tiempo_ms = resultado['tiempo_segundos'] * 1000
                print(f"✓ {tiempo_ms:.3f}ms ({resultado['iteraciones']} iter) → {p} × {q}")
            else:
                if resultado['timeout']:
                    print(f"✗ Timeout")
                else:
                    print(f"✗ No encontrado")
        

        stats = mostrar_resumen(resultados)
        estadisticas_por_tamanio[bits] = stats
        
        # Guardar resultados de este tamaño en JSON
        nombre_archivo = f"fermat_resultados_{bits}bits.json"
        guardar_resultados(resultados, nombre_archivo)
    

    archivo_excel = "fermat_tabla_resumen.csv"
    generar_tabla_excel(estadisticas_por_tamanio, TAMANIOS, archivo_excel)

    print("\n" + "="*70)
    print(" EXPERIMENTACIÓN COMPLETADA")
    print("="*70)
