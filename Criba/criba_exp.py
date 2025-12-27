#!/usr/bin/env python3
"""
Criba Cuadrática (Quadratic Sieve) - Versión para Experimentación
------------------------------------------------------------------
Algoritmo de Criba Cuadrática para factorización de enteros.
Implementación basada en los apuntes de clase.
"""

import math
import time
from typing import List, Dict, Tuple, Optional
import json
import csv


def es_primo(n: int) -> bool:
    """Verifica si un número es primo (test simple)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def simbolo_legendre(a: int, p: int) -> int:
    """
    Calcula el símbolo de Legendre (a/p).
    Retorna: 1 si a es residuo cuadrático mod p
             -1 si a no es residuo cuadrático mod p
             0 si p divide a
    """
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


def generar_base_factores(n: int, B_limite: int) -> List[int]:
    """
    Genera la base de factores según los apuntes:
    B = {p₁, p₂, p₃, ..., pₖ} base de factores primos pequeños
    
    NOTA: Los apuntes incluyen -1 en la base (para manejar números negativos)
    Ejemplo: Para n=10579, B = {-1, 2, 3, 5, 7, 13}
    
    Criterio de selección (estándar de la literatura):
    Incluir primos p ≤ B_limite tales que n es residuo cuadrático mod p.
    
    Args:
        n: Número a factorizar
        B_limite: Límite superior para los primos
        
    Returns:
        Lista de primos que forman la base de factores
    """
    base = [-1, 2]  # Incluir -1 para números negativos y 2 siempre
    
    for p in range(3, B_limite + 1, 2):
        if es_primo(p) and simbolo_legendre(n, p) == 1:
            base.append(p)
    
    return base


def factorizar_sobre_base(numero: int, base: List[int]) -> Optional[List[int]]:
    """
    Intenta factorizar un número usando solo primos de la base.
    Retorna vector de exponentes si es factorizable en B, None si no lo es.
    
    Según los apuntes:
    "if b es factorizable en B then..."
    
    NOTA: La base incluye -1 como primer elemento para manejar signos.
    
    Args:
        numero: Número a factorizar
        base: Base de factores (incluyendo -1 como primer elemento)
        
    Returns:
        Lista de exponentes [e1, e2, ..., ek] donde numero = (-1)^e1 * p2^e2 * ... * pk^ek
        o None si el número no es factorizable en B
    """
    if numero == 0:
        return None
    
    exponentes = []
    num_trabajo = abs(numero)
    signo = 1 if numero > 0 else -1
    
    # Primer elemento: -1 (para el signo)
    if base[0] == -1:
        exponentes.append(0 if signo > 0 else 1)
    
    # Resto de primos
    for primo in base[1:]:
        exp = 0
        while num_trabajo % primo == 0:
            exp += 1
            num_trabajo //= primo
        exponentes.append(exp)
    
    # Si queda resto, no es factorizable en B
    if num_trabajo != 1:
        return None
    
    return exponentes


def criba_cuadratica_factorizar(n: int, timeout: float = None, B_limite: int = None) -> Dict:
    """
    Implementación simplificada de la Criba Cuadrática según los apuntes.
    
    PSEUDOCÓDIGO DE LOS APUNTES:
    1: B = {p₁, p₂, p₃, ..., pₖ} base de factores primos pequeños
    2: m = ⌊√n⌋
    3: repeat
    4:   Considera tᵢ en el orden 0, ±1, ±2 ...
    5:   a = (m + tᵢ)
    6:   b = (m + tᵢ)² - n
    7:   if b es factorizable en B then
    8:     aᵢ = a; bᵢ = b
    9:     vᵢ = exponentes de la factorización de bᵢ
    10:  end if
    11: until Se hayan considerado suficientes valores vᵢ
    
    NOTA: Esta es una versión educativa simplificada. Para números grandes
    (>80 bits), se recomienda usar implementaciones optimizadas como msieve o YAFU.
    
    Args:
        n: Número a factorizar
        timeout: Tiempo máximo en segundos
        B_limite: Límite superior para la base de factores
                 Si None, se calcula según fórmula estándar
        
    Returns:
        Diccionario con resultados y métricas
    """
    inicio = time.time()
    bits = n.bit_length()
    
    # Caso trivial: n par
    if n % 2 == 0:
        return {
            'n': n,
            'bits': bits,
            'factores': (2, n // 2),
            'tiempo_segundos': time.time() - inicio,
            'iteraciones': 0,
            'exito': True,
            'metodo': 'quadratic_sieve',
            'timeout': False
        }
    
    # Línea 1: Determinar B_limite
    # Los apuntes dicen "base de factores primos pequeños" sin especificar criterio
    # Se usa fórmula estándar de la literatura
    if B_limite is None:
        ln_n = math.log(n)
        ln_ln_n = math.log(ln_n) if ln_n > 1 else 1
        B_limite_teorico = int(math.exp(0.5 * math.sqrt(ln_n * ln_ln_n)))
        
        # Para implementación simplificada, necesitamos bases más grandes
        # Ajuste empírico basado en experimentación
        if bits <= 40:
            B_limite = min(B_limite_teorico, 100)
        elif bits <= 56:
            B_limite = max(500, min(B_limite_teorico * 2, 2000))
        elif bits <= 64:
            B_limite = max(1000, min(B_limite_teorico * 2, 3000))
        elif bits <= 72:
            B_limite = max(1500, min(B_limite_teorico * 2, 4000))
        elif bits <= 80:
            B_limite = max(2000, min(B_limite_teorico * 2, 5000))
        else:
            B_limite = max(3000, min(B_limite_teorico * 2, 10000))
        
        B_limite = max(10, min(B_limite, 10000))
    
    # Línea 1: Generar base de factores B
    base = generar_base_factores(n, B_limite)
    
    if not base:
        return {
            'n': n,
            'bits': bits,
            'factores': None,
            'tiempo_segundos': time.time() - inicio,
            'iteraciones': 0,
            'exito': False,
            'metodo': 'quadratic_sieve',
            'timeout': False,
            'motivo': 'base_vacia'
        }
    
    # Necesitamos al menos len(base) + 1 relaciones factorizables en B
    # Para ser conservadores, buscamos más relaciones
    objetivo_relaciones = len(base) + 10
    relaciones = []
    valores_a = []
    
    # Línea 2: m = ⌊√n⌋
    m = int(math.floor(math.sqrt(n)))
    
    # Línea 3-11: repeat... until suficientes valores
    # Línea 4: Considera tᵢ en el orden 0, ±1, ±2 ...
    # Para números grandes necesitamos más intentos
    max_intentos = min(500000, len(base) * 10000)
    
    for intento in range(max_intentos):
        # Verificar timeout
        if timeout and (time.time() - inicio) > timeout:
            return {
                'n': n,
                'bits': bits,
                'factores': None,
                'tiempo_segundos': time.time() - inicio,
                'iteraciones': intento,
                'exito': False,
                'metodo': 'quadratic_sieve',
                'timeout': True
            }
        
        # Generar tᵢ: 0, 1, -1, 2, -2, 3, -3, ...
        if intento == 0:
            t_i = 0
        elif intento % 2 == 1:
            t_i = (intento + 1) // 2
        else:
            t_i = -(intento // 2)
        
        # Línea 5: a = (m + tᵢ)
        a = m + t_i
        
        # Línea 6: b = (m + tᵢ)² - n
        b = a * a - n
        
        # Línea 7: if b es factorizable en B
        exponentes = factorizar_sobre_base(b, base)
        
        if exponentes is not None:
            # Líneas 8-9: aᵢ = a; bᵢ = b; vᵢ = exponentes
            relaciones.append(exponentes)
            valores_a.append(a)
            
            # Línea 11: until suficientes valores
            if len(relaciones) >= objetivo_relaciones:
                break
    
    # Verificar si encontramos suficientes relaciones
    if len(relaciones) < len(base) + 1:
        return {
            'n': n,
            'bits': bits,
            'factores': None,
            'tiempo_segundos': time.time() - inicio,
            'iteraciones': intento + 1,
            'exito': False,
            'metodo': 'quadratic_sieve',
            'timeout': False,
            'motivo': 'insuficientes_relaciones'
        }
    
    # Líneas 12-18: Fase de álgebra lineal
    # Línea 13: Encontrar vectores cuya suma resulte en vector con componentes pares
    
    # Probar combinaciones de diferentes tamaños (2, 3, 4, 5 relaciones)
    from itertools import combinations
    
    for tamaño_combo in [2, 3, 4, 5]:
        if tamaño_combo > len(relaciones):
            continue
            
        for indices in combinations(range(len(relaciones)), tamaño_combo):
            # Sumar exponentes de las relaciones seleccionadas
            exp_suma = [0] * len(base)
            for idx in indices:
                for k in range(len(base)):
                    exp_suma[k] += relaciones[idx][k]
            
            # ¿Todos pares?
            if all(e % 2 == 0 for e in exp_suma):
                # Línea 14: x = ∏ aᵢ (producto de los a seleccionados)
                x_producto = 1
                for idx in indices:
                    x_producto *= valores_a[idx]
                
                # Línea 15: y = ∏ pᵢ^(exponentes/2)
                y_producto = 1
                for k, primo in enumerate(base):
                    if primo == -1:
                        # Manejar signo
                        if (exp_suma[k] // 2) % 2 == 1:
                            y_producto *= -1
                    else:
                        y_producto *= primo ** (exp_suma[k] // 2)
                
                # Línea 16: if x ≡ ±y (mod n) buscar otro conjunto
                if x_producto % n == y_producto % n or x_producto % n == (-y_producto) % n:
                    continue
                
                # Línea 17: return mcd(x - y, n)
                d = math.gcd(x_producto - y_producto, n)
                
                if 1 < d < n:
                    q = n // d
                    return {
                        'n': n,
                        'bits': bits,
                        'factores': (min(d, q), max(d, q)),
                        'tiempo_segundos': time.time() - inicio,
                        'iteraciones': intento + 1,
                        'exito': True,
                        'metodo': 'quadratic_sieve',
                        'timeout': False,
                        'B_limite': B_limite,
                        'relaciones_encontradas': len(relaciones),
                        'relaciones_usadas': len(indices)
                    }
    
    # No se encontró factorización válida
    return {
        'n': n,
        'bits': bits,
        'factores': None,
        'tiempo_segundos': time.time() - inicio,
        'iteraciones': intento + 1,
        'exito': False,
        'metodo': 'quadratic_sieve',
        'timeout': False,
        'motivo': 'no_factorización_válida',
        'relaciones_encontradas': len(relaciones)
    }


def calcular_estadisticas(resultados: List[Dict]) -> Dict:
    """
    Calcula estadísticas de un conjunto de resultados.
    Incluye todos los tiempos (éxitos + fallos).
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


def mostrar_resumen(resultados: List[Dict]) -> Dict:
    """
    Muestra un resumen de los resultados.
    Retorna las estadísticas para garantizar coherencia con la tabla Excel.
    """
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


def cargar_retos_poliformat(archivo: str = "reto.txt"):
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
                        archivo_salida: str = "quadratic_sieve_resultados.csv"):
    """
    Genera un archivo CSV con estadísticas por tamaño, listo para Excel.
    Usa comas decimales (formato europeo).
    Usa estadísticas pre-calculadas para garantizar coherencia.
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
    print("TABLA DE RESULTADOS CRIBA CUADRÁTICA (copiar y pegar en Excel)")
    print("="*120)
    print("\t".join(columnas))
    print("-"*120)
    for fila in filas:
        print("\t".join(str(fila[col]) for col in columnas))
    print("="*120)


def test_ejemplo_apuntes():
    """
    Prueba con el ejemplo de los apuntes: n = 10579
    Base B = {-1, 2, 3, 5, 7, 13}
    """
    print("\n" + "="*70)
    print("PRUEBA: Ejemplo de los apuntes (n = 10579)")
    print("="*70)
    
    n = 10579
    print(f"\nFactorizando n = {n} con Criba Cuadrática...")
    print(f"Los apuntes usan: B = {{-1, 2, 3, 5, 7, 13}}")
    print(f"Esperado: factores 71 y 149")
    
    # Para este ejemplo usamos B_limite pequeño como en los apuntes
    resultado = criba_cuadratica_factorizar(n, B_limite=15)
    
    print(f"\nResultado:")
    print(f"  Factores: {resultado['factores']}")
    print(f"  Tiempo: {resultado['tiempo_segundos']:.6f} segundos")
    print(f"  Éxito: {resultado['exito']}")
    
    if resultado['exito']:
        p, q = resultado['factores']
        print(f"  Verificación: {p} × {q} = {p * q}")
        if sorted([p, q]) == [71, 149]:
            print(f"  ✓ ¡Coincide con los apuntes! (71 × 149)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" CRIBA CUADRÁTICA - EXPERIMENTACIÓN")
    print("="*70)
    print("\nAlgoritmo: Quadratic Sieve (según apuntes)")
    print("Base: B = {p₁, p₂, ..., pₖ} primos pequeños")
    print("Complejidad: O(exp(√(ln n ln ln n)))")
    print("Ventaja: Muy eficiente para números de 60-100 bits")
    print("Desventaja: Complejo, requiere memoria")
    print("\n⚠️  NOTA: Implementación simplificada educativa")
    print("   Para números >80 bits, usar msieve o YAFU")
    print("="*70)
    
    # Primero probar con ejemplo de los apuntes
    test_ejemplo_apuntes()
    
    # Comentar para ejecución automática:
    # input("\nPresiona Enter para continuar con los retos de Poliformat...")
    
    # Cargar retos de Poliformat
    retos = cargar_retos_poliformat("reto.txt")
    
    # Configuración
    # IMPORTANTE: Esta implementación simplificada funciona mejor en números pequeños
    # Para números >56 bits, las limitaciones del álgebra lineal simplificada
    # hacen que la tasa de éxito sea muy baja
    TAMANIOS = [40, 44, 48, 52, 56]  # Rango realista para implementación simplificada
    TIMEOUT = 60.0
    
    estadisticas_por_tamanio = {}
    
    # Procesar cada tamaño
    for bits in TAMANIOS:
        if bits not in retos:
            print(f"\n⚠️  No hay retos de {bits} bits en el archivo")
            continue
        
        print(f"\n{'='*70}")
        print(f" Procesando {len(retos[bits])} números de {bits} bits")
        print(f" (B_limite se calcula automáticamente)")
        print(f"{'='*70}")
        
        resultados = []
        
        # Factorizar cada número
        for i, n in enumerate(retos[bits], 1):
            print(f"Problema {i}/{len(retos[bits])} (n={n})...", end=" ", flush=True)
            
            resultado = criba_cuadratica_factorizar(n, timeout=TIMEOUT)
            resultados.append(resultado)
            
            # Mostrar resultado
            if resultado['exito']:
                p, q = resultado['factores']
                tiempo_ms = resultado['tiempo_segundos'] * 1000
                print(f"✓ {tiempo_ms:.3f}ms → {p} × {q}")
            else:
                if resultado.get('timeout', False):
                    print(f"✗ Timeout")
                else:
                    motivo = resultado.get('motivo', 'fallo')
                    print(f"✗ Fallo ({motivo})")
        
        # Resumen y guardado
        stats = mostrar_resumen(resultados)
        estadisticas_por_tamanio[bits] = stats
        
        nombre_archivo = f"quadratic_sieve_resultados_{bits}bits.json"
        guardar_resultados(resultados, nombre_archivo)
    
    # Generar tabla Excel
    archivo_excel = "quadratic_sieve_tabla_resumen.csv"
    generar_tabla_excel(estadisticas_por_tamanio, TAMANIOS, archivo_excel)

    print("\n" + "="*70)
    print(" EXPERIMENTACIÓN COMPLETADA")
    print("="*70)