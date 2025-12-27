#!/usr/bin/env python3
"""
Algoritmo de Fermat para Factorización de Enteros
--------------------------------------------------
Implementación paso a paso del algoritmo de Fermat según los apuntes de clase.

Basado en encontrar dos números x e y tales que:
    x² - y² = n
    (x + y)(x - y) = n
    
Por lo tanto:
    p = x + y
    q = x - y
"""

import math
import time
from typing import Tuple, Optional


def es_cuadrado_perfecto(n: int) -> bool:
    """
    Verifica si un número es un cuadrado perfecto.
    
    Args:
        n: Número entero a verificar
        
    Returns:
        True si n es un cuadrado perfecto, False en caso contrario
        
    Ejemplo:
        >>> es_cuadrado_perfecto(16)
        True
        >>> es_cuadrado_perfecto(15)
        False
    """
    if n < 0:
        return False
    
    raiz = int(math.sqrt(n))
    return raiz * raiz == n


def fermat_basico(n: int, max_iteraciones: int = 10000) -> Optional[Tuple[int, int]]:
    """
    Implementación básica del algoritmo de Fermat.
    
    Este algoritmo funciona mejor cuando los factores de n son cercanos entre sí.
    
    Args:
        n: Número entero compuesto a factorizar
        max_iteraciones: Número máximo de iteraciones antes de abortar
        
    Returns:
        Una tupla (p, q) con los factores de n, o None si no se encontraron
        
    Complejidad:
        O(√n) en el peor caso cuando los factores están muy separados
        
    Ejemplo según apuntes (n = 44461):
        n = 44461
        A = 211, B = 60 (no es cuadrado)
        A = 212, B = 483 (no es cuadrado)
        A = 213, B = 908 (no es cuadrado)
        A = 214, B = 1335 (no es cuadrado)
        A = 215, B = 1764 = 42² (¡es cuadrado!)
        p = 215 + 42 = 257
        q = 215 - 42 = 173
    """
    # Paso 1: Verificar que n no es primo y es impar (optimización)
    if n % 2 == 0:
        return (2, n // 2)
    
    # Paso 2: A = ⌈√n⌉ (techo de la raíz cuadrada de n)
    A = int(math.ceil(math.sqrt(n)))
    
    # Paso 3: Iterar hasta encontrar un cuadrado perfecto
    for iteracion in range(max_iteraciones):
        # B = A² - n
        B = A * A - n
        
        # Verificar si B es un cuadrado perfecto
        if es_cuadrado_perfecto(B):
            raiz_B = int(math.sqrt(B))
            
            # Factores: p = A + √B, q = A - √B
            p = A + raiz_B
            q = A - raiz_B
            
            # Verificar que los factores son correctos
            if p * q == n:
                return (min(p, q), max(p, q))  # Retornar en orden
        
        # Incrementar A para la siguiente iteración
        A += 1
    
    # No se encontró factorización en el límite de iteraciones
    return None


def fermat_con_info(n: int, max_iteraciones: int = 10000, verbose: bool = False) -> dict:
    """
    Versión del algoritmo de Fermat que retorna información detallada.
    
    Útil para la experimentación y análisis del comportamiento del algoritmo.
    
    Args:
        n: Número entero compuesto a factorizar
        max_iteraciones: Número máximo de iteraciones antes de abortar
        verbose: Si True, imprime información de cada iteración
        
    Returns:
        Diccionario con:
            - 'n': Número factorizado
            - 'factores': Tupla (p, q) con los factores, o None
            - 'tiempo': Tiempo de ejecución en segundos
            - 'iteraciones': Número de iteraciones realizadas
            - 'bits': Tamaño de n en bits
            - 'exito': Boolean indicando si se encontró la factorización
    """
    inicio = time.time()
    
    # Información del problema
    bits = n.bit_length()
    
    if n % 2 == 0:
        fin = time.time()
        return {
            'n': n,
            'factores': (2, n // 2),
            'tiempo': fin - inicio,
            'iteraciones': 0,
            'bits': bits,
            'exito': True
        }
    
    A = int(math.ceil(math.sqrt(n)))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Factorizando n = {n} ({bits} bits)")
        print(f"A inicial (⌈√n⌉) = {A}")
        print(f"{'='*60}")
        print(f"{'Iter':<6} {'A':<15} {'B':<15} {'√B':<15} {'¿Cuadrado?':<12}")
        print(f"{'-'*60}")
    
    for iteracion in range(max_iteraciones):
        B = A * A - n
        
        if es_cuadrado_perfecto(B):
            raiz_B = int(math.sqrt(B))
            p = A + raiz_B
            q = A - raiz_B
            
            if verbose:
                print(f"{iteracion:<6} {A:<15} {B:<15} {raiz_B:<15} {'SÍ':<12}")
                print(f"\n✓ Factorización encontrada:")
                print(f"  p = {A} + {raiz_B} = {p}")
                print(f"  q = {A} - {raiz_B} = {q}")
                print(f"  Verificación: {p} × {q} = {p*q}")
            
            fin = time.time()
            return {
                'n': n,
                'factores': (min(p, q), max(p, q)),
                'tiempo': fin - inicio,
                'iteraciones': iteracion + 1,
                'bits': bits,
                'exito': True
            }
        
        if verbose and iteracion < 10:  # Solo mostrar primeras iteraciones
            raiz_aprox = math.sqrt(B)
            print(f"{iteracion:<6} {A:<15} {B:<15} {raiz_aprox:<15.1f} {'No':<12}")
        
        A += 1
    
    fin = time.time()
    
    if verbose:
        print(f"\n✗ No se encontró factorización en {max_iteraciones} iteraciones")
    
    return {
        'n': n,
        'factores': None,
        'tiempo': fin - inicio,
        'iteraciones': max_iteraciones,
        'bits': bits,
        'exito': False
    }


# ============================================================================
# EJEMPLOS Y PRUEBAS
# ============================================================================

def test_ejemplo_apuntes():
    """
    Prueba con el ejemplo de los apuntes: n = 44461
    Esperado: factores 257 y 173
    """
    print("\n" + "="*70)
    print("PRUEBA 1: Ejemplo de los apuntes (n = 44461)")
    print("="*70)
    
    n = 44461
    resultado = fermat_con_info(n, verbose=True)
    
    print(f"\nResultado:")
    print(f"  Factores: {resultado['factores']}")
    print(f"  Tiempo: {resultado['tiempo']:.6f} segundos")
    print(f"  Iteraciones: {resultado['iteraciones']}")


def test_factores_cercanos():
    """
    Prueba con factores muy cercanos (caso óptimo para Fermat)
    """
    print("\n" + "="*70)
    print("PRUEBA 2: Factores cercanos (caso óptimo)")
    print("="*70)
    
    # 101 * 103 = 10403
    n = 10403
    resultado = fermat_con_info(n, verbose=True)
    
    print(f"\nResultado:")
    print(f"  Factores: {resultado['factores']}")
    print(f"  Tiempo: {resultado['tiempo']:.6f} segundos")
    print(f"  Iteraciones: {resultado['iteraciones']}")


def test_factores_lejanos():
    """
    Prueba con factores muy separados (caso difícil para Fermat)
    """
    print("\n" + "="*70)
    print("PRUEBA 3: Factores lejanos (caso difícil)")
    print("="*70)
    
    # 7 * 1009 = 7063
    n = 7063
    resultado = fermat_con_info(n, max_iteraciones=1000, verbose=False)
    
    print(f"\nFactorizando n = {n}")
    print(f"Resultado:")
    print(f"  Factores: {resultado['factores']}")
    print(f"  Tiempo: {resultado['tiempo']:.6f} segundos")
    print(f"  Iteraciones: {resultado['iteraciones']}")
    print(f"  Éxito: {resultado['exito']}")


def test_comparacion_casos():
    """
    Compara el rendimiento en diferentes casos
    """
    print("\n" + "="*70)
    print("PRUEBA 4: Comparación de rendimiento")
    print("="*70)
    
    casos = [
        ("Factores cercanos", 101, 103),
        ("Factores medianamente cercanos", 97, 113),
        ("Factores algo separados", 89, 127),
    ]
    
    print(f"\n{'Caso':<30} {'n':<12} {'Factores':<20} {'Iter':<8} {'Tiempo (ms)':<12}")
    print("-" * 90)
    
    for nombre, p, q in casos:
        n = p * q
        resultado = fermat_con_info(n)
        if resultado['exito']:
            tiempo_ms = resultado['tiempo'] * 1000
            print(f"{nombre:<30} {n:<12} {str(resultado['factores']):<20} "
                  f"{resultado['iteraciones']:<8} {tiempo_ms:<12.4f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ALGORITMO DE FERMAT - IMPLEMENTACIÓN Y PRUEBAS")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_ejemplo_apuntes()
    test_factores_cercanos()
    test_factores_lejanos()
    test_comparacion_casos()
    
    print("\n" + "="*70)
    print(" FIN DE LAS PRUEBAS")
    print("="*70)
