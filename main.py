import cv2
import numpy as np

from src.segmentation import (
    cargar_imagen,
    calcular_histograma,
    evaluar_individuo
)

from src.differential_evolution import (
    inicializar_poblacion,
    mutacion,
    cruza_binomial
)


RUTA_IMAGEN = "images/mar.png"

NUM_UMBRALES = 5
NUM_SOLUCIONES = 50

F = 0.9
CR = 0.3

SEMILLA = 42


def main():

    # -------------------------
    # Procesamiento de imagen
    # -------------------------

    imagen_grises = cargar_imagen(
        RUTA_IMAGEN
    )

    cv2.imwrite(
        "results/imagen_grises.jpg",
        imagen_grises
    )

    probabilidades, niveles_gris = calcular_histograma(
        imagen_grises
    )

    # -------------------------
    # Prueba de evaluación
    # -------------------------

    individuo_prueba = [
        50,
        100,
        150,
        200,
        230
    ]

    fitness = evaluar_individuo(
        individuo_prueba,
        probabilidades,
        niveles_gris
    )

    print("Individuo de prueba:")
    print(individuo_prueba)

    print("\nFitness:")
    print(fitness)

    # -------------------------
    # Evolución Diferencial
    # -------------------------

    rng = np.random.default_rng(
        SEMILLA
    )

    poblacion = inicializar_poblacion(
        num_soluciones=NUM_SOLUCIONES,
        num_umbrales=NUM_UMBRALES,
        semilla=SEMILLA
    )

    print("\nPoblación inicial:")
    print(poblacion[:5])

    mutaciones = mutacion(
        poblacion,
        factor_f=F,
        rng=rng
    )

    print("\nPrimeros 5 vectores mutantes:")
    print(mutaciones[:5])

    nueva_poblacion = cruza_binomial(
        poblacion,
        mutaciones,
        cr=CR,
        rng=rng
    )

    print("\nPrimeros 5 individuos después de la cruza:")
    print(nueva_poblacion[:5])


if __name__ == "__main__":
    main()