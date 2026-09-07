import cv2
import numpy as np

from src.segmentation import (
    cargar_imagen,
    calcular_histograma,
    evaluar_individuo,
    evaluar_poblacion
)

from src.differential_evolution import (
    inicializar_poblacion,
    mutacion,
    cruza_binomial,
    seleccion
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

    # Evaluar población inicial

    fitness_actual = evaluar_poblacion(
        poblacion,
        probabilidades,
        niveles_gris
    )

    print("\nMejor fitness inicial:")
    print(np.min(fitness_actual))

    # -------------------------
    # Mutación
    # -------------------------

    mutaciones = mutacion(
        poblacion,
        factor_f=F,
        rng=rng
    )

    # -------------------------
    # Cruza
    # -------------------------

    nueva_poblacion = cruza_binomial(
        poblacion,
        mutaciones,
        cr=CR,
        rng=rng
    )

    # -------------------------
    # Evaluar nuevos individuos
    # -------------------------

    fitness_nuevo = evaluar_poblacion(
        nueva_poblacion,
        probabilidades,
        niveles_gris
    )
    
    # -------------------------
    # Selección
    # -------------------------
    fitness_anterior = fitness_actual.copy()

    poblacion, fitness_actual = seleccion(
        poblacion,
        nueva_poblacion,
        fitness_actual,
        fitness_nuevo
    )

    print(
        "\nIndividuos reemplazados:"
    )

    print(
        np.sum(
            fitness_nuevo < fitness_anterior
        )
    )


    print("Mejor fitness después de una generación:")
    print(np.min(fitness_actual))



if __name__ == "__main__":
    main()