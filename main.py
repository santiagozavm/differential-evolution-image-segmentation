import cv2
import numpy as np

from src.segmentation import (
    cargar_imagen,
    calcular_histograma,
    segmentar_imagen
)

from src.differential_evolution import (
    evolucion_diferencial
)


RUTA_IMAGEN = "images/mar.png"

NUM_UMBRALES = 5
NUM_SOLUCIONES = 50

F = 0.9
CR = 0.3

ITERACIONES = 100

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
    # Evolución Diferencial
    # -------------------------

    (
        poblacion,
        mejores_umbrales,
        mejor_fitness,
        historial
    ) = evolucion_diferencial(
        probabilidades=probabilidades,
        niveles_gris=niveles_gris,
        num_soluciones=NUM_SOLUCIONES,
        num_umbrales=NUM_UMBRALES,
        factor_f=F,
        cr=CR,
        iteraciones=ITERACIONES,
        semilla=SEMILLA
    )

    print("\nMejores umbrales encontrados:")
    print(mejores_umbrales)

    print("\nMejor fitness:")
    print(mejor_fitness)

    # -------------------------
    # Segmentación
    # -------------------------

    imagen_segmentada = segmentar_imagen(
        imagen_grises,
        mejores_umbrales
    )

    cv2.imwrite(
        "results/imagen_segmentada.jpg",
        imagen_segmentada
    )

    print("\nImagen segmentada guardada correctamente.")


if __name__ == "__main__":
    main()