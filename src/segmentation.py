import cv2
import numpy as np


def cargar_imagen(ruta_imagen):
    """
    Carga una imagen y la convierte a escala de grises.
    """

    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen: {ruta_imagen}"
        )

    imagen_grises = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    return imagen_grises


def calcular_histograma(imagen_grises):
    """
    Calcula el histograma normalizado y los niveles de gris.
    """

    histograma, _ = np.histogram(
        imagen_grises.flatten(),
        bins=256,
        range=(0, 256)
    )

    niveles_gris = np.arange(256)

    probabilidades = (
        histograma / imagen_grises.size
    )

    return probabilidades, niveles_gris


def evaluar_individuo(
    individuo,
    probabilidades,
    niveles_gris
):
    """
    Evalúa una solución de umbrales utilizando
    la varianza interclase.

    Un menor valor de fitness representa
    una mejor solución.
    """

    umbrales = np.sort(
        np.unique(individuo)
    )

    if len(umbrales) == 0:
        return np.inf

    grupos = len(umbrales) + 1

    probabilidades_grupos = np.zeros(grupos)
    medias_grupos = np.zeros(grupos)

    subarreglos_probabilidades = np.split(
        probabilidades,
        umbrales
    )

    subarreglos_niveles = np.split(
        niveles_gris,
        umbrales
    )

    for i in range(grupos):

        probabilidad = np.sum(
            subarreglos_probabilidades[i]
        )

        probabilidades_grupos[i] = probabilidad

        if probabilidad > 0:

            media = np.sum(
                subarreglos_niveles[i]
                * subarreglos_probabilidades[i]
            ) / probabilidad

        else:
            media = 0

        medias_grupos[i] = media

    intensidad_promedio = np.sum(
        probabilidades_grupos
        * medias_grupos
    )

    varianza_interclase = np.sum(
        probabilidades_grupos
        * np.square(
            medias_grupos
            - intensidad_promedio
        )
    )

    if varianza_interclase <= 0:
        return np.inf

    fitness = 1 / varianza_interclase

    return fitness


def evaluar_poblacion(
    poblacion,
    probabilidades,
    niveles_gris
):
    """
    Evalúa todos los individuos de la población.
    """

    fitness = np.zeros(
        len(poblacion)
    )

    for i, individuo in enumerate(poblacion):

        fitness[i] = evaluar_individuo(
            individuo,
            probabilidades,
            niveles_gris
        )

    return fitness


def segmentar_imagen(
    imagen_grises,
    umbrales
):
    """
    Segmenta una imagen utilizando múltiples umbrales.
    """

    umbrales = np.sort(
        np.unique(umbrales)
    )

    imagen_segmentada = np.zeros_like(
        imagen_grises
    )

    for i, umbral in enumerate(umbrales):

        if i == 0:

            imagen_segmentada[
                imagen_grises <= umbral
            ] = 0

        else:

            valor_medio = int(
                (
                    umbrales[i]
                    + umbrales[i - 1]
                ) / 2
            )

            mascara = (
                (imagen_grises > umbrales[i - 1])
                &
                (imagen_grises <= umbral)
            )

            imagen_segmentada[
                mascara
            ] = valor_medio

    if len(umbrales) > 0:

        imagen_segmentada[
            imagen_grises > umbrales[-1]
        ] = 255

    return imagen_segmentada