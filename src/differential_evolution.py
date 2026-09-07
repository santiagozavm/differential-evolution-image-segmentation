import numpy as np
from src.segmentation import evaluar_poblacion

def inicializar_poblacion(
    num_soluciones,
    num_umbrales,
    limite_inferior=1,
    limite_superior=255,
    semilla=None
):
    """
    Genera la población inicial de posibles
    soluciones de umbrales.
    """

    rng = np.random.default_rng(semilla)

    poblacion = rng.integers(
        limite_inferior,
        limite_superior + 1,
        size=(
            num_soluciones,
            num_umbrales
        )
    )

    return poblacion


def mutacion(
    poblacion,
    factor_f,
    limite_inferior=1,
    limite_superior=255,
    rng=None
):
    """
    Realiza la mutación DE/rand/1.

    Para cada individuo i:

    v = x_r1 + F * (x_r2 - x_r3)

    Los índices r1, r2 y r3 son diferentes
    entre sí y diferentes de i.
    """

    if rng is None:
        rng = np.random.default_rng()

    num_soluciones, num_umbrales = poblacion.shape

    mutaciones = np.empty(
        (
            num_soluciones,
            num_umbrales
        ),
        dtype=float
    )

    for i in range(num_soluciones):

        indices_disponibles = np.delete(
            np.arange(num_soluciones),
            i
        )

        r1, r2, r3 = rng.choice(
            indices_disponibles,
            size=3,
            replace=False
        )

        mutante = (
            poblacion[r1]
            + factor_f
            * (
                poblacion[r2]
                - poblacion[r3]
            )
        )

        mutante = np.clip(
            mutante,
            limite_inferior,
            limite_superior
        )

        mutaciones[i] = mutante

    return mutaciones


def cruza_binomial(
    poblacion,
    mutaciones,
    cr,
    rng=None
):
    """
    Realiza la cruza binomial de
    Evolución Diferencial.

    Se garantiza que al menos un valor
    provenga del vector mutante.
    """

    if rng is None:
        rng = np.random.default_rng()

    num_soluciones, num_umbrales = poblacion.shape

    nueva_poblacion = np.empty_like(
        mutaciones
    )

    for i in range(num_soluciones):

        mascara = rng.random(
            num_umbrales
        ) < cr

        indice_forzado = rng.integers(
            num_umbrales
        )

        mascara[indice_forzado] = True

        nueva_poblacion[i] = np.where(
            mascara,
            mutaciones[i],
            poblacion[i]
        )

    nueva_poblacion = np.round(
        nueva_poblacion
    ).astype(int)
    
    return nueva_poblacion

def seleccion(
    poblacion,
    nueva_poblacion,
    fitness_actual,
    fitness_nuevo
):
    """
    Selecciona entre el individuo actual y
    el nuevo individuo.

    Como el fitness se define como:

        1 / varianza_interclase

    un menor fitness representa una mejor solución.
    """

    mascara = fitness_nuevo < fitness_actual

    poblacion_siguiente = np.where(
        mascara[:, np.newaxis],
        nueva_poblacion,
        poblacion
    )

    fitness_siguiente = np.where(
        mascara,
        fitness_nuevo,
        fitness_actual
    )

    return poblacion_siguiente, fitness_siguiente

def evolucion_diferencial(
    probabilidades,
    niveles_gris,
    num_soluciones=50,
    num_umbrales=5,
    factor_f=0.9,
    cr=0.3,
    iteraciones=100,
    limite_inferior=1,
    limite_superior=255,
    semilla=None
):
    """
    Ejecuta el algoritmo de Evolución Diferencial
    para encontrar umbrales de segmentación.

    Retorna:
        poblacion: población final.
        mejor_individuo: mejores umbrales encontrados.
        mejor_fitness: fitness de la mejor solución.
        historial: mejor fitness de cada iteración.
    """

    rng = np.random.default_rng(semilla)

    poblacion = inicializar_poblacion(
        num_soluciones=num_soluciones,
        num_umbrales=num_umbrales,
        limite_inferior=limite_inferior,
        limite_superior=limite_superior,
        semilla=semilla
    )

    fitness_actual = evaluar_poblacion(
        poblacion,
        probabilidades,
        niveles_gris
    )

    historial = []

    for iteracion in range(iteraciones):

        mutaciones = mutacion(
            poblacion,
            factor_f=factor_f,
            limite_inferior=limite_inferior,
            limite_superior=limite_superior,
            rng=rng
        )

        nueva_poblacion = cruza_binomial(
            poblacion,
            mutaciones,
            cr=cr,
            rng=rng
        )

        fitness_nuevo = evaluar_poblacion(
            nueva_poblacion,
            probabilidades,
            niveles_gris
        )

        poblacion, fitness_actual = seleccion(
            poblacion,
            nueva_poblacion,
            fitness_actual,
            fitness_nuevo
        )

        mejor_fitness = np.min(fitness_actual)

        historial.append(mejor_fitness)

        print(
            f"Iteración {iteracion + 1}: "
            f"{mejor_fitness}"
        )

    indice_mejor = np.argmin(fitness_actual)

    mejor_individuo = np.sort(
        poblacion[indice_mejor]
    )

    mejor_fitness = fitness_actual[indice_mejor]

    return (
        poblacion,
        mejor_individuo,
        mejor_fitness,
        historial
    )