import numpy as np


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

    return nueva_poblacion