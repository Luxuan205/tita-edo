# tita

`tita` es una librería en Python para trabajar con ecuaciones diferenciales ordinarias (EDO) de primer y segundo orden: **representa, clasifica, resuelve y grafica** EDOs usando álgebra simbólica (`sympy`), sin métodos numéricos ni librerías de resolución externas — cada técnica está implementada "a mano" siguiendo el método clásico.

Proyecto de la asignatura Programación 1, Universidad El Bosque.

## Alcance (6 ramas)

1. **Núcleo** — representar una EDO (variable dependiente/independiente, forma estándar `y'=f(x,y)`, EDO vs EDP), clasificarla (orden y grado, lineal/no lineal, homogénea/no homogénea, coeficientes, autónoma/no autónoma), identificar el tipo de problema (valor inicial/frontera) y verificar soluciones candidatas.
2. **EDO de primer orden** — 9 técnicas de solución: separables, homogéneas, exactas, factor integrante, lineal, Bernoulli, Ricatti, sustituciones generales, trayectorias ortogonales.
3. **EDO lineales de orden 2** — homogéneas de coeficientes constantes, vía ecuación característica (raíces reales distintas, repetidas, complejas conjugadas).
4. **Sistemas de ecuaciones diferenciales** — reducir una EDO lineal de orden `n` a un sistema de primer orden y expresarlo en forma matricial.
5. **Módulo gráfico** — graficar curvas solución y el campo de direcciones con curvas superpuestas.
6. **Análisis cualitativo** — campo de direcciones (mismo código que el módulo gráfico).

## Instalación

```bash
python -m venv venv
venv/bin/pip install -e ".[dev]"
```

## Uso

Las ecuaciones se construyen con objetos de `sympy` (no se escriben como texto libre):

```python
from sympy import Eq, Function, symbols, solve
from tita.core import classify_ode, solve_ode

x = symbols("x")
y = Function("y")

ode = Eq(y(x).diff(x), x * y(x))  # y' = x*y

classify_ode(ode, x, y)   # -> ['separable', 'linear']
solve_ode(ode, x, y)      # -> Eq(y(x), C1*exp(x**2/2))
```

`solve_ode` clasifica la ecuación y elige automáticamente un método entre los que apliquen (prioriza `linear > bernoulli > homogeneous > separable > exact`). Para EDO de orden 2, solo resuelve el caso lineal homogéneo de coeficientes constantes.

### Graficar

```python
from tita.plotting.direction_field import plot_direction_field
from tita.plotting.solution_curves import plot_direction_field_with_solutions

# solo el campo de direcciones
figure = plot_direction_field(x * y(x), x, y, (-3, 3), (-3, 3))
figure.savefig("campo.png")

# campo + curva solución superpuesta
general_solution = solve_ode(ode, x, y)
curva = solve(general_solution, y(x))[0].subs("C1", 1)
figure = plot_direction_field_with_solutions(
    x * y(x), x, y, (-3, 3), (-3, 3), [curva]
)
figure.savefig("campo_con_curva.png")
```

### Demo completa

```bash
venv/bin/python -m tita.main
```

Corre 3 ejemplos (uno por cada tipo de combinación relevante), imprime la clasificación y la solución de cada uno, y guarda el campo de direcciones de los dos de primer orden como PNG.

## Arquitectura

```
src/tita/
  exceptions.py   jerarquía de excepciones: ODEError, MethodNotApplicableError,
                  SingularityError, InvalidInputError
  validators.py   validación de entradas (orden, condiciones iniciales, rangos)
  utils.py        lógica compartida: substitute_function, linear_form,
                  solve_integration_constants
  parser.py       extraer f(x,y) o M(x,y)/N(x,y) de una Eq de sympy
  classifier.py   SOLO clasifica y verifica, nunca resuelve
  core.py         classify_ode / solve_ode: punto de entrada que orquesta
                  parser + classifier + analytic/
  systems.py      reducir EDO de orden n a sistema de primer orden (forma matricial)
  analytic/
    first_order.py          las 9 técnicas de primer orden
    integrating_factor.py   factor integrante para ecuaciones no exactas
    second_order.py         ecuación característica + solución homogénea
  plotting/
    direction_field.py   campo de direcciones
    solution_curves.py   curvas solución (+ superposición)
  main.py         script de demostración
tests/            un archivo de test por módulo, espejo 1:1 de src/tita/
```

**Regla de separación estricta:** `classifier.py` nunca resuelve; resolver vive en `analytic/` y `systems.py`; graficar vive en `plotting/`; `core.py` solo orquesta.

## Tests

```bash
venv/bin/python -m pytest -v                              # correr todo
venv/bin/python -m pytest --cov=tita --cov-report=term-missing  # cobertura
venv/bin/flake8 src tests                                  # estilo
```

121 tests, 100% de cobertura en el código de la librería (`main.py` es un script de demo, no se prueba con pytest).

Cada técnica de resolución se verifica sustituyendo la solución de vuelta en la ecuación original con `classifier.verify_solution` (que usa `sympy.checkodesol`), no solo comparando contra un resultado esperado — así se confirma matemáticamente que la solución es correcta, no solo que "se ve bien".
