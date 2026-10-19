# Physical AI, Together

**Certified Control and Learning for Robotics**

Companion code and numerical results for the book by Mehdi Heydari Shahna.

## Available material

| Chapter | Example | Code and results |
| --- | --- | --- |
| 3 | The smallest complete example: a cart approaching a wall | [Implementation and instructions](code/ch03/README.md), [numerical results](code/ch03/results/) |

The cart example implements the five ALLOVI modules and the viability governor. It compares fixed, reactive and forecasting capability predictors, then examines a persisting predictor whose capability labels become invalid. These are deterministic simulations of the stated cart model.

## Reproduce Chapter 3

Python 3.12 is the reference environment. The numerical example uses only the Python standard library.

From the repository root:

```text
python code/ch03/cart_example.py --check
```

The command regenerates the files in `code/ch03/results/` and checks 110 reported numerical values. The final line should be:

```text
[check] 110 quoted values: all as quoted
```

To inspect the candidates considered in period 32:

```text
python code/ch03/cart_example.py --period 32
```

See the [Chapter 3 README](code/ch03/README.md) for configurations, output columns and units. Numerical agreement with the supplied outputs is a reproducibility check; the general guarantees depend on their stated mathematical assumptions.

## Miniforge environment

```text
conda env create -f environment.yml
conda activate physical-ai-together
python code/ch03/cart_example.py --check
```

The environment also includes Git and the GitHub CLI for publishing updates.

## Repository layout

```text
code/
  ch03/
    cart_example.py
    README.md
    results/
environment.yml
CITATION.cff
```

Additional chapter folders will be added under `code/`. Book text and illustration sources are maintained separately. This repository currently provides the numerical code and data for Chapter 3.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).
