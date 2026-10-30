# An export of RoboCompiler: the six-UPS Stewart platform

The three JSON files in this folder were written, unchanged, by the compile step of RoboCompiler for its six-UPS Stewart platform:

- `physical_graph.json`: the input, the physical joint graph (bodies, joints, actuators);
- `compiled_model.json`: the compiled tree, its point closures and its coordinates;
- `topology_plan.json`: the plan of the compilation.

Source: RoboCompiler, https://github.com/Mehdi-Heydari-Shahna/RoboCompiler, commit `a4e4437c18cbc4485b346e71674a9d26cd261be4` (10 October 2026), by Mehdi Heydari Shahna, Joongheon Kim and Jouni Mattila. Command, run in `Stewart_CMG_PACDM/workflows/compiler` with NumPy 2.5.3 and SciPy 1.18.1 (the repository pins NumPy 2.3.5 and SciPy 1.17.0 for its recorded measurements; two runs gave identical files):

```
python3 compile_physical_graph.py --input inputs/physical_graph.json --out OUT
```

RoboCompiler is licensed under the Apache License, Version 2.0; a copy is in `LICENSE`. The files are reproduced here under that license so that `../../model_providers.py` runs without RoboCompiler or NumPy; `../robocompiler.py` reads them with the standard library.
