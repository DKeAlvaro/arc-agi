# ARC Project

This project is an attempt to solve the Abstraction and Reasoning Corpus (ARC) challenge.

## Code Structure

*   `main.py`: The main entry point for running the project. It loads the data, initializes the model, and starts the evaluation process.

*   `data.py`: Contains helper functions and paths for loading the ARC dataset in JSON format.

*   `models.py`: Defines the models used to solve the ARC tasks.
    *   `ARCModel`: An abstract base class for all models.
    *   `DummyModel`: A simple baseline model that returns a grid of zeros.
    *   `ProgramSynthesisModel`: A more advanced model that uses Breadth-First Search (BFS) to find a sequence of transformations (a "program") that solves a given ARC task.

*   `evaluate.py`: Contains the `make_predictions` function, which evaluates a given model against a set of challenges and reports accuracy metrics. It can also generate a PDF with visualizations of the model's predictions.

*   `utils.py`: Provides utility functions, such as `plot_pairs` for visualizing input-output pairs and `get_small_sample` for sampling the data.

*   `data/`: This directory contains the ARC dataset files in JSON format.

*   `predictions/`: This directory is used to store the output of the model's predictions, including a PDF file with visualizations.
