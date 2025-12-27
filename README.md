# Sofiev Plume Rise Model

This project provides a Python implementation of the Sofiev plume rise model, a box model for simulating the rise of plumes from biomass burning. The model is based on the paper:

*Sofiev, M., Ermakova, T., & Vankevich, R. (2012). A box model for the plume rise of forest fire emissions. Atmos. Chem. Phys., 12(4), 1995–2006. https://doi.org/10.5194/acp-12-1995-2012*

## Installation

To install the necessary dependencies, you can use pip with the provided `pyproject.toml` file:

```bash
pip install .
```

For development, you can install the optional dependencies, which include tools for linting and testing:

```bash
pip install .[dev]
```

## Usage

The model can be run from the command line, though the specifics of the command-line interface are still under development.

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with a descriptive commit message.
4.  Push your changes to your fork.
5.  Create a pull request to the main repository.

Before submitting a pull request, please ensure that your code adheres to the coding standards by running the linter and that all tests pass.
