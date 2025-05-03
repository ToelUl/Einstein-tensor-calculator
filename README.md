# Einstein-tensor-calculator


A Python module using SymPy to symbolically compute the Einstein tensor, Ricci tensor/scalar, Riemann curvature tensor, and Christoffel symbols for user-defined metrics in General Relativity.

## Motivation

Calculating tensors in General Relativity (GR) by hand is tedious, time-consuming, and prone to errors, especially for complex metrics. This project aims to provide a simple yet effective tool to automate these symbolic calculations, aiding students, educators, and researchers in GR.

## Features

* Calculates the following tensors symbolically:
    * Christoffel Symbols (of the second kind, $\Gamma^\rho_{\mu\nu}$)
    * Riemann Curvature Tensor ($R^\rho{}_{\sigma\mu\nu}$)
    * Ricci Tensor ($R_{\mu\nu}$)
    * Ricci Scalar ($R$)
    * Einstein Tensor ($G_{\mu\nu}$)
* Uses [SymPy](https://www.sympy.org/) for all symbolic manipulations.
* Accepts user-defined coordinates and metric tensors as SymPy objects.
* Includes helper functions for simplifying expressions and displaying results (using LaTeX via IPython or printing to console).

## Run examples on Google Colab

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Einstein-tensor-calculator/blob/main/2D_sphere.ipynb) **2D Sphere**
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Einstein-tensor-calculator/blob/main/3D_sphere.ipynb) **3D Sphere**
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Einstein-tensor-calculator/blob/main/Static_Spherically_Symmetric.ipynb) **Static Spherically Symmetric**
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Einstein-tensor-calculator/blob/main/Schwarzschild.ipynb) **Schwarzschild**
- If you have the courage and patience (or just have a lot of time), try this ↓↓↓
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Einstein-tensor-calculator/blob/main/Kerr_Boyer_Lindquist.ipynb) **Kerr Boyer-Lindquist**
    
## Usage

Here's a basic example of calculating the Christoffel symbols for a 2D sphere:

```python
import sympy
from sympy import symbols, Matrix, sin, latex
from IPython.display import display, Math # Optional for rich display

# Assuming the module's functions are in 'gr.py' or similar
# Adjust the import based on your project structure
from gr import compute_christoffel_symbols, display_christoffel_symbols, print_christoffel_symbols

# 1. Define Coordinates and Parameters
theta, phi = symbols("θ φ", real=True)
coords_2d = [theta, phi]
R = symbols("R", real=True, positive=True) # Radius of the sphere

# 2. Define the Metric Tensor (2D Sphere)
metric_sphere_2d = Matrix([
    [ R**2,               0     ],
    [  0,   R**2* sin(theta)**2 ]
])

print("Metric Tensor:")
display(Math(latex(metric_sphere_2d))) # Use display if in Jupyter/IPython

# 3. Compute Christoffel Symbols
print("\nCalculating Christoffel Symbols...")
christoffel_symbols = compute_christoffel_symbols(coords_2d, metric_sphere_2d)

# 4. Display Results
# Option A: Rich display using LaTeX (requires IPython/Jupyter)
print("\nDisplaying non-zero Christoffel Symbols (LaTeX):")
display_christoffel_symbols(christoffel_symbols, coords_2d)

# Option B: Print to console
# print("\nPrinting non-zero Christoffel Symbols:")
# print_christoffel_symbols(christoffel_symbols, coords_2d)

# Similarly, you can compute other tensors:
# riemann = compute_riemann_curvature_tensor(coords_2d, metric_sphere_2d, christoffel_symbols)
# ricci = compute_ricci_tensor(coords_2d, metric_sphere_2d, riemann_tensor=riemann)
# scalar = compute_ricci_scalar(coords_2d, metric_sphere_2d, ricci_tensor=ricci)
# einstein = compute_einstein_tensor(coords_2d, metric_sphere_2d, ricci_tensor=ricci, ricci_scalar=scalar)
# ... and use display_* or print_* functions for them.