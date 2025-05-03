# -*- coding: utf-8 -*-
"""
Module for computing various tensors in General Relativity using SymPy.

This module provides functions to calculate Christoffel symbols, the Riemann
curvature tensor, the Ricci tensor, the Ricci scalar, and the Einstein tensor
given a set of coordinates and a metric tensor defined in those coordinates.
It includes helper functions for simplification and options to display results
using LaTeX in environments like Jupyter notebooks or print them to the console.
"""

import sys
from typing import List, Optional, Tuple, Union, Sequence

# Check if running in an IPython environment for display capabilities
try:
    from IPython.display import display, Math
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False
    # Define dummy display/Math if not in IPython to avoid NameErrors
    # Users should rely on print_ functions in non-IPython environments.
    def display(*args, **kwargs): # pylint: disable=unused-argument
        """Dummy display function when IPython is not available."""
        print("IPython.display is not available. Use print_ functions instead.",
              file=sys.stderr)

    class Math: # pylint: disable=too-few-public-methods
        """Dummy Math class when IPython is not available."""
        def __init__(self, data: str):
            self.data = data
            print("IPython.display.Math is not available.", file=sys.stderr)

# Third-party imports
import sympy
from sympy import (
    symbols, Matrix, diff, sin, cos, exp, Function, log, Rational, simplify,
    trigsimp, Symbol, cancel, factor, ratsimp, powsimp, radsimp, latex, S
)

# Type aliases for clarity
SympyExpr = sympy.Expr
SympyMatrix = sympy.Matrix
CoordList = Sequence[sympy.Symbol]
Tensor4D = List[List[List[List[SympyExpr]]]]
Tensor3D = List[List[List[SympyExpr]]]
Tensor2D = List[List[SympyExpr]]


def _enhanced_simplify(expr: SympyExpr) -> SympyExpr:
    """Applies a sequence of SymPy simplification functions for better results.

    This helper function applies trigonometric simplification followed by
    general simplification, rational simplification, cancellation, factoring,
    and power simplification. The sequence is chosen empirically to handle
    expressions arising in GR calculations, especially with trigonometric
    functions from spherical coordinates. Some potentially counter-productive
    simplifications (like radsimp, factor, powsimp in certain cases, or a
    final simplify) are commented out but can be re-enabled for experimentation.

    Args:
        expr: The SymPy expression to simplify.

    Returns:
        The simplified SymPy expression.
    """
    simplified_expr = trigsimp(expr)
    simplified_expr = simplify(simplified_expr)
    # simplified_expr = radsimp(simplified_expr) # Can sometimes complicate
    simplified_expr = cancel(simplified_expr)
    simplified_expr = ratsimp(simplified_expr)
    # Factor and powsimp might sometimes make expressions look more complex
    # depending on the desired form.
    simplified_expr = factor(simplified_expr)
    simplified_expr = powsimp(simplified_expr, force=True)
    simplified_expr = trigsimp(simplified_expr)
    simplified_expr = cancel(simplified_expr)
    simplified_expr = trigsimp(simplified_expr)
    # A final simplify pass might undo some specific forms achieved earlier.
    # simplified_expr = simplify(simplified_expr)
    return simplified_expr


def compute_christoffel_symbols(
    coords: CoordList,
    metric: SympyMatrix
) -> Tensor3D:
    """Computes the Christoffel symbols of the second kind (Gamma^rho_{mu nu}).

    Calculates Gamma^rho_{mu nu} = (1/2) * g^{rho sigma} * (
        d(g_{nu sigma})/dx^mu + d(g_{mu sigma})/dx^nu - d(g_{mu nu})/dx^sigma
    ). Applies enhanced simplification to each component.

    Args:
        coords: A list or sequence of SymPy symbols representing the coordinates
            (e.g., [t, r, theta, phi]).
        metric: A SymPy Matrix representing the metric tensor g_{mu nu}.

    Returns:
        A 3D list (list of lists of lists) representing the Christoffel symbols
        Gamma^rho_{mu nu}, where the indices correspond to [rho][mu][nu].
        Each element is a simplified SymPy expression.

    Raises:
        TypeError: If the metric is not invertible.
    """
    dim = len(coords)
    try:
        inv_metric = metric.inv()
    except Exception as e:
        raise TypeError(f"Metric inversion failed: {e}. Metric:\n{metric}") from e

    # Pre-calculate all first derivatives of the metric tensor components.
    # metric_diffs[i][j][k] = d(g_{ij}) / dx^k
    metric_diffs = [
        [
            [_enhanced_simplify(diff(metric[i, j], coords[k])) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]

    # Initialize Christoffel symbols tensor with zeros.
    christoffel_symbols: Tensor3D = [
        [[S.Zero for _ in range(dim)] for _ in range(dim)] for _ in range(dim)
    ]

    # Compute components using the standard formula.
    for rho in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                temp_sum = S.Zero
                for sigma in range(dim):
                    d_mu_g_nu_sigma = metric_diffs[nu][sigma][mu]
                    d_nu_g_mu_sigma = metric_diffs[mu][sigma][nu]
                    d_sigma_g_mu_nu = metric_diffs[mu][nu][sigma]

                    term = (d_mu_g_nu_sigma + d_nu_g_mu_sigma - d_sigma_g_mu_nu)
                    temp_sum += term * inv_metric[rho, sigma]

                # Apply enhanced simplification to the final component.
                christoffel_symbols[rho][mu][nu] = _enhanced_simplify(temp_sum / 2)

    return christoffel_symbols


def display_christoffel_symbols(
    christoffel_symbols: Tensor3D,
    coords: CoordList
) -> None:
    """Displays non-zero Christoffel symbols using LaTeX via IPython.display.

    Formats the non-zero components as Gamma^rho_{mu nu} = value in LaTeX.
    Exploits the symmetry Gamma^rho_{mu nu} = Gamma^rho_{nu mu} by default,
    only iterating through mu <= nu. Includes a check for potential asymmetry
    (which indicates a calculation error if found). Requires an IPython
    environment (like Jupyter).

    Args:
        christoffel_symbols: The 3D list representing Gamma^rho_{mu nu}.
        coords: The list of coordinate symbols used for LaTeX labels.
    """
    if not _HAS_IPYTHON:
        print("IPython environment not detected. Cannot display LaTeX.",
              file=sys.stderr)
        print("Use print_christoffel_symbols() instead.", file=sys.stderr)
        return

    dim = len(coords)
    found_non_zero = False
    print("\n--- Christoffel Symbols Γ^ρ_{μν} ---")

    for rho in range(dim):
        for mu in range(dim):
            # Exploit symmetry: Gamma^rho_{mu,nu} = Gamma^rho_{nu,mu}
            # Only compute and display for mu <= nu
            for nu in range(mu, dim):
                value = christoffel_symbols[rho][mu][nu]

                # Check the explicitly calculated symmetric component if mu != nu
                # This helps verify the computation's internal consistency.
                if mu != nu:
                    symmetric_value = christoffel_symbols[rho][nu][mu]
                    # Check for potential calculation asymmetry
                    if value != symmetric_value:
                        print(
                            f"Warning: Asymmetry detected for "
                            f"Γ^{{{latex(coords[rho])}}} ({latex(coords[mu])},"
                            f"{latex(coords[nu])}) = {latex(value)} vs "
                            f"({latex(coords[nu])},{latex(coords[mu])}) = "
                            f"{latex(symmetric_value)}"
                        )

                if value != S.Zero:
                    found_non_zero = True
                    # Construct LaTeX string using robust latex() function
                    rho_l = latex(coords[rho])
                    mu_l = latex(coords[mu])
                    nu_l = latex(coords[nu])
                    val_l = latex(value)
                    latex_str = f"\\Gamma^{{{rho_l}}}_{{{{{mu_l} {nu_l}}}}} = {val_l}"
                    display(Math(latex_str))

    if not found_non_zero:
        print('All Christoffel symbols are zero.')


def print_christoffel_symbols(
    christoffel_symbols: Tensor3D,
    coords: CoordList
) -> None:
    """Prints non-zero Christoffel symbols to the console.

    Formats the non-zero components as Gamma^rho(mu, nu) = value.
    Exploits the symmetry Gamma^rho_{mu nu} = Gamma^rho_{nu mu} by default,
    only iterating through mu <= nu. Includes a check and warning for
    potential asymmetry. Uses coordinate names for printing.

    Args:
        christoffel_symbols: The 3D list representing Gamma^rho_{mu nu}.
        coords: The list of coordinate symbols used for printing labels.
    """
    dim = len(coords)
    coord_latex = [latex(c) for c in coords] # Use latex for consistency
    found_non_zero = False
    print("\n--- Christoffel Symbols Γ^ρ_{μν} ---")

    for rho in range(dim):
        for mu in range(dim):
            # Exploit symmetry: Gamma^rho_{mu,nu} = Gamma^rho_{nu,mu}
            for nu in range(mu, dim):
                value = christoffel_symbols[rho][mu][nu]
                if value != S.Zero:
                    found_non_zero = True
                    # Symbols are assumed already simplified during computation
                    print(f'Γ^{coord_latex[rho]} ( {coord_latex[mu]}, {coord_latex[nu]} ) = {value}')

                    # If mu != nu, check the symmetric component for verification
                    if mu != nu:
                        symmetric_value = christoffel_symbols[rho][nu][mu]
                        if symmetric_value != value:
                             print(f'Warning: Asymmetry detected! '
                                   f'Γ^{coord_latex[rho]} ({coord_latex[nu]},{coord_latex[mu]}) '
                                   f'= {symmetric_value} (Expected {value})')

    if not found_non_zero:
        print('All Christoffel symbols are zero.')


def compute_riemann_curvature_tensor(
    coords: CoordList,
    metric: SympyMatrix,
    christoffel_symbols: Optional[Tensor3D] = None
) -> Tensor4D:
    """Computes the Riemann curvature tensor (type 1, 3) R^rho_{sigma mu nu}.

    Calculates R^rho_{sigma mu nu} = d(Gamma^rho_{nu sigma})/dx^mu
        - d(Gamma^rho_{mu sigma})/dx^nu
        + Gamma^rho_{mu lambda} * Gamma^lambda_{nu sigma}
        - Gamma^rho_{nu lambda} * Gamma^lambda_{mu sigma}.
    Applies enhanced simplification to each component.

    Args:
        coords: A list or sequence of SymPy symbols representing the coordinates.
        metric: A SymPy Matrix representing the metric tensor g_{mu nu}. Used
            only if christoffel_symbols are not provided.
        christoffel_symbols: Optional pre-computed Christoffel symbols
            (Gamma^rho_{mu nu}) as a 3D list. If None, they will be computed.

    Returns:
        A 4D list (list of lists of lists of lists) representing the Riemann
        tensor R^rho_{sigma mu nu}, where indices correspond to
        [rho][sigma][mu][nu]. Each element is a simplified SymPy expression.

    Raises:
        TypeError: If computation of Christoffel symbols fails (e.g., due
                   to non-invertible metric).
    """
    dim = len(coords)
    if christoffel_symbols is None:
        # Compute Christoffel symbols if not provided
        christoffel_symbols = compute_christoffel_symbols(coords, metric)

    # Pre-calculate derivatives of Christoffel symbols.
    # christoffel_diffs[rho][mu][nu][k] = d(Gamma^rho_{mu nu}) / dx^k
    christoffel_diffs = [
        [
            [
                [_enhanced_simplify(diff(christoffel_symbols[rho][mu][nu], coords[k]))
                 for k in range(dim)]
                for nu in range(dim)
            ]
            for mu in range(dim)
        ]
        for rho in range(dim)
    ]

    # Initialize Riemann tensor with zeros.
    riemann_tensor: Tensor4D = [
        [[[S.Zero for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        for _ in range(dim)
    ]

    # Compute components using the standard formula.
    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    term1 = christoffel_diffs[rho][nu][sigma][mu] # d(G^rho_nu_sigma)/dx^mu
                    term2 = christoffel_diffs[rho][mu][sigma][nu] # d(G^rho_mu_sigma)/dx^nu

                    term3 = S.Zero
                    term4 = S.Zero
                    for lam in range(dim): # Sum over lambda
                        term3 += (christoffel_symbols[rho][mu][lam] *
                                  christoffel_symbols[lam][nu][sigma])
                        term4 += (christoffel_symbols[rho][nu][lam] *
                                  christoffel_symbols[lam][mu][sigma])

                    # Apply enhanced simplification to the final component.
                    riemann_tensor[rho][sigma][mu][nu] = _enhanced_simplify(
                        term1 - term2 + term3 - term4
                    )

    return riemann_tensor


def display_riemann_curvature_tensor(
    riemann_tensor: Tensor4D,
    coords: CoordList
) -> None:
    """Displays non-zero Riemann tensor components using LaTeX via IPython.display.

    Formats non-zero R^rho_{sigma mu nu} = value in LaTeX.
    Note: The Riemann tensor has anti-symmetry in the last two indices:
    R^rho_{sigma mu nu} = -R^rho_{sigma nu mu}. This function displays all
    non-zero components as calculated. Requires an IPython environment.

    Args:
        riemann_tensor: The 4D list representing R^rho_{sigma mu nu}.
        coords: The list of coordinate symbols used for LaTeX labels.
    """
    if not _HAS_IPYTHON:
        print("IPython environment not detected. Cannot display LaTeX.",
              file=sys.stderr)
        print("Use print_riemann_curvature_tensor() instead.", file=sys.stderr)
        return

    dim = len(coords)
    found_non_zero = False
    print("\n--- Riemann Curvature Tensor R^ρ_{σ μ ν} ---")

    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    value = riemann_tensor[rho][sigma][mu][nu]
                    if value != S.Zero:
                        found_non_zero = True
                        # Construct LaTeX string
                        rho_l = latex(coords[rho])
                        sig_l = latex(coords[sigma])
                        mu_l = latex(coords[mu])
                        nu_l = latex(coords[nu])
                        val_l = latex(value)
                        latex_str = (f"R^{{{rho_l}}}_{{{{{sig_l} {mu_l} {nu_l}}}}} "
                                     f"= {val_l}")
                        display(Math(latex_str))
                        # Optional: Check anti-symmetry explicitly
                        # if mu < nu:
                        #     anti_sym_val = riemann_tensor[rho][sigma][nu][mu]
                        #     if value != -anti_sym_val:
                        #          print(f"Warning: Anti-symmetry check failed for "
                        #                f"R^{{{rho_l}}}_{{{sig_l}{mu_l}{nu_l}}} vs "
                        #                f"R^{{{rho_l}}}_{{{sig_l}{nu_l}{mu_l}}}")

    if not found_non_zero:
        print('All components of the Riemann curvature tensor are zero.')


def print_riemann_curvature_tensor(
    riemann_tensor: Tensor4D,
    coords: CoordList
) -> None:
    """Prints non-zero Riemann tensor components R^rho_{sigma mu nu} to console.

    Formats non-zero components as R^rho(_sigma, mu, nu) = value.
    Uses coordinate names for printing. Displays all non-zero components.

    Args:
        riemann_tensor: The 4D list representing R^rho_{sigma mu nu}.
        coords: The list of coordinate symbols used for printing labels.
    """
    dim = len(coords)
    coord_latex = [latex(c) for c in coords] # Use latex for consistency
    found_non_zero = False
    print("\n--- Riemann Curvature Tensor R^ρ_{σ μ ν} ---")

    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    value = riemann_tensor[rho][sigma][mu][nu]
                    if value != S.Zero:
                        found_non_zero = True
                        # Symbols assumed already simplified during computation
                        print(f'R^{coord_latex[rho]} ('
                              f'_{coord_latex[sigma]}, {coord_latex[mu]}, {coord_latex[nu]}'
                              f') = {value}')

    if not found_non_zero:
        print('All components of the Riemann curvature tensor are zero.')


def compute_ricci_tensor(
    coords: CoordList,
    metric: SympyMatrix,
    riemann_tensor: Optional[Tensor4D] = None
) -> Tensor2D:
    """Computes the Ricci curvature tensor R_{mu nu}.

    Calculates R_{mu nu} = R^rho_{mu rho nu} (contraction of the Riemann tensor).
    Applies enhanced simplification to each component.

    Args:
        coords: A list or sequence of SymPy symbols representing the coordinates.
        metric: A SymPy Matrix representing the metric tensor g_{mu nu}. Used
            only if riemann_tensor is not provided.
        riemann_tensor: Optional pre-computed Riemann tensor R^rho_{sigma mu nu}
            as a 4D list. If None, it will be computed (which also requires
            computing Christoffel symbols).

    Returns:
        A 2D list (list of lists) representing the Ricci tensor R_{mu nu},
        where indices correspond to [mu][nu]. Each element is a simplified
        SymPy expression.

    Raises:
        TypeError: If computation of the Riemann tensor (or underlying
                   Christoffel symbols) fails.
    """
    dim = len(coords)
    if riemann_tensor is None:
        # Note: This implicitly computes Christoffel symbols if needed.
        riemann_tensor = compute_riemann_curvature_tensor(coords, metric)

    # Initialize Ricci tensor with zeros.
    ricci_tensor: Tensor2D = [[S.Zero for _ in range(dim)] for _ in range(dim)]

    # Compute components by contracting the Riemann tensor: R_{mu nu} = R^rho_{mu rho nu}
    for mu in range(dim):
        for nu in range(dim):
            term = S.Zero
            for rho in range(dim): # Sum over rho
                term += riemann_tensor[rho][mu][rho][nu]
            ricci_tensor[mu][nu] = _enhanced_simplify(term)

    return ricci_tensor


def display_ricci_tensor(ricci_tensor: Tensor2D, coords: CoordList) -> None:
    """Displays non-zero Ricci tensor components using LaTeX via IPython.display.

    Formats non-zero R_{mu nu} = value in LaTeX.
    Exploits the symmetry R_{mu nu} = R_{nu mu} by default, only iterating
    through mu <= nu. Includes a check for potential asymmetry.
    Requires an IPython environment.

    Args:
        ricci_tensor: The 2D list representing R_{mu nu}.
        coords: The list of coordinate symbols used for LaTeX labels.
    """
    if not _HAS_IPYTHON:
        print("IPython environment not detected. Cannot display LaTeX.",
              file=sys.stderr)
        print("Use print_ricci_tensor() instead.", file=sys.stderr)
        return

    dim = len(coords)
    found_non_zero = False
    print("\n--- Ricci Tensor R_{μν} ---")

    for mu in range(dim):
        # Exploit symmetry R_{mu,nu} = R_{nu,mu}
        for nu in range(mu, dim):
            value = ricci_tensor[mu][nu]

            # Check symmetric component for potential calculation errors
            if mu != nu:
                symmetric_value = ricci_tensor[nu][mu]
                if value != symmetric_value:
                    print(
                        f"Warning: Asymmetry detected for R("
                        f"{latex(coords[mu])},{latex(coords[nu])}) = {latex(value)} vs "
                        f"({latex(coords[nu])},{latex(coords[mu])}) = "
                        f"{latex(symmetric_value)}"
                    )

            if value != S.Zero:
                found_non_zero = True
                # Construct LaTeX string
                mu_l = latex(coords[mu])
                nu_l = latex(coords[nu])
                val_l = latex(value)
                latex_str = f"R_{{{{{mu_l} {nu_l}}}}} = {val_l}"
                display(Math(latex_str))

    if not found_non_zero:
        print('All components of the Ricci curvature tensor are zero.')


def print_ricci_tensor(ricci_tensor: Tensor2D, coords: CoordList) -> None:
    """Prints non-zero Ricci tensor components R_{mu nu} to the console.

    Formats non-zero components as R(mu, nu) = value.
    Exploits the symmetry R_{mu nu} = R_{nu mu} by default, iterating through
    mu <= nu. Includes a check for potential asymmetry. Uses coordinate names.

    Args:
        ricci_tensor: The 2D list representing R_{mu nu}.
        coords: The list of coordinate symbols used for printing labels.
    """
    dim = len(coords)
    coord_latex = [latex(c) for c in coords] # Use latex for consistency
    found_non_zero = False
    print("\n--- Ricci Tensor R_{μν} ---")

    for mu in range(dim):
        # Exploit symmetry R_{mu,nu} = R_{nu,mu}
        for nu in range(mu, dim):
            value = ricci_tensor[mu][nu]
            if value != S.Zero:
                found_non_zero = True
                # Symbols assumed already simplified during computation
                print(f'R({coord_latex[mu]}, {coord_latex[nu]}) = {value}')
                if mu != nu:
                    symmetric_value = ricci_tensor[nu][mu]
                    if symmetric_value != value:
                         print(f'Warning: Asymmetry detected! '
                               f'R({coord_latex[nu]}, {coord_latex[mu]}) = '
                               f'{symmetric_value} (Expected {value})')

    if not found_non_zero:
        print('All components of the Ricci curvature tensor are zero.')


def compute_ricci_scalar(
    coords: CoordList,
    metric: SympyMatrix,
    ricci_tensor: Optional[Tensor2D] = None
) -> Union[SympyExpr, None]:
    """Computes the Ricci scalar R.

    Calculates R = g^{mu nu} R_{mu nu} (contraction of the Ricci tensor with
    the inverse metric). Applies enhanced simplification.

    Args:
        coords: A list or sequence of SymPy symbols representing the coordinates.
                Used only if ricci_tensor is not provided.
        metric: A SymPy Matrix representing the metric tensor g_{mu nu}. Used
                to compute the inverse metric and potentially the Ricci tensor.
        ricci_tensor: Optional pre-computed Ricci tensor R_{mu nu} as a 2D
                      list. If None, it will be computed.

    Returns:
        The simplified Ricci scalar R as a SymPy expression, or None if the
        metric is not invertible.

    Raises:
        TypeError: If computation of the underlying Ricci tensor fails (when
                   it needs to be computed).
    """
    dim = len(coords)
    if ricci_tensor is None:
        # This computes Riemann and Christoffel if needed.
        ricci_tensor = compute_ricci_tensor(coords, metric)

    try:
        inv_metric = metric.inv()
    except Exception as e:
        print(f"Error inverting metric: {e}", file=sys.stderr)
        # Cannot compute Ricci scalar without the inverse metric.
        return None

    ricci_scalar = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            ricci_scalar += inv_metric[mu, nu] * ricci_tensor[mu][nu]

    return _enhanced_simplify(ricci_scalar)


def display_ricci_scalar(ricci_scalar_value: Optional[SympyExpr]) -> None:
    """Displays the Ricci scalar using LaTeX via IPython.display.

    Formats the result as R = value in LaTeX. Requires an IPython environment.

    Args:
        ricci_scalar_value: The computed Ricci scalar (a SymPy expression) or
                           None if computation failed.
    """
    if not _HAS_IPYTHON:
        print("IPython environment not detected. Cannot display LaTeX.",
              file=sys.stderr)
        print("Use print() for the scalar value instead.", file=sys.stderr)
        return

    print("\n--- Ricci Scalar R ---")
    if ricci_scalar_value is not None:
        display(Math(f"R = {latex(ricci_scalar_value)}"))
    else:
        print("Ricci scalar computation failed or value is None.")


def compute_einstein_tensor(
    coords: CoordList,
    metric: SympyMatrix,
    ricci_tensor: Optional[Tensor2D] = None,
    ricci_scalar: Optional[SympyExpr] = None
) -> Union[Tensor2D, None]:
    """Computes the Einstein tensor G_{mu nu}.

    Calculates G_{mu nu} = R_{mu nu} - (1/2) * g_{mu nu} * R.
    Applies enhanced simplification to each component.

    Args:
        coords: A list or sequence of SymPy symbols representing the coordinates.
                Used only if Ricci tensor/scalar are not provided.
        metric: A SymPy Matrix representing the metric tensor g_{mu nu}.
        ricci_tensor: Optional pre-computed Ricci tensor R_{mu nu} as a 2D
                      list. If None, it will be computed.
        ricci_scalar: Optional pre-computed Ricci scalar R as a SymPy
                      expression. If None, it will be computed. Note that
                      computing the scalar might require computing the Ricci
                      tensor first if it's not also provided.

    Returns:
        A 2D list (list of lists) representing the Einstein tensor G_{mu nu},
        where indices correspond to [mu][nu]. Each element is a simplified
        SymPy expression. Returns None if the Ricci scalar computation fails
        (e.g., due to non-invertible metric).

    Raises:
        TypeError: If computation of the underlying Ricci tensor fails (when
                   it needs to be computed).
    """
    dim = len(coords)

    # Compute Ricci tensor if not provided
    if ricci_tensor is None:
        # This call computes Christoffel and Riemann if needed.
        ricci_tensor = compute_ricci_tensor(coords, metric)

    # Compute Ricci scalar if not provided
    if ricci_scalar is None:
        # Pass the already computed/provided ricci_tensor to avoid re-computation
        ricci_scalar = compute_ricci_scalar(coords, metric, ricci_tensor=ricci_tensor)
        # Check if scalar computation failed (e.g., metric inversion)
        if ricci_scalar is None:
             print("Cannot compute Einstein tensor because Ricci scalar "
                   "computation failed.", file=sys.stderr)
             return None

    # Initialize Einstein tensor with zeros.
    einstein_tensor: Tensor2D = [[S.Zero for _ in range(dim)] for _ in range(dim)]
    one_half = Rational(1, 2) # Use SymPy Rational for precision

    # Compute components using the formula G_{mu nu} = R_{mu nu} - (1/2) g_{mu nu} R
    for mu in range(dim):
        for nu in range(dim):
            # Ensure ricci_tensor indices are valid (should be if logic is correct)
            ricci_component = ricci_tensor[mu][nu]
            metric_component = metric[mu, nu]

            term = ricci_component - one_half * metric_component * ricci_scalar
            einstein_tensor[mu][nu] = _enhanced_simplify(term)

    return einstein_tensor


def display_einstein_tensor(
    einstein_tensor: Optional[Tensor2D],
    coords: CoordList
) -> None:
    """Displays non-zero Einstein tensor components using LaTeX via IPython.display.

    Formats non-zero G_{mu nu} = value in LaTeX.
    Exploits the symmetry G_{mu nu} = G_{nu mu} by default, only iterating
    through mu <= nu. Includes a check for potential asymmetry.
    Requires an IPython environment.

    Args:
        einstein_tensor: The 2D list representing G_{mu nu}, or None if
                         computation failed.
        coords: The list of coordinate symbols used for LaTeX labels.
    """
    if not _HAS_IPYTHON:
        print("IPython environment not detected. Cannot display LaTeX.",
              file=sys.stderr)
        print("Use print_einstein_tensor() instead.", file=sys.stderr)
        return

    print("\n--- Einstein Tensor G_{μν} ---")
    if einstein_tensor is None:
        print("Einstein tensor computation failed or was not performed.")
        return

    dim = len(coords)
    found_non_zero = False

    for mu in range(dim):
        # Exploit symmetry G_{mu,nu} = G_{nu,mu}
        for nu in range(mu, dim):
            value = einstein_tensor[mu][nu]

            # Check symmetric component for potential calculation errors
            if mu != nu:
                symmetric_value = einstein_tensor[nu][mu]
                if value != symmetric_value:
                    print(
                        f"Warning: Asymmetry detected for G("
                        f"{latex(coords[mu])},{latex(coords[nu])}) = {latex(value)} vs "
                        f"({latex(coords[nu])},{latex(coords[mu])}) = "
                        f"{latex(symmetric_value)}"
                    )

            if value != S.Zero:
                found_non_zero = True
                # Construct LaTeX string
                mu_l = latex(coords[mu])
                nu_l = latex(coords[nu])
                val_l = latex(value)
                latex_str = f"G_{{{{{mu_l} {nu_l}}}}} = {val_l}"
                display(Math(latex_str))

    if not found_non_zero:
        print('All components of the Einstein tensor are zero.')


def print_einstein_tensor(
    einstein_tensor: Optional[Tensor2D],
    coords: CoordList
) -> None:
    """Prints non-zero Einstein tensor components G_{mu nu} to the console.

    Formats non-zero components as G(mu, nu) = value.
    Exploits the symmetry G_{mu nu} = G_{nu mu} by default, iterating through
    mu <= nu. Includes a check for potential asymmetry. Uses coordinate names.

    Args:
        einstein_tensor: The 2D list representing G_{mu nu}, or None if
                         computation failed.
        coords: The list of coordinate symbols used for printing labels.
    """
    print("\n--- Einstein Tensor G_{μν} ---")
    if einstein_tensor is None:
        print("Einstein tensor computation failed or was not performed.")
        return

    dim = len(coords)
    coord_latex = [latex(c) for c in coords] # Use latex for consistency
    found_non_zero = False

    for mu in range(dim):
        # Exploit symmetry G_{mu,nu} = G_{nu,mu}
        for nu in range(mu, dim):
            value = einstein_tensor[mu][nu]
            if value != S.Zero:
                found_non_zero = True
                # Symbols assumed already simplified during computation
                print(f'G({coord_latex[mu]}, {coord_latex[nu]}) = {value}')
                if mu != nu:
                    symmetric_value = einstein_tensor[nu][mu]
                    if symmetric_value != value:
                         print(f'Warning: Asymmetry detected! '
                               f'G({coord_latex[nu]}, {coord_latex[mu]}) = '
                               f'{symmetric_value} (Expected {value})')

    if not found_non_zero:
        print('All components of the Einstein tensor are zero.')


# --- Example Usage ---
if __name__ == '__main__':

    # Check if we can use display (requires IPython)
    use_display = _HAS_IPYTHON

    # --- Example 1: 2D Polar Coordinates (Flat Space) ---
    print("="*40)
    print(" Example 1: 2D Polar Coordinates (Flat Space) ")
    print("="*40)

    # Define coordinates
    r_pol, phi_pol = symbols('r phi', positive=True) # Assume r > 0
    coords_polar: CoordList = [r_pol, phi_pol]

    # Define metric tensor components for polar coordinates
    g_rr_pol = 1
    g_phiphi_pol = r_pol**2
    metric_polar: SympyMatrix = Matrix([[g_rr_pol, 0], [0, g_phiphi_pol]])

    print("\nMetric Tensor g_{μν}:")
    if use_display:
        display(Math(latex(metric_polar)))
    else:
        sympy.pprint(metric_polar) # Use pretty print if no display

    # Compute Christoffel symbols
    gamma_polar = compute_christoffel_symbols(coords_polar, metric_polar)
    if use_display:
        display_christoffel_symbols(gamma_polar, coords_polar)
    else:
        print_christoffel_symbols(gamma_polar, coords_polar)

    # Compute Riemann tensor (should be zero for flat space)
    riemann_polar = compute_riemann_curvature_tensor(
        coords_polar, metric_polar, christoffel_symbols=gamma_polar
    )
    if use_display:
        display_riemann_curvature_tensor(riemann_polar, coords_polar)
    else:
        print_riemann_curvature_tensor(riemann_polar, coords_polar)

    # Compute Ricci tensor (should be zero)
    ricci_polar = compute_ricci_tensor(
        coords_polar, metric_polar, riemann_tensor=riemann_polar
    )
    if use_display:
        display_ricci_tensor(ricci_polar, coords_polar)
    else:
        print_ricci_tensor(ricci_polar, coords_polar)

    # Compute Ricci scalar (should be zero)
    ricci_scalar_polar = compute_ricci_scalar(
        coords_polar, metric_polar, ricci_tensor=ricci_polar
    )
    if use_display:
        display_ricci_scalar(ricci_scalar_polar)
    else:
        print("\n--- Ricci Scalar R ---")
        print(f"R = {ricci_scalar_polar}")


    # Compute Einstein tensor (should be zero)
    einstein_polar = compute_einstein_tensor(
        coords_polar, metric_polar,
        ricci_tensor=ricci_polar, ricci_scalar=ricci_scalar_polar
    )
    if use_display:
        display_einstein_tensor(einstein_polar, coords_polar)
    else:
        print_einstein_tensor(einstein_polar, coords_polar)

    print("\n" + "="*40 + "\n")

    # --- Example 2: 3D Spherical Coordinates (Flat Space) ---
    print("="*40)
    print(" Example 2: 3D Spherical Coordinates (Flat Space) ")
    print("="*40)

    # Define coordinates
    r_sph, theta_sph, phi_sph = symbols('r theta phi', positive=True) # r>0
    coords_sph: CoordList = [r_sph, theta_sph, phi_sph]

    # Define metric tensor components for spherical coordinates
    g_rr_sph = 1
    g_thetatheta_sph = r_sph**2
    g_phiphi_sph = r_sph**2 * sin(theta_sph)**2
    metric_sph: SympyMatrix = Matrix([
        [g_rr_sph, 0, 0],
        [0, g_thetatheta_sph, 0],
        [0, 0, g_phiphi_sph]
    ])

    print("\nMetric Tensor g_{μν}:")
    if use_display:
        display(Math(latex(metric_sph)))
    else:
        sympy.pprint(metric_sph)

    # Compute Christoffel symbols
    gamma_sph = compute_christoffel_symbols(coords_sph, metric_sph)
    if use_display:
        display_christoffel_symbols(gamma_sph, coords_sph)
    else:
        print_christoffel_symbols(gamma_sph, coords_sph)

    # Compute Riemann tensor (should be zero)
    riemann_sph = compute_riemann_curvature_tensor(
        coords_sph, metric_sph, christoffel_symbols=gamma_sph
    )
    if use_display:
        display_riemann_curvature_tensor(riemann_sph, coords_sph)
    else:
        print_riemann_curvature_tensor(riemann_sph, coords_sph)

    # Compute Ricci tensor (should be zero)
    ricci_sph = compute_ricci_tensor(
        coords_sph, metric_sph, riemann_tensor=riemann_sph
    )
    if use_display:
        display_ricci_tensor(ricci_sph, coords_sph)
    else:
        print_ricci_tensor(ricci_sph, coords_sph)

    # Compute Ricci scalar (should be zero)
    ricci_scalar_sph = compute_ricci_scalar(
        coords_sph, metric_sph, ricci_tensor=ricci_sph
    )
    if use_display:
        display_ricci_scalar(ricci_scalar_sph)
    else:
        print("\n--- Ricci Scalar R ---")
        print(f"R = {ricci_scalar_sph}")


    # Compute Einstein tensor (should be zero)
    einstein_sph = compute_einstein_tensor(
        coords_sph, metric_sph,
        ricci_tensor=ricci_sph, ricci_scalar=ricci_scalar_sph
    )
    if use_display:
        display_einstein_tensor(einstein_sph, coords_sph)
    else:
        print_einstein_tensor(einstein_sph, coords_sph)

    print("\n" + "="*40)