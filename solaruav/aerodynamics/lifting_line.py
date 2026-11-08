"""Lifting-line solution for spanwise lift and induced drag (Fourier method).

Purpose
-------
Solve Prandtl's lifting-line equation for one symmetric lifting surface using
the Fourier (Glauert) series for the circulation. This gives the spanwise
lift distribution, total lift, induced drag, span efficiency and the
surface's pitching moment.

Method
------
The spanwise coordinate is written as s = -(b/2) cos(theta), so the tips are at
theta = 0 and pi and the root is at theta = pi/2. The circulation is expanded as

    Gamma(theta) = 2 b V  sum_n  A_n sin(n theta).

For a symmetric surface at zero sideslip only odd n appear. Substituting into
the lifting-line equation and enforcing it at N collocation points on one
side gives N linear equations for the N coefficients A_n (the monoplane
equation):

    sum_n A_n sin(n theta) (n mu + sin theta) = mu (alpha - alpha_L0) sin theta,
    with mu = a0 c / (4 b).

From the coefficients:

    CL  = pi AR A_1
    CDi = pi AR sum_n n A_n^2
    e   = CL^2 / (pi AR CDi)

Chord and twist are taken from the surface geometry at each collocation point.
Both vary linearly between stations, so twist is linearly interpolated across
each panel.

Sweep correction (Helmbold / DATCOM). Lifting-line theory assumes an unswept
wing. The Helmbold formula used in DATCOM for the lift-curve slope of a swept
wing, in incompressible flow,

    CL_alpha = 2 pi AR / (2 + sqrt(AR^2 (1 + tan^2 Lambda_c/2) / kappa^2 + 4)),
    kappa = a0 / (2 pi),

is the unswept formula with kappa replaced by kappa cos(Lambda_c/2), where
Lambda_c/2 is the sweep of the mid-chord line. So the correction is applied
by using an effective section lift slope along the whole surface,

    a0_effective = a0 cos(Lambda_c/2),

where for a surface of several panels cos(Lambda_c/2) is the area-weighted
mean over its panels (for a single trapezoidal panel it is exactly DATCOM's).
Like the formula it comes from, this is a whole-surface correction: it
reduces the lift-curve slope of a swept surface as DATCOM does, but does not
capture how sweep redistributes load along the span (aft sweep loads the
tips), and Cm0 and the zero-lift angle are not corrected. It is reasonable to
about 20-25 deg of mid-chord sweep; beyond that use a vortex-lattice method.

Assumptions: high-aspect-ratio lifting line, with sweep only through the
correction above; linear section lift (no stall); one airfoil along the span;
steady incompressible flow at zero sideslip; one surface in isolation (no
downwash from other surfaces).

Inputs
------
- A ``LiftingSurface`` (symmetric, horizontal, starting at the centreline):
  stations giving span, chord and twist.
- An ``Airfoil``: section lift curve slope [1/rad], zero-lift angle [deg], Cm0.
- Flight condition: angle of attack of the body x-axis [deg], airspeed [m/s],
  air density [kg/m^3].
- Number of Fourier terms (int) and of output points along the span (int).
- x position in body axes about which to take the pitching moment [m].
- Whether to apply the sweep correction (bool, default True).

Outputs
-------
- A ``LiftingLineResult`` holding the spanwise distributions (local lift
  coefficient, circulation, lift per unit span, induced angle), each an array
  of length n_output from root to tip, and the totals: CL, CDi, span
  efficiency e, lift [N], induced drag [N], pitching moment [N m] and Cm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.surface import LiftingSurface
from .airfoil import Airfoil


@dataclass(frozen=True)
class LiftingLineResult:
    """The solution of the lifting-line equation for one surface.

    Attributes
    ----------
    surface_name : str
        Name of the analysed surface.
    alpha_deg : float
        Angle of attack of the body x-axis [deg].
    airspeed_m_s : float
        Freestream airspeed [m/s].
    air_density_kg_m3 : float
        Air density [kg/m^3].
    reference_area_m2, reference_span_m, reference_chord_m : float
        The surface's own area [m^2], span [m] and mean aerodynamic chord [m],
        used to form the coefficients.
    harmonics : ndarray of int, shape (n_terms,)
        Fourier harmonic numbers n = 1, 3, 5, ...
    fourier_coefficients : ndarray, shape (n_terms,)
        The coefficients A_n of the circulation series.
    s_m : ndarray, shape (n_output,)
        Spanwise positions of the output distributions [m], root (0) to tip (b/2).
    local_lift_coefficient : ndarray, shape (n_output,)
        Section lift coefficient Cl at each output position.
    circulation_m2_s : ndarray, shape (n_output,)
        Circulation Gamma [m^2/s] at each output position.
    lift_per_span_N_m : ndarray, shape (n_output,)
        Lift per unit span [N/m] at each output position, one side.
    induced_angle_deg : ndarray, shape (n_output,)
        Induced angle of attack [deg] at each output position (positive = downwash).
    moment_reference_x_m : float
        Body-axis x position [m] about which the pitching moment is taken.
    sweep_correction : bool
        Whether the section lift slope was reduced by cos(mid-chord sweep).
    pitching_moment_N_m : float
        Pitching moment of the whole surface about ``moment_reference_x_m``
        [N m], positive nose-up.

    Properties (computed on demand)
    -------------------------------
    aspect_ratio, dynamic_pressure_Pa, lift_coefficient (CL),
    induced_drag_coefficient (CDi), span_efficiency (e), lift_N,
    induced_drag_N, pitching_moment_coefficient (Cm);
    downwash_m_s and induced_drag_per_span_N_m (spanwise arrays, shape (n_output,)).
    """

    surface_name: str
    alpha_deg: float
    airspeed_m_s: float
    air_density_kg_m3: float
    reference_area_m2: float
    reference_span_m: float
    reference_chord_m: float
    harmonics: np.ndarray
    fourier_coefficients: np.ndarray
    s_m: np.ndarray
    local_lift_coefficient: np.ndarray
    circulation_m2_s: np.ndarray
    lift_per_span_N_m: np.ndarray
    induced_angle_deg: np.ndarray
    moment_reference_x_m: float
    pitching_moment_N_m: float
    sweep_correction: bool = True

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the surface, b^2 / S."""
        return self.reference_span_m**2 / self.reference_area_m2

    @property
    def dynamic_pressure_Pa(self) -> float:
        """Freestream dynamic pressure q = 0.5 rho V^2 [Pa]."""
        return 0.5 * self.air_density_kg_m3 * self.airspeed_m_s**2

    @property
    def lift_coefficient(self) -> float:
        """Surface lift coefficient CL = pi AR A_1."""
        return float(np.pi * self.aspect_ratio * self.fourier_coefficients[0])

    @property
    def induced_drag_coefficient(self) -> float:
        """Induced drag coefficient CDi = pi AR sum(n A_n^2)."""
        return float(np.pi * self.aspect_ratio * np.sum(self.harmonics * self.fourier_coefficients**2))

    @property
    def span_efficiency(self) -> float:
        """Span efficiency factor e = CL^2 / (pi AR CDi). Equal to 1 for elliptic loading.
        Undefined (NaN) when the surface carries no lift."""
        if self.induced_drag_coefficient == 0:
            return float("nan")
        return self.lift_coefficient**2 / (np.pi * self.aspect_ratio * self.induced_drag_coefficient)

    @property
    def lift_N(self) -> float:
        """Total lift of the surface [N], both sides."""
        return self.dynamic_pressure_Pa * self.reference_area_m2 * self.lift_coefficient

    @property
    def induced_drag_N(self) -> float:
        """Total induced drag of the surface [N], both sides."""
        return self.dynamic_pressure_Pa * self.reference_area_m2 * self.induced_drag_coefficient

    @property
    def downwash_m_s(self) -> np.ndarray:
        """Downwash velocity at the lifting line, w = V alpha_i [m/s] (small-angle,
        as in the linear theory), positive downward. Shape (n_output,)."""
        return self.airspeed_m_s * np.radians(self.induced_angle_deg)

    @property
    def induced_drag_per_span_N_m(self) -> np.ndarray:
        """Induced drag per unit span D_i' = L' alpha_i [N/m] at each output
        position, one side: the local lift vector is tilted back by the
        induced angle. Shape (n_output,). Integrates over both sides to
        ``induced_drag_N``."""
        return self.lift_per_span_N_m * np.radians(self.induced_angle_deg)

    @property
    def pitching_moment_coefficient(self) -> float:
        """Pitching moment coefficient Cm = M / (q S c_mac) about the moment reference."""
        return self.pitching_moment_N_m / (
            self.dynamic_pressure_Pa * self.reference_area_m2 * self.reference_chord_m
        )


def solve_lifting_line(
    surface: LiftingSurface,
    alpha_deg: float,
    airspeed_m_s: float,
    air_density_kg_m3: float,
    airfoil: Airfoil = Airfoil(),
    n_terms: int = 30,
    n_output: int = 101,
    moment_reference_x_m: float = 0.0,
    sweep_correction: bool = True,
) -> LiftingLineResult:
    """Solve the lifting-line equation for one surface by the Fourier method.

    Parameters
    ----------
    surface : LiftingSurface
        A symmetric, horizontal surface whose first station is on the centreline.
    alpha_deg : float
        Angle of attack of the body x-axis [deg]. Each section sees this plus
        its local twist.
    airspeed_m_s : float
        Freestream airspeed [m/s]. Must be positive.
    air_density_kg_m3 : float
        Air density [kg/m^3]. Must be positive.
    airfoil : Airfoil
        Section properties, applied along the whole span.
    n_terms : int
        Number of odd Fourier terms (and collocation points). 20-40 is plenty
        for a smooth planform.
    n_output : int
        Number of points, root to tip, at which distributions are returned.
    moment_reference_x_m : float
        Body-axis x [m] about which the pitching moment is taken (e.g. the CG).
    sweep_correction : bool
        If True (default), reduce the section lift slope by the cosine of the
        surface's mid-chord sweep (Helmbold / DATCOM; see the module notes).
        No effect on an unswept surface.

    Returns
    -------
    LiftingLineResult
    """
    _check_surface_is_supported(surface)
    if not (airspeed_m_s > 0 and air_density_kg_m3 > 0):
        raise ValueError("Airspeed and air density must be positive")
    if n_terms < 1 or n_output < 2:
        raise ValueError("Need n_terms >= 1 and n_output >= 2")

    harmonics = 2 * np.arange(n_terms) + 1
    coefficients = _solve_fourier_coefficients(
        surface, airfoil, np.radians(alpha_deg), harmonics, sweep_correction
    )

    # Output points from root (theta = pi/2) to tip (theta = 0).
    theta = np.linspace(np.pi / 2, 0.0, n_output)
    s = (surface.span / 2) * np.cos(theta)
    s[0] = 0.0  # cos(pi/2) is only zero to rounding error
    circulation, local_cl, induced_angle_rad = _distributions(
        surface, harmonics, coefficients, theta, s, airspeed_m_s
    )
    lift_per_span = air_density_kg_m3 * airspeed_m_s * circulation

    dynamic_pressure = 0.5 * air_density_kg_m3 * airspeed_m_s**2
    moment = _pitching_moment(surface, airfoil, s, lift_per_span, dynamic_pressure, moment_reference_x_m)

    return LiftingLineResult(
        surface_name=surface.name,
        alpha_deg=float(alpha_deg),
        airspeed_m_s=float(airspeed_m_s),
        air_density_kg_m3=float(air_density_kg_m3),
        reference_area_m2=surface.area,
        reference_span_m=surface.span,
        reference_chord_m=surface.mac,
        harmonics=harmonics,
        fourier_coefficients=coefficients,
        s_m=s,
        local_lift_coefficient=local_cl,
        circulation_m2_s=circulation,
        lift_per_span_N_m=lift_per_span,
        induced_angle_deg=np.degrees(induced_angle_rad),
        moment_reference_x_m=float(moment_reference_x_m),
        pitching_moment_N_m=moment,
        sweep_correction=bool(sweep_correction),
    )


def solve_lifting_line_for_lift(
    surface: LiftingSurface,
    lift_N: float,
    airspeed_m_s: float,
    air_density_kg_m3: float,
    airfoil: Airfoil = Airfoil(),
    n_terms: int = 30,
    n_output: int = 101,
    moment_reference_x_m: float = 0.0,
    sweep_correction: bool = True,
) -> LiftingLineResult:
    """Solve the lifting line at the angle of attack that gives a required lift.

    The lifting-line model is linear, so lift varies linearly with angle of
    attack: L(alpha) = L(0) + alpha * dL/dalpha. Two solves find that line and
    a third returns the full solution at the required angle. Use this to get
    the lift distribution at a load case, e.g. lift_N = n * W.

    Parameters
    ----------
    surface, airspeed_m_s, air_density_kg_m3, airfoil, n_terms, n_output,
    moment_reference_x_m, sweep_correction
        As for ``solve_lifting_line``.
    lift_N : float
        Required total lift of the surface, both sides [N].

    Returns
    -------
    LiftingLineResult
        The solution at the required lift; its ``alpha_deg`` is the angle found.
    """
    settings = dict(airfoil=airfoil, n_terms=n_terms, n_output=2, sweep_correction=sweep_correction)
    lift_at_zero = solve_lifting_line(surface, 0.0, airspeed_m_s, air_density_kg_m3, **settings).lift_N
    lift_at_one = solve_lifting_line(surface, 1.0, airspeed_m_s, air_density_kg_m3, **settings).lift_N
    lift_per_degree = lift_at_one - lift_at_zero
    alpha_deg = (lift_N - lift_at_zero) / lift_per_degree
    return solve_lifting_line(
        surface, alpha_deg, airspeed_m_s, air_density_kg_m3,
        airfoil=airfoil, n_terms=n_terms, n_output=n_output,
        moment_reference_x_m=moment_reference_x_m, sweep_correction=sweep_correction,
    )


def _check_surface_is_supported(surface: LiftingSurface) -> None:
    """Raise ValueError unless the surface suits the symmetric Fourier solution."""
    if not surface.symmetric:
        raise ValueError(f"{surface.name}: lifting line needs a symmetric surface")
    if surface.orientation != "horizontal":
        raise ValueError(f"{surface.name}: lifting line needs a horizontal surface")
    if surface.s[0] != 0:
        raise ValueError(f"{surface.name}: lifting line needs the first station on the centreline (s = 0)")


def _solve_fourier_coefficients(
    surface: LiftingSurface,
    airfoil: Airfoil,
    alpha_rad: float,
    harmonics: np.ndarray,
    sweep_correction: bool = True,
) -> np.ndarray:
    """Solve the monoplane equation for the Fourier coefficients A_n.

    Inputs: surface, airfoil, body angle of attack [rad], odd harmonics, shape
    (N,), whether to apply the sweep correction.
    Output: coefficients A_n, shape (N,).
    """
    n_points = len(harmonics)
    span = surface.span

    # Collocation points on one side, from near the tip to the root (theta = pi/2).
    theta = np.arange(1, n_points + 1) * np.pi / (2 * n_points)
    s = (span / 2) * np.cos(theta)

    chord = surface.chord_at(s)
    section_alpha_rad = alpha_rad + np.radians(surface.twist_deg_at(s))
    lift_slope = effective_lift_slope_per_rad(surface, airfoil) if sweep_correction else airfoil.lift_slope_per_rad
    mu = lift_slope * chord / (4 * span)

    sin_n_theta = np.sin(np.outer(theta, harmonics))  # shape (N points, N terms)
    matrix = sin_n_theta * (np.outer(mu, harmonics) + np.sin(theta)[:, None])
    right_hand_side = mu * (section_alpha_rad - airfoil.alpha_zero_lift_rad) * np.sin(theta)
    return np.linalg.solve(matrix, right_hand_side)


def effective_lift_slope_per_rad(surface: LiftingSurface, airfoil: Airfoil) -> float:
    """Section lift slope corrected for sweep, a0 cos(Lambda_c/2) [1/rad].

    cos(Lambda_c/2) is the area-weighted mean, over the surface's panels, of
    the cosine of each panel's mid-chord sweep (Helmbold / DATCOM; see the
    module notes). Equal to a0 for an unswept surface.

    Inputs: surface; airfoil. Output: effective lift slope [1/rad] (float).
    """
    midchord_sweep_rad = np.radians(surface.sweep_deg(chord_fraction=0.5))  # shape (n_panels,)
    panel_areas = surface.panel_areas()
    mean_cosine = np.sum(panel_areas * np.cos(midchord_sweep_rad)) / np.sum(panel_areas)
    return float(airfoil.lift_slope_per_rad * mean_cosine)


def _distributions(
    surface: LiftingSurface,
    harmonics: np.ndarray,
    coefficients: np.ndarray,
    theta: np.ndarray,
    s: np.ndarray,
    airspeed_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Fourier series at output points.

    Inputs: odd harmonics and coefficients A_n, shape (N,); output angles theta
    and spanwise positions s [m], shape (M,); airspeed [m/s].
    Outputs, each shape (M,): circulation [m^2/s], local lift coefficient,
    induced angle [rad].
    """
    span = surface.span
    sin_n_theta = np.sin(np.outer(theta, harmonics))
    series = sin_n_theta @ coefficients

    circulation = 2 * span * airspeed_m_s * series
    local_cl = 4 * span * series / surface.chord_at(s)

    # Induced angle = sum(n A_n sin n theta) / sin theta. At the tip (sin theta = 0)
    # use its limit, sum(n^2 A_n).
    sin_theta = np.sin(theta)
    at_tip = np.isclose(sin_theta, 0.0)
    numerator = sin_n_theta @ (harmonics * coefficients)
    induced_angle = np.empty_like(theta)
    induced_angle[~at_tip] = numerator[~at_tip] / sin_theta[~at_tip]
    induced_angle[at_tip] = np.sum(harmonics**2 * coefficients)
    return circulation, local_cl, induced_angle


def _pitching_moment(
    surface: LiftingSurface,
    airfoil: Airfoil,
    s: np.ndarray,
    lift_per_span: np.ndarray,
    dynamic_pressure: float,
    moment_reference_x_m: float,
) -> float:
    """Pitching moment of the whole surface [N m], positive nose-up.

    Section lift acts at the local quarter chord; each section also carries its
    zero-lift moment Cm0 q c^2. With x positive aft, lift aft of the reference
    point gives a nose-down (negative) moment.

    Inputs: spanwise positions s [m] and lift per span [N/m], shape (M,), root
    to tip; dynamic pressure [Pa]; moment reference x in body axes [m].
    """
    x_quarter_chord = surface.origin[0] + surface.x_le_at(s) + 0.25 * surface.chord_at(s)
    lift_moment_one_side = -_integrate_over_span(lift_per_span * (x_quarter_chord - moment_reference_x_m), s)

    # The integral of c^2 over both sides is S * MAC, exactly.
    section_moment = airfoil.cm0 * dynamic_pressure * surface.area * surface.mac
    return float(2 * lift_moment_one_side + section_moment)


def _integrate_over_span(values: np.ndarray, s: np.ndarray) -> float:
    """Trapezoidal integral of ``values`` over spanwise positions ``s`` (both shape (M,))."""
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(s)))
