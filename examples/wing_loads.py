"""Interactive cantilever-wing shear-force and bending-moment calculator.

Run from the repository root with:

    python examples/wing_loads.py

Enter comma-separated span stations, lift and mass-per-span values in the
window. The example values are illustrative, not validated aircraft loads.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from solaruav.structures.loads import WingLoads, calculate_wing_loads


def parse_values(text: str, field_name: str) -> list[float]:
    """Parse comma-separated numeric values from one UI input field."""
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as error:
        raise ValueError(f"{field_name} must contain numbers separated by commas.") from error
    if not values:
        raise ValueError(f"{field_name} cannot be empty.")
    return values


def main() -> None:
    """Open the input form and live shear-force and bending-moment plots."""
    default_span = np.linspace(0.0, 2.5, 6)
    default_lift = 30.0 * (1.0 - default_span / 2.5)
    default_mass = np.full_like(default_span, 0.8)

    figure = plt.figure(figsize=(13, 8))
    figure.canvas.manager.set_window_title("Wing shear force and bending moment")

    figure.text(0.04, 0.95, "Cantilever wing load calculator", fontsize=16, weight="bold")
    figure.text(
        0.04,
        0.91,
        "Enter one value per station, separated by commas. Use root-to-tip positions in metres.",
        fontsize=10,
    )

    def add_input(label: str, initial_value: str, y: float) -> TextBox:
        figure.text(0.04, y + 0.045, label, fontsize=10)
        axis = figure.add_axes((0.04, y, 0.31, 0.04))
        return TextBox(axis, "", initial=initial_value)

    span_input = add_input("Span stations [m]", ", ".join(f"{v:g}" for v in default_span), 0.81)
    lift_input = add_input(
        "Lift per span at each station [N/m]",
        ", ".join(f"{v:g}" for v in default_lift),
        0.70,
    )
    mass_input = add_input(
        "Mass per span at each station [kg/m]",
        ", ".join(f"{v:g}" for v in default_mass),
        0.59,
    )

    figure.text(0.04, 0.51, "Limit load factor", fontsize=10)
    factor_axis = figure.add_axes((0.04, 0.46, 0.13, 0.04))
    factor_input = TextBox(factor_axis, "", initial="2.5")
    figure.text(0.21, 0.51, "Safety factor", fontsize=10)
    safety_axis = figure.add_axes((0.21, 0.46, 0.14, 0.04))
    safety_input = TextBox(safety_axis, "", initial="1.5")

    status_text = figure.text(0.04, 0.31, "", fontsize=9, color="firebrick", wrap=True)
    result_text = figure.text(0.04, 0.18, "", fontsize=10, va="top", wrap=True)

    shear_axis = figure.add_axes((0.42, 0.56, 0.54, 0.34))
    moment_axis = figure.add_axes((0.42, 0.12, 0.54, 0.34), sharex=shear_axis)

    def calculate(_event=None) -> None:
        """Read the fields, recalculate loads, and refresh the plots."""
        try:
            span = parse_values(span_input.text, "Span stations")
            lift = parse_values(lift_input.text, "Lift per span")
            mass = parse_values(mass_input.text, "Mass per span")
            limit_factor = float(factor_input.text.strip())
            safety_factor = float(safety_input.text.strip())
            loads = calculate_wing_loads(
                span,
                lift,
                mass,
                limit_load_factor=limit_factor,
                safety_factor=safety_factor,
            )
        except ValueError as error:
            status_text.set_text(str(error))
            result_text.set_text("")
            figure.canvas.draw_idle()
            return

        status_text.set_text("")
        draw_loads(shear_axis, moment_axis, loads)
        result_text.set_text(
            f"Root shear: {loads.shear_force_limit_N[0]:.2f} N limit, "
            f"{loads.shear_force_ultimate_N[0]:.2f} N ultimate\n"
            f"Root bending moment: {loads.bending_moment_limit_N_m[0]:.2f} N m limit, "
            f"{loads.bending_moment_ultimate_N_m[0]:.2f} N m ultimate"
        )
        figure.canvas.draw_idle()

    def draw_loads(shear, moment, loads: WingLoads) -> None:
        """Draw limit and ultimate shear and moment distributions."""
        shear.clear()
        shear.plot(loads.span_m, loads.shear_force_limit_N, label="Limit")
        shear.plot(loads.span_m, loads.shear_force_ultimate_N, label="Ultimate")
        shear.set_ylabel("Shear force [N]")
        shear.grid(True)
        shear.legend()

        moment.clear()
        moment.plot(loads.span_m, loads.bending_moment_limit_N_m, label="Limit")
        moment.plot(loads.span_m, loads.bending_moment_ultimate_N_m, label="Ultimate")
        moment.set_xlabel("Distance from wing root [m]")
        moment.set_ylabel("Bending moment [N m]")
        moment.grid(True)
        moment.legend()

    calculate_axis = figure.add_axes((0.04, 0.37, 0.14, 0.06))
    calculate_button = Button(calculate_axis, "Calculate")
    calculate_button.on_clicked(calculate)

    example_axis = figure.add_axes((0.21, 0.37, 0.14, 0.06))
    example_button = Button(example_axis, "Restore example")

    def restore_example(_event=None) -> None:
        """Put the illustrative inputs back into the form and recalculate."""
        span_input.set_val(", ".join(f"{v:g}" for v in default_span))
        lift_input.set_val(", ".join(f"{v:g}" for v in default_lift))
        mass_input.set_val(", ".join(f"{v:g}" for v in default_mass))
        factor_input.set_val("2.5")
        safety_input.set_val("1.5")
        calculate()

    example_button.on_clicked(restore_example)
    calculate()
    plt.show()


if __name__ == "__main__":
    main()
