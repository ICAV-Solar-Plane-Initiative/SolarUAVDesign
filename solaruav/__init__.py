"""SolarUAVDesign: conceptual and preliminary design tools for ICAV-SPI's solar aircraft."""

from .geometry import Aircraft, LiftingSurface, Station
from .plotting import plot_aircraft, plot_surface

__all__ = ["Aircraft", "LiftingSurface", "Station", "plot_aircraft", "plot_surface"]
__version__ = "0.1.0"
