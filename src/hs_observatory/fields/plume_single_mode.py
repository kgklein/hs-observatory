"""One physically phased plane wave from a selected PLUME ``.mode1`` row."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.fields.base import FieldModel
from hs_observatory.utils.units import vector3

ELEMENTARY_CHARGE_C = 1.602176634e-19
PROTON_MASS_KG = 1.67262192369e-27


@dataclass(frozen=True)
class ProtonReferenceScales:
    """Physical proton scales derived from reference temperature and field."""

    omega_ref_rad_s: float
    vth_ref_m_s: float
    rho_ref_m: float
    rho_ref_km: float


@dataclass(frozen=True)
class PlumeEigenmode:
    """Dimensionless PLUME mode values and selected component eigenfunctions."""

    k_perp_rho_ref: float
    k_parallel_rho_ref: float
    omega_r_over_omega_ref: float
    gamma_over_omega_ref: float
    B_complex: np.ndarray
    v_complex: np.ndarray
    n_complex: complex


def get_plume_column_map(n_components: int) -> dict[str, int | tuple[int, int]]:
    """Return zero-based columns for a PLUME eigenfunction output row.

    PLUME documentation uses one-based columns. The common block occupies
    columns 1--18. It is followed by six velocity columns per component and
    then two density columns per component.
    """

    if n_components not in {2, 3, 4}:
        raise ValueError("n_components must be one of: 2, 3, 4")
    columns: dict[str, int | tuple[int, int]] = {
        "k_perp_rho_ref": 0,
        "k_parallel_rho_ref": 1,
        "omega_r_over_Omega_ref": 4,
        "gamma_over_Omega_ref": 5,
        "B_x": (6, 7),
        "B_y": (8, 9),
        "B_z": (10, 11),
    }
    velocity_start = 18
    density_start = velocity_start + 6 * n_components
    for component in range(n_components):
        species = component + 1
        start = velocity_start + 6 * component
        columns[f"U_{species}_x"] = (start, start + 1)
        columns[f"U_{species}_y"] = (start + 2, start + 3)
        columns[f"U_{species}_z"] = (start + 4, start + 5)
        columns[f"n_{species}"] = (
            density_start + 2 * component,
            density_start + 2 * component + 1,
        )
    columns["U_ref_x"] = columns["U_1_x"]
    columns["U_ref_y"] = columns["U_1_y"]
    columns["U_ref_z"] = columns["U_1_z"]
    columns["n_ref"] = columns["n_1"]
    columns["minimum_columns"] = 18 + 8 * n_components
    return columns


def load_plume_row(
    path: str | Path,
    row_index: int,
    row_index_base: int = 0,
) -> np.ndarray:
    """Load one explicitly indexed numeric row from a whitespace PLUME file."""

    if row_index_base not in {0, 1}:
        raise ValueError("row_index_base must be 0 or 1")
    zero_based_index = int(row_index) - row_index_base
    if zero_based_index < 0:
        raise IndexError(
            f"row_index {row_index} is invalid for row_index_base {row_index_base}"
        )
    expanded_path = Path(path).expanduser()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"PLUME mode file not found: {expanded_path}")
    table = np.loadtxt(expanded_path, dtype=float, ndmin=2)
    if zero_based_index >= table.shape[0]:
        raise IndexError(
            f"PLUME row index {row_index} (base {row_index_base}) is outside "
            f"{expanded_path}, which contains {table.shape[0]} numeric rows"
        )
    return np.asarray(table[zero_based_index], dtype=float)


def complex_from_pair(row: np.ndarray, col_re: int, col_im: int) -> complex:
    """Construct one complex eigenfunction value from real/imaginary columns."""

    values = np.asarray(row, dtype=float)
    if max(col_re, col_im) >= values.size:
        raise ValueError(
            f"PLUME row has {values.size} columns; requested complex pair "
            f"({col_re}, {col_im})"
        )
    return complex(values[col_re], values[col_im])


def _complex_for_name(
    row: np.ndarray,
    columns: dict[str, int | tuple[int, int]],
    name: str,
) -> complex:
    pair = columns[name]
    if not isinstance(pair, tuple):
        raise TypeError(f"PLUME column {name} is not a complex pair")
    return complex_from_pair(row, *pair)


def _species_number(output_species: str | int, n_components: int) -> int:
    if output_species == "ref":
        return 1
    if isinstance(output_species, bool) or not isinstance(output_species, int):
        raise NotImplementedError("output_species must be 'ref' or a one-based integer")
    if not 1 <= output_species <= n_components:
        raise ValueError(
            f"output_species must lie between 1 and n_components={n_components}"
        )
    return output_species


def parse_plume_eigenmode(
    row: np.ndarray,
    n_components: int,
    output_species: str | int = "ref",
) -> PlumeEigenmode:
    """Parse common fields plus one selected species from a PLUME row."""

    values = np.asarray(row, dtype=float)
    columns = get_plume_column_map(n_components)
    minimum_columns = int(columns["minimum_columns"])
    if values.ndim != 1 or values.size < minimum_columns:
        raise ValueError(
            f"PLUME eigenmode row for {n_components} components requires at least "
            f"{minimum_columns} columns; found {values.size}"
        )
    species = _species_number(output_species, n_components)
    B_complex = np.asarray(
        [_complex_for_name(values, columns, f"B_{axis}") for axis in "xyz"],
        dtype=complex,
    )
    v_complex = np.asarray(
        [
            _complex_for_name(values, columns, f"U_{species}_{axis}")
            for axis in "xyz"
        ],
        dtype=complex,
    )
    return PlumeEigenmode(
        k_perp_rho_ref=float(values[int(columns["k_perp_rho_ref"])]),
        k_parallel_rho_ref=float(values[int(columns["k_parallel_rho_ref"])]),
        omega_r_over_omega_ref=float(
            values[int(columns["omega_r_over_Omega_ref"])]
        ),
        gamma_over_omega_ref=float(
            values[int(columns["gamma_over_Omega_ref"])]
        ),
        B_complex=B_complex,
        v_complex=v_complex,
        n_complex=_complex_for_name(values, columns, f"n_{species}"),
    )


def compute_proton_reference_scales(
    T_ref_eV: float,
    B_ref_nT: float,
) -> ProtonReferenceScales:
    """Compute ``Omega_p`` and ``rho_p`` using ``vth=sqrt(2 kT/m_p)``.

    Here ``k_B T = T_ref_eV * e`` in joules. ``Omega_p=eB/m_p`` is an
    angular frequency in rad/s; no extra factor of ``2*pi`` is applied.
    """

    temperature_eV = float(T_ref_eV)
    magnetic_nT = float(B_ref_nT)
    if not np.isfinite(temperature_eV) or temperature_eV <= 0.0:
        raise ValueError("T_ref_eV must be positive and finite")
    if not np.isfinite(magnetic_nT) or magnetic_nT <= 0.0:
        raise ValueError("B_ref_nT must be positive and finite")
    thermal_energy_joule = temperature_eV * ELEMENTARY_CHARGE_C
    magnetic_tesla = magnetic_nT * 1.0e-9
    omega = ELEMENTARY_CHARGE_C * magnetic_tesla / PROTON_MASS_KG
    vth = np.sqrt(2.0 * thermal_energy_joule / PROTON_MASS_KG)
    rho_m = vth / omega
    return ProtonReferenceScales(
        omega_ref_rad_s=float(omega),
        vth_ref_m_s=float(vth),
        rho_ref_m=float(rho_m),
        rho_ref_km=float(rho_m / 1000.0),
    )


class PlumeSingleModeField(FieldModel):
    """Background values plus one real PLUME eigenmode plane wave."""

    name = "plume_single_mode"

    def __init__(self, config: dict[str, Any]) -> None:
        self.background_n = float(config.get("background", {}).get("n", 0.0))
        self.background_v = vector3(
            config.get("background", {}).get("v", (0.0, 0.0, 0.0)),
            "field.background.v",
        )
        self.background_B = vector3(
            config.get("background", {}).get("B", (0.0, 0.0, 0.0)),
            "field.background.B",
        )
        if not np.isfinite(self.background_n):
            raise ValueError("field.background.n must be finite")

        plume = config.get("plume")
        if not isinstance(plume, dict):
            raise ValueError("plume_single_mode requires a field.plume mapping")
        geometry = plume.get("geometry", {})
        if geometry.get("k_perp_axis", "x") != "x":
            raise NotImplementedError("PLUME k_perp_axis must currently be x")
        if geometry.get("k_parallel_axis", "z") != "z":
            raise NotImplementedError("PLUME k_parallel_axis must currently be z")
        if bool(plume.get("include_growth", False)):
            raise NotImplementedError(
                "PLUME growth/damping is not applied yet; set include_growth: false"
            )

        self.path = Path(str(plume.get("path", ""))).expanduser()
        self.n_components = int(plume.get("n_components", 0))
        self.row_index = int(plume.get("row_index", 0))
        self.row_index_base = int(plume.get("row_index_base", 0))
        self.output_species = plume.get("output_species", "ref")
        self.phase0 = float(plume.get("phase0", 0.0))
        if not np.isfinite(self.phase0):
            raise ValueError("field.plume.phase0 must be finite")
        amplitude = plume.get("amplitude", {})
        if not isinstance(amplitude, dict):
            raise ValueError("field.plume.amplitude must be a mapping")
        self.B_rms_nT = float(amplitude.get("B_rms_nT", 0.1))
        if not np.isfinite(self.B_rms_nT) or self.B_rms_nT <= 0.0:
            raise ValueError("field.plume.amplitude.B_rms_nT must be positive and finite")

        row = load_plume_row(self.path, self.row_index, self.row_index_base)
        self.mode = parse_plume_eigenmode(
            row, self.n_components, self.output_species
        )
        try:
            self.T_ref_eV = float(plume["T_ref_eV"])
            self.B_ref_nT = float(plume["B_ref_nT"])
        except KeyError as error:
            raise ValueError(f"field.plume.{error.args[0]} is required") from error
        self.reference_scales = compute_proton_reference_scales(
            self.T_ref_eV,
            self.B_ref_nT,
        )
        magnetic_trace_amplitude = float(np.linalg.norm(self.mode.B_complex))
        if magnetic_trace_amplitude == 0.0:
            raise ValueError("Selected PLUME mode has zero magnetic trace amplitude")
        self.scale = self.B_rms_nT / magnetic_trace_amplitude
        self.B_complex_scaled = self.scale * self.mode.B_complex
        self.v_complex_scaled = self.scale * self.mode.v_complex
        self.n_complex_scaled = self.scale * self.mode.n_complex

        rho_km = self.reference_scales.rho_ref_km
        omega_ref = self.reference_scales.omega_ref_rad_s
        self.k_1_per_km = np.asarray(
            [
                self.mode.k_perp_rho_ref / rho_km,
                0.0,
                self.mode.k_parallel_rho_ref / rho_km,
            ]
        )
        self.omega_r_rad_s = self.mode.omega_r_over_omega_ref * omega_ref
        self.gamma_rad_s = self.mode.gamma_over_omega_ref * omega_ref
        self.metadata = {
            "field_model": self.name,
            "plume_path": str(self.path.resolve()),
            "n_components": self.n_components,
            "row_index": self.row_index,
            "row_index_base": self.row_index_base,
            "k_perp_rho_ref": self.mode.k_perp_rho_ref,
            "k_parallel_rho_ref": self.mode.k_parallel_rho_ref,
            "omega_r_over_Omega_ref": self.mode.omega_r_over_omega_ref,
            "gamma_over_Omega_ref": self.mode.gamma_over_omega_ref,
            "rho_ref_km": rho_km,
            "vth_ref_m_s": self.reference_scales.vth_ref_m_s,
            "Omega_ref_rad_s": omega_ref,
            "T_ref_eV": self.T_ref_eV,
            "B_ref_nT": self.B_ref_nT,
            "k_perp_1_per_km": float(self.k_1_per_km[0]),
            "k_parallel_1_per_km": float(self.k_1_per_km[2]),
            "omega_r_rad_s": self.omega_r_rad_s,
            "gamma_rad_s": self.gamma_rad_s,
            "B_rms_nT": self.B_rms_nT,
            "eigenfunction_scale": self.scale,
            "output_species": self.output_species,
            "include_growth": False,
        }
        print(self.selection_summary())

    def selection_summary(self) -> str:
        """Return a concise dimensionless/physical selected-mode report."""

        m = self.metadata
        return (
            f"Selected PLUME row index: {self.row_index} (base {self.row_index_base})\n"
            f"  k_perp rho_ref = {m['k_perp_rho_ref']:.8g}\n"
            f"  k_parallel rho_ref = {m['k_parallel_rho_ref']:.8g}\n"
            f"  omega_r / Omega_ref = {m['omega_r_over_Omega_ref']:.8g}\n"
            f"  gamma / Omega_ref = {m['gamma_over_Omega_ref']:.8g}\n"
            f"  rho_ref = {m['rho_ref_km']:.8g} km\n"
            f"  Omega_ref = {m['Omega_ref_rad_s']:.8g} rad/s\n"
            f"  k = ({m['k_perp_1_per_km']:.8g}, 0, "
            f"{m['k_parallel_1_per_km']:.8g}) 1/km\n"
            f"  omega_r = {m['omega_r_rad_s']:.8g} rad/s\n"
            f"  gamma = {m['gamma_rad_s']:.8g} rad/s"
        )

    def evaluate_quantity(
        self, quantity: str, time: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        if quantity not in {"n", "v", "B"}:
            raise ValueError(f"Unknown field quantity: {quantity}")
        time = np.asarray(time, dtype=float)
        position = np.asarray(position, dtype=float)
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("position must have shape (n_spacecraft, n_time, 3)")
        n_spacecraft, n_time, _ = position.shape
        if time.shape == (n_time,):
            time_by_spacecraft = time[:, None]
        elif time.shape == (n_time, n_spacecraft):
            time_by_spacecraft = time
        else:
            raise ValueError(
                "time must have shape (n_time,) or (n_time, n_spacecraft)"
            )
        positions_time_first = np.swapaxes(position, 0, 1)
        phase = (
            np.einsum("tsd,d->ts", positions_time_first, self.k_1_per_km)
            - self.omega_r_rad_s * time_by_spacecraft
            + self.phase0
        )
        oscillation = np.exp(1j * phase)
        if quantity == "n":
            fluctuation = np.real(self.n_complex_scaled * oscillation)
            return self.background_n + fluctuation
        eigenfunction = (
            self.v_complex_scaled if quantity == "v" else self.B_complex_scaled
        )
        background = self.background_v if quantity == "v" else self.background_B
        fluctuation = np.real(oscillation[..., None] * eigenfunction)
        return background + fluctuation
