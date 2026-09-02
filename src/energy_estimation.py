from abc import ABC, abstractmethod
import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
import os

import hwcomponents as hwc
from hwcomponents_neurosim import MemoryCell, RowDrivers, ShiftAdd
from hwcomponents_adc import ADC
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class ArchAttrs():
    """Architecture attributes."""

    # Non-optional fields
    cell_config: str
    rows: int
    cols: int
    g_min: float
    g_max: float
    read_voltage: float

    # Optional fields
    read_pulse_width: float = 1e-09
    cycle_seconds: float = 1e-07
    voltage: int = 1
    mvm_latency: int = 1
    threshold_voltage: float = 0
    tech_node: float = 32e-9
    adc_resolution: int = 8
    # Adders per column (based on mapping and MVM profiling)
    adder_col_scale: float = 0
    # ADC conversions per column (based on mapping)
    adc_col_scale: float = 1

    # Derived/constant fields
    cols_active_at_once: int = field(init=False)
    sequential: bool = field(init=False)
    global_cycle_seconds: float = field(init=False)
    n_instances: int = field(init=False)
    temporal_dac_bits: int = field(init=False)
    temporal_spiking: bool = field(init=False)
    throughput: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'cols_active_at_once', self.cols)
        object.__setattr__(self, 'sequential', False)
        object.__setattr__(self, 'global_cycle_seconds', self.cycle_seconds)
        object.__setattr__(self, 'n_instances', 1)
        object.__setattr__(self, 'temporal_dac_bits', 1)
        object.__setattr__(self, 'temporal_spiking', 1)
        object.__setattr__(self, 'throughput', 1 /
                           (self.cycle_seconds * self.mvm_latency))
        self._update_cell_config()

    def _update_cell_config(self):
        """
        Modify cell config read voltage/read pulse/resistance
        entries with values derived from this ArchAttrs instance.
        """
        with open(self.cell_config, 'r') as f:
            config = yaml.safe_load(f)

        # -ReadVoltage (V)
        config['-ReadVoltage (V)'] = self.read_voltage
        # -ReadPulse (ns): read_pulse_width is stored in seconds
        config['-ReadPulse (ns)'] = self.read_pulse_width * 1e9
        # -ResistanceOn / -ResistanceOff (ohm): convert conductance -> resistance.
        config['-ResistanceOn (ohm)'] = 1.0 / self.g_max
        config['-ResistanceOff (ohm)'] = 1.0 / self.g_min

        # Dump YAML
        orig_path = self.cell_config
        parent_dir = os.path.dirname(orig_path)
        tmp_dir = os.path.join(parent_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        base_name = os.path.basename(orig_path)
        stem, ext = os.path.splitext(base_name)
        new_path = os.path.join(tmp_dir, f"{stem}_temp{ext}")

        with open(new_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Use modified cell config
        object.__setattr__(self, 'cell_config', new_path)

    def get_neurosim_component_kwargs(self) -> dict:
        """Get a partial dict of parameters to be passed to a NeuroSim
        component."""
        return {
            'tech_node': self.tech_node,
            'cycle_period': self.global_cycle_seconds,
            'rows': self.rows,
            'cols': self.cols,
            'cols_active_at_once': self.cols_active_at_once,
            'cell_config': self.cell_config,
            'read_pulse_width': self.read_pulse_width,
            'adc_resolution': 0,
            'temporal_dac_bits': self.temporal_dac_bits,
            'temporal_spiking': self.temporal_spiking,
            'voltage': self.voltage,
            'threshold_voltage': self.threshold_voltage,
            'sequential': self.sequential,
            'n_instances': self.n_instances,
        }

    def get_adc_component_kwargs(self) -> dict:
        """Get a dict of parameters to be passed to an ADC component."""
        return {
            'n_bits': self.adc_resolution,
            'tech_node': self.tech_node,
            'throughput': self.throughput,
            'n_adcs': 1,
        }

    def get_shift_adder_components_kwargs(self) -> dict:
        """Get a dict of parameters to be passed to a post-processing adder."""
        return {
            'tech_node': self.tech_node,
            'cycle_period': self.cycle_seconds,
            'n_bits': self.adc_resolution,
            'shift_register_n_bits': self.adc_resolution + 1,
            'n_instances': self.n_instances,
        }


@dataclass(frozen=True)
class MVMAttrs:
    """MVM attributes."""

    active_rows: int
    active_cols: int
    average_cell_value: float
    average_input_value: float


class BaseEnergyModel(ABC):
    """Base energy model."""

    def __init__(self, name: str, arch_attrs: ArchAttrs):
        self._name = name
        self._arch_attrs = arch_attrs

    @abstractmethod
    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        ...

    @abstractmethod
    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        ...

    @property
    def name(self):
        return self._name

    def _get_neurosim_component_kwargs(self, mvm_attrs: MVMAttrs) -> dict:
        kwargs = self._arch_attrs.get_neurosim_component_kwargs()
        kwargs['average_cell_value'] = mvm_attrs.average_cell_value
        kwargs['average_input_value'] = mvm_attrs.average_input_value
        return kwargs


class RowDriverEnergyModel(BaseEnergyModel):
    """Row driver energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        row_driver_comp = RowDrivers(
            **self._get_neurosim_component_kwargs(mvm_attrs))
        return row_driver_comp.read().energy * mvm_attrs.active_rows

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        row_driver_comp = RowDrivers(
            **self._get_neurosim_component_kwargs(mvm_attrs))
        return row_driver_comp.write().energy * mvm_attrs.active_rows


class MemristorEnergyModel(BaseEnergyModel):
    """Memristor energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        memory_cell_comp = MemoryCell(
            **self._get_neurosim_component_kwargs(mvm_attrs))
        return (memory_cell_comp.read().energy *
                mvm_attrs.active_rows *
                mvm_attrs.active_cols)

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        memory_cell_comp = MemoryCell(
            **self._get_neurosim_component_kwargs(mvm_attrs))
        return (memory_cell_comp.write().energy *
                mvm_attrs.active_rows *
                mvm_attrs.active_cols)


class ADCEnergyModel(BaseEnergyModel):
    """ADC energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        adc_comp = ADC(**self._arch_attrs.get_adc_component_kwargs())
        return (adc_comp.convert().energy *
                mvm_attrs.active_cols *
                self._arch_attrs.adc_col_scale)

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        return 0.0


class AdderEnergyModel(BaseEnergyModel):
    """Shift Adder energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        adder_comp = ShiftAdd(
            **self._arch_attrs.get_shift_adder_components_kwargs())
        return (adder_comp.add().energy *
                mvm_attrs.active_cols *
                self._arch_attrs.adder_col_scale)

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        return 0.0


class CrossbarEnergyModel(BaseEnergyModel):
    """Crossbar energy model.

    Integrates sub-components and provides per component energy.
    """

    def __init__(self,
                 name: str,
                 arch_attrs: ArchAttrs,
                 with_adders: bool = True):
        super().__init__(name, arch_attrs)
        self.row_driver = RowDriverEnergyModel(
            name + ".row_driver", arch_attrs)
        self.memristors = MemristorEnergyModel(
            name + ".memristors", arch_attrs)
        self.adcs = ADCEnergyModel(name + ".adcs", arch_attrs)
        self.components = [self.row_driver,
                           self.memristors, self.adcs]
        if with_adders:
            self.adders = AdderEnergyModel(name + ".adders", arch_attrs)
            self.components.append(self.adders)

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> dict:
        return {c.name: c.get_mvm_energy(mvm_attrs) for c in self.components}

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> dict:
        return {c.name: c.get_write_energy(mvm_attrs) for c in self.components}


def run_single_energy_estimation(mvm_profile: dict,
                                 cell_config: str,
                                 xbar_size: tuple,
                                 hrs_lrs: tuple,
                                 read_voltage: float,
                                 adc_resolution: int,
                                 m_mode: str,
                                 tech_node: float,
                                 cycle_seconds: float
                                 ) -> tuple[dict, dict]:
    """Run energy estimation for a single configuration/MVM profile."""
    # Make read voltage positive (otherwise neurosim returns zero
    # memristor energy).
    read_voltage = abs(read_voltage)
    # Convert HRS/LRS current (uA) to min/max conductance (S)
    g_min, g_max = (cur / read_voltage * 1e-6 for cur in hrs_lrs)

    # Adder scaling for different BNN/TNN mappings.
    # TODO: Support for int mappings (bit slicing)
    bt_adder_scales = {
        'BNN_I': 0.5,    # 1 addition per 2 columns for digital correction
        'BNN_II': 0.5,   # 1 addition per 2 columns for digital correction
        'BNN_III': 1,    # 1 addition per MVM for digital correction and accumulation
        'BNN_IV': 1,     # 1 addition per MVM for digital correction and accumulation
        'BNN_V': 1,      # 1 addition per column for digital correction
        'BNN_VI': 0,     # No addition
        'TNN_I': 0,      # No addition
        'TNN_II': 0.5,   # 1 addition per 2 columns for digital correction/accumulation
        'TNN_III': 0.5,  # 1 addition per 2 columns for digital correction/accumulation
        'TNN_IV': 1,     # 1 addition per MVM for digital correction and accumulation
        'TNN_V': 1,      # 1 addition per MVM for digital correction and accumulation
    }

    # ADC scaling indicating number of ADCs per active column.
    bt_adc_scales = {
        'BNN_I': 0.5,
        'BNN_II': 0.5,
        'BNN_III': 1,
        'BNN_IV': 1,
        'BNN_V': 1,
        'BNN_VI': 0.5,
        'TNN_I': 0.5,
        'TNN_II': 0.5,
        'TNN_III': 0.5,
        'TNN_IV': 1,
        'TNN_V': 1,
    }

    # Number of MAC scaling based on mappings.
    # Active rows/cols to actual matrix sizes * logical MVMs per crossbar MVM
    bt_mac_scales = {
        'BNN_I':   0.5 * 1,
        'BNN_II':  0.5 * 1,
        'BNN_III': 1 * 0.5,
        'BNN_IV':  1 * 0.5,
        'BNN_V':   0.5 * 1,
        'BNN_VI':  0.25 * 1,
        'TNN_I':   0.25 * 1,
        'TNN_II':  0.5 * 0.5,
        'TNN_III': 0.5 * 0.5,
        'TNN_IV':  0.5 * 0.5,
        'TNN_V':   0.5 * 0.5,
    }

    arch_attrs: ArchAttrs = ArchAttrs(
        cell_config=cell_config,
        rows=xbar_size[1],
        cols=xbar_size[0],
        g_min=g_min,
        g_max=g_max,
        read_voltage=read_voltage,
        adc_resolution=adc_resolution,
        adder_col_scale=bt_adder_scales[m_mode],
        adc_col_scale=bt_adc_scales[m_mode],
        tech_node=tech_node,
        cycle_seconds=cycle_seconds
    )

    cem: CrossbarEnergyModel = CrossbarEnergyModel("crossbar", arch_attrs)
    component_names = [c.name for c in cem.components]
    energy_estimates: dict = {}
    for l_name, l_prof in mvm_profile.items():
        num_macs = 0
        tot_mvms = 0
        tot_cols = 0
        tot_rows = 0
        if l_name not in energy_estimates:
            energy_estimates[l_name] = defaultdict(float)
        for hists in l_prof:
            active_rows = hists["stratum"]["rows"]
            active_cols = hists["stratum"]["cols"]
            average_cell_value = hists["stratum"]["avg_cell_val"]
            for kv in hists["histogram"]["hist"]:
                if num_mvms := kv[1]:
                    average_input_value = kv[0]
                    mvm_attrs = MVMAttrs(active_rows,
                                         active_cols,
                                         average_cell_value,
                                         average_input_value)

                    num_macs += num_mvms * active_rows * \
                        active_cols * bt_mac_scales[m_mode]
                    tot_mvms += num_mvms
                    tot_cols += active_cols * num_mvms
                    tot_rows += active_rows * num_mvms
                    mvm_energy = cem.get_mvm_energy(mvm_attrs)
                    for c, e in mvm_energy.items():
                        energy_estimates[l_name][c] += e * num_mvms
        energy_estimates[l_name]["tot_energy"] = sum(
            [energy_estimates[l_name][cn] for cn in component_names])
        energy_estimates[l_name]["num_macs"] = num_macs
        energy_estimates[l_name]["num_mvms"] = tot_mvms
        energy_estimates[l_name]["col_util"] = tot_cols / \
            (tot_mvms * xbar_size[0])
        energy_estimates[l_name]["row_util"] = tot_rows / \
            (tot_mvms * xbar_size[1])

    return energy_estimates


def run_energy_estimation(df: pd.DataFrame,
                          profiles: dict[int, dict],
                          args) -> dict:
    """Run energy estimation for all MVM profiles in a configuration sweep."""
    energy_estimates: dict[int, dict] = {}
    for c, p in profiles.items():
        df_c = df[df['config_idx'] == c]
        if len(df_c) != 1:
            msg = f"Expected exactly one entry for config_idx={
                c}, got {len(df_c)}"
            raise ValueError(msg)
        row = df_c.iloc[0]

        def _parse(val):
            return ast.literal_eval(val) if isinstance(val, str) else val

        def _get_with_default(row, key, default):
            val = row.get(key, default)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return val

        xbar_size: tuple = tuple(_parse(row['xbar_size']))
        hrs_lrs: tuple = tuple(_parse(row['hrs_lrs']))
        m_mode: str = str(row['m_mode'])
        adc_resolution: int = int(_get_with_default(row, 'resolution', 8))
        read_voltage: float = float(_get_with_default(row, 'V_read', 0.2))
        cell_config = args.cell_config
        tech_node = args.tech_node * 1e-9
        cycle_seconds = args.cycle_period * 1e-9

        energy_estimates[c] = run_single_energy_estimation(p,
                                                           cell_config,
                                                           xbar_size,
                                                           hrs_lrs,
                                                           read_voltage,
                                                           adc_resolution,
                                                           m_mode,
                                                           tech_node,
                                                           cycle_seconds)

    return energy_estimates


def main(args):
    """Energy estimation utility that uses results of an MVM profiling
    run to compute per-layer energy estimates.

    The output is dumped as a JSON file to experiment results directory.
    """
    exp_name = args.config.split('/')[-1].split('.json')[0]
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    exp_result_path = repo_path + '/results/' + exp_name
    df = pd.read_csv(f"{exp_result_path}/{exp_name}.csv")
    store_path = os.path.join(exp_result_path, 'energy_estimates.json')
    profiles: dict[int, dict] = {
        c: json.load(open(f"{exp_result_path}/mvm_prof_{int(c)}.json", 'r'))
        for c in df.loc[:, "config_idx"]
    }

    energy_estimates = run_energy_estimation(df, profiles, args)

    with open(store_path, 'w') as json_out_file:
        json.dump(energy_estimates, json_out_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Path to experiment config',
                        required=True)

    parser.add_argument('--cell_config',
                        type=str,
                        help='Path to cell config',
                        required=True)

    parser.add_argument('--tech_node',
                        type=int,
                        help='Technology node in nanometers',
                        default=32)

    parser.add_argument('--cycle_period',
                        type=int,
                        help="Cycle period in nanoseconds",
                        default=100)

    args = parser.parse_args()
    main(args)
