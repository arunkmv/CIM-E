from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os

import hwcomponents as hwc
from hwcomponents_neurosim import MemoryCell, RowDrivers, ShiftAdd
from hwcomponents_adc import ADC
import yaml


@dataclass(frozen=True)
class ArchAttrs():
    """Architecture attributes."""
    # Non-optional fields
    cell_config: str
    rows: int
    cols: int
    v_read: float
    g_min: float
    g_max: float

    # Optional fields
    read_latency: float = 1e-07
    read_pulse_width: float = 1e-08
    cycle_seconds: float = 1e-07                
    voltage: int = 1
    threshold_voltage: float = 0
    tech_node: float = 65e-09
    adc_resolution: int = 8
    adder_scale: float = 0

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
        object.__setattr__(self, 'throughput', 1 / self.read_latency)
        self._update_cell_config()

    def _update_cell_config(self):
        """
        Modify cell config read voltage/read pulse/resistance
        entries with values derived from this ArchAttrs instance.
        """
        with open(self.cell_config, 'r') as f:
            config = yaml.safe_load(f)

        # -ReadVoltage (V)
        config['-ReadVoltage (V)'] = self.v_read
        # -ReadPulse (ns): read_pulse_width is stored in seconds
        config['-ReadPulse (ns)'] = self.read_pulse_width * 1e9
        # -ResistanceOn / -ResistanceOff (ohm): convert conductance -> resistance.
        config['-ResistanceOn (ohm)'] = 1.0 / self.g_max
        config['-ResistanceOff (ohm)'] = 1.0 / self.g_min

        # Dump YAML
        orig_path = self.cell_config
        parent_dir = os.path.dirname(orig_path)
        base_name = os.path.basename(orig_path)
        stem, ext = os.path.splitext(base_name)
        new_path = os.path.join(parent_dir, f"{stem}_temp{ext}")

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
            'adc_resolution': self.adc_resolution,
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
        row_driver_comp = RowDrivers(**self._get_neurosim_component_kwargs(mvm_attrs))
        return row_driver_comp.read().energy
        
    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        row_driver_comp = RowDrivers(**self._get_neurosim_component_kwargs(mvm_attrs))
        return row_driver_comp.write().energy


class MemristorEnergyModel(BaseEnergyModel):
    """Memristor energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        memory_cell_comp = MemoryCell(**self._get_neurosim_component_kwargs(mvm_attrs))
        return memory_cell_comp.read().energy * mvm_attrs.active_rows * mvm_attrs.active_rows

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        memory_cell_comp = MemoryCell(**self._get_neurosim_component_kwargs(mvm_attrs))
        return memory_cell_comp.write().energy * mvm_attrs.active_rows * mvm_attrs.active_rows


class ADCEnergyModel(BaseEnergyModel):
    """ADC energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        adc_comp = ADC(**self._arch_attrs.get_adc_component_kwargs())
        return adc_comp.convert().energy * mvm_attrs.active_cols

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        return 0.0


class AdderEnergyModel(BaseEnergyModel):
    """Shift Adder energy model."""

    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> float:
        adder_comp = ShiftAdd(**self._arch_attrs.get_shift_adder_components_kwargs())
        return adder_comp.add().energy * self._arch_attrs.adder_scale * mvm_attrs.active_cols

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> float:
        return 0.0


class CrossbarEnergyModel(BaseEnergyModel):
    """Crossbar energy model.

    Integrates sub-components and provides per component energy.
    """

    def __init__(self, name: str, arch_attrs: ArchAttrs):
        super().__init__(name, arch_attrs)
        self.row_driver = RowDriverEnergyModel(name + ".row_driver", arch_attrs)
        self.memristors = MemristorEnergyModel(name + ".memristors", arch_attrs)
        self.adcs = ADCEnergyModel(name + ".adcs", arch_attrs)
        self.adders = AdderEnergyModel(name + ".adders", arch_attrs)
        self.components = [self.row_driver, self.memristors, self.adcs, self.adders]
        
    def get_mvm_energy(self, mvm_attrs: MVMAttrs) -> dict:
        return {c.name : c.get_mvm_energy(mvm_attrs) for c in self.components}

    def get_write_energy(self, mvm_attrs: MVMAttrs) -> dict:
        return {c.name : c.get_write_energy(mvm_attrs) for c in self.components}

        

if __name__=="__main__":
    arch_attrs: ArchAttrs = ArchAttrs(
    cell_config="configs/memory_cells/rram_base.yaml",
    rows=128,
    cols=128,
    v_read=1,
    g_min=2.5e-6,
    g_max=20e-6,
    adder_scale=1
)
    xbar = CrossbarEnergyModel("crossbar", arch_attrs)
    print(xbar.get_mvm_energy(MVMAttrs(128, 128, 0.5, 0.5)))
