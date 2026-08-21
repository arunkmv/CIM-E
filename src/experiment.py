##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
from typing import List, Tuple, Optional, Union, get_type_hints, Dict
from dataclasses import dataclass
import numpy as np
from itertools import product
import json


@dataclass
class ExpConfig:
    nn_names: List[str]
    ifm: List[List[int]]
    nn_data_set: str
    nn_data: str
    batch: int
    num_runs: int
    xbar_size: List[Tuple[int, int]]
    digital_only: bool
    hrs_lrs: Optional[List[Tuple[float]]]
    gmin_gmax: Optional[List[Tuple[float]]]
    adc_type: str
    hrs_noise: List[float]
    lrs_noise: List[float]
    verbose: bool
    m_mode: List[str]
    resolution: Optional[List[int]] = None
    adc_profile: Optional[int] = None
    adc_calib_mode: Optional[List[str]] = None
    adc_calib_dict: Optional[Dict] = None
    read_disturb: Optional[bool] = None
    V_read: Optional[List[float]] = None
    t_read: Optional[List[float]] = None
    read_disturb_update_freq: Optional[int] = None
    read_disturb_mitigation_strategy: Optional[str] = None
    read_disturb_mitigation_fp: Optional[List[float]] = None
    read_disturb_update_tolerance: Optional[List[float]] = None
    parasitics: Optional[bool] = None
    w_res: Optional[List[float]] = None
    c2c_var: Optional[bool] = None
    mvm_profile: Optional[int] = None

    def _check_paramters(self):
        if self.nn_data_set not in ["cifar10", "cifar100", "mnist"]:
            raise ValueError("nn_data_set not supported.")
        for ifm_shape in self.ifm:
            for elem in ifm_shape:
                if elem <= 0:
                    raise ValueError("ifm values should be greater than 0")
        if self.nn_data not in ["TEST", "TRAIN"]:
            raise ValueError("nn_data should be either 'TEST' or 'TRAIN'")
        if self.batch <= 0:
            raise ValueError("batch should be greater than 0")
        if self.num_runs <= 0:
            raise ValueError("num_runs should be greater than 0")
        for (m, n) in self.xbar_size:
            if m <= 0 or n <= 0:
                raise ValueError("xbar_size should be greater than 0")

        if self.hrs_lrs is not None:
            for (hrs, lrs) in self.hrs_lrs:
                if hrs < 0.0 or lrs <= 0.0 or hrs >= lrs:
                    raise ValueError(
                        "error in hrs_lrs should be greater than 0")
        else:
            if self.gmin_gmax is None:
                raise ValueError(
                    "Either hrs_lrs or gmin_gmax should be provided.")
            else:
                if self.V_read is None:
                    raise ValueError(
                        "V_read should be provided when gmin_gmax is used.")
                for (gmin, gmax) in self.gmin_gmax:
                    if gmin < 0.0 or gmax <= 0.0 or gmin >= gmax:
                        raise ValueError(
                            "Error in gmin_gmax: should be greater than 0")

        # Check ADC parameters
        if self.adc_type not in ["FP_ALPHA_ADC", "INF_ADC"]:
            raise ValueError("adc_type not valid.")
        if self.adc_type != "INF_ADC":
            for r in self.resolution:
                if r != -1 and r <= 0:
                    raise ValueError("resolution should be greater than 0")
            if self.adc_calib_mode is not None:
                for acm in self.adc_calib_mode:
                    if acm not in ["MAX", "CALIB"]:
                        raise ValueError(
                            f"Unknown ADC calibration mode: {acm}.")
                    if acm == "CALIB" and self.adc_calib_dict is None:
                        raise ValueError(
                            "Calibrated ADC requires calibration limits.")

        for mode in self.m_mode:
            if mode not in [
                    "BNN_I", "BNN_II", "BNN_III", "BNN_IV", "BNN_V", "BNN_VI",
                    "TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"
            ]:
                raise ValueError(f"m_mode {mode} not valid.")
        for noise in self.hrs_noise:
            if noise < 0.0:
                raise ValueError("hrs_noise should be greater than 0")
        for noise in self.lrs_noise:
            if noise < 0.0:
                raise ValueError("lrs_noise should be greater than 0")

        # Check read disturb parameters
        if (self.read_disturb):
            if self.V_read is None or self.t_read is None:
                raise ValueError(
                    "V_read and t_read should be provided when read_disturb is True"
                )
            for v in self.V_read:
                if v >= 0.0:
                    raise ValueError(
                        "Read disturb model requires negative V_read values")
            for t in self.t_read:
                if t <= 0.0:
                    raise ValueError("t_read should be greater than 0")
            if self.read_disturb_update_freq is not None:
                for f in self.read_disturb_update_freq:
                    if f <= 0:
                        raise ValueError(
                            "read_disturb_update_freq should be greater than 0 (minimum: 1)"
                        )
            if self.read_disturb_mitigation_strategy is not None:
                if self.read_disturb_mitigation_strategy == "SOFTWARE":
                    if self.read_disturb_mitigation_fp is None:
                        raise ValueError(
                            "read_disturb_mitigation_fp should be provided for SOFTWARE strategy"
                        )
                    else:
                        for fp in self.read_disturb_mitigation_fp:
                            if fp < 1.0:
                                raise ValueError(
                                    "read_disturb_mitigation_fp must be at least 1.0."
                                )

                elif self.read_disturb_mitigation_strategy == "CELL_BASED":
                    if self.read_disturb_update_tolerance is None:
                        raise ValueError(
                            "read_disturb_update_tolerance should be provided for CELL_BASED strategy"
                        )
                elif self.read_disturb_mitigation_strategy != "OFF":
                    raise ValueError(
                        "read_disturb_mitigation_strategy should be either 'OFF', 'SOFTWARE', or 'CELL_BASED'"
                    )

        # Check parasitics parameters
        if (self.parasitics):
            if self.w_res is None:
                raise ValueError(
                    "w_res should be provided when parasitics is True.")
            for res in self.w_res:
                if res < 0.0:
                    raise ValueError("w_res should be non-negative.")
            if self.V_read is None:
                raise ValueError(
                    "V_read should be provided when parasitics is True.")
            for v in self.V_read:
                if v >= 0.0:
                    raise ValueError(
                        "Parasitics model requires negative V_read values")

    def __post_init__(self):
        type_hints = get_type_hints(self.__class__)
        for field_name, field_type in type_hints.items():
            value = getattr(self, field_name)

            is_optional = (getattr(field_type, '__origin__', None) is Union
                           and type(None) in field_type.__args__)

            if not is_optional and value is None:
                raise ValueError(
                    f"Argument '{field_name}' is missing in ExpConfig. Please provide all required arguments."
                )
        self._check_paramters()

    def iterate_sweep(self) -> list:
        """Generate all possible sweep configurations."""
        cfg = []
        static_fields = {
            key: value
            for key, value in {
                'nn_data_set': self.nn_data_set,
                'nn_data': self.nn_data,
                'batch': self.batch,
                'num_runs': self.num_runs,
                'digital_only': self.digital_only,
                'adc_type': self.adc_type,
                'adc_profile': self.adc_profile,
                'adc_calib_dict': self.adc_calib_dict,
                'verbose': self.verbose,
                'mvm_profile': self.mvm_profile,
                'read_disturb': self.read_disturb,
                'read_disturb_mitigation_strategy': self.read_disturb_mitigation_strategy,
                'parasitics': self.parasitics,
                'c2c_var': self.c2c_var,
            }.items() if value is not None
        }
        iterable_fields = {
            key: value
            for key, value in {
                'nn_name': self.nn_names,
                'xbar_size': self.xbar_size,
                'hrs_lrs': self.hrs_lrs,
                'gmin_gmax': self.gmin_gmax,
                'resolution': self.resolution,
                'adc_calib_mode': self.adc_calib_mode,
                'm_mode': self.m_mode,
                'hrs_noise': self.hrs_noise,
                'lrs_noise': self.lrs_noise,
                'V_read': self.V_read,
                't_read': self.t_read,
                'read_disturb_update_freq': self.read_disturb_update_freq,
                'read_disturb_mitigation_fp': self.read_disturb_mitigation_fp,
                'read_disturb_update_tolerance': self.read_disturb_update_tolerance,
                'w_res': self.w_res
            }.items() if value is not None
        }
        iterable_fields = {k: v for k,
                           v in iterable_fields.items() if v != None}
        for combination in product(*iterable_fields.values()):
            config_entry = {**static_fields}
            for key, value in zip(iterable_fields.keys(), combination):
                config_entry[key] = value
                nn_idx = [
                    idx for idx, i in enumerate(self.nn_names)
                    if i == config_entry['nn_name']
                ][0]
                config_entry['ifm'] = self.ifm[nn_idx]
            cfg.append(config_entry)

        if len(cfg) == 0:
            raise Exception("Could not iterate experiment sweep!")
        return cfg

    @staticmethod
    def dump_acs_config(cfg: dict, file_name: str) -> dict:
        """Dump a single ACS JSON configuration file"""
        if cfg['adc_type'] == "INF_ADC":
            adc_type = "INF_ADC"
        else:
            if cfg['m_mode'] in [
                    'BNN_I', 'BNN_II', 'BNN_VI', 'TNN_I', 'TNN_II', 'TNN_III'
            ]:
                adc_type = "SYM_RANGE_ADC"
            elif cfg['m_mode'] in [
                    'BNN_III', 'BNN_IV', 'BNN_V', 'TNN_IV', 'TNN_V'
            ]:
                adc_type = "POS_RANGE_ONLY_ADC"
            else:
                raise ValueError("m_mode not supported")

        # Required parameters (not optional)
        acs_data = {
            "M":
            cfg['xbar_size'][0],
            "N":
            cfg['xbar_size'][1],
            "digital_only":
            cfg['digital_only'],
            "HRS":
            cfg['hrs_lrs'][0] if 'hrs_lrs' in cfg.keys() else cfg['gmin_gmax'][0] *
            abs(cfg['V_read']),
            "LRS":
            cfg['hrs_lrs'][1] if 'hrs_lrs' in cfg.keys() else cfg['gmin_gmax'][1] *
            abs(cfg['V_read']),
            "adc_type":
            adc_type,
            "m_mode":
            cfg['m_mode'],
            "HRS_NOISE":
            cfg['hrs_noise'],
            "LRS_NOISE":
            cfg['lrs_noise'],
            "verbose":
            cfg['verbose']
        }

        # Optional parameters
        if cfg.get('resolution') is not None:
            acs_data["resolution"] = cfg['resolution']
        if cfg.get('adc_profile') is not None:
            acs_data["adc_profile"] = cfg['adc_profile'] > 0
            if acs_data['adc_profile']:
                acs_data["adc_profile_bin_size"] = cfg['adc_profile']
        if cfg.get('adc_calib_mode') is not None:
            acs_data["adc_calib_mode"] = cfg['adc_calib_mode']
        if cfg.get('adc_calib_dict') is not None:
            acs_data["adc_calib_dict"] = cfg['adc_calib_dict'][cfg['nn_name']][str(
                cfg['xbar_size'])][cfg['m_mode']]
        if cfg.get('read_disturb') is not None:
            acs_data["read_disturb"] = cfg['read_disturb']
        if cfg.get('V_read') is not None:
            acs_data["V_read"] = cfg['V_read']
        if cfg.get('t_read') is not None:
            acs_data["t_read"] = cfg['t_read']
        if cfg.get('read_disturb_update_freq') is not None:
            acs_data["read_disturb_update_freq"] = cfg['read_disturb_update_freq']
        if cfg.get('read_disturb_mitigation_strategy') is not None:
            acs_data["read_disturb_mitigation_strategy"] = cfg[
                'read_disturb_mitigation_strategy']
        if cfg.get('read_disturb_mitigation_fp') is not None:
            acs_data["read_disturb_mitigation_fp"] = cfg[
                'read_disturb_mitigation_fp']
        if cfg.get('read_disturb_update_tolerance') is not None:
            acs_data["read_disturb_update_tolerance"] = cfg[
                'read_disturb_update_tolerance']
        if cfg.get('parasitics') is not None:
            acs_data["parasitics"] = cfg['parasitics']
        if cfg.get('w_res') is not None:
            acs_data["w_res"] = cfg['w_res']
        if cfg.get('c2c_var') is not None:
            acs_data["c2c_var"] = cfg['c2c_var']
        if cfg.get('mvm_profile') is not None:
            acs_data["mvm_profile"] = cfg['mvm_profile'] > 0
            if acs_data['mvm_profile']:
                acs_data["mvm_profile_bin_size"] = cfg['mvm_profile']

        if cfg['m_mode'] in ['TNN_IV', 'TNN_V']:
            acs_data["SPLIT"] = [1, 1]
            acs_data["W_BIT"] = 2

        with open(f"{file_name}", "w") as f:
            json.dump(acs_data, f, indent=4)

        return acs_data


@dataclass
class SimulationStats:
    config: dict
    config_idx: int
    cycles_p: np.ndarray
    cycles_m: np.ndarray
    write_ops: int
    mvm_ops: int
    refresh_ops: int
    refresh_cell_ops: int
