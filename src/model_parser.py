##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
import re

from experiment import ExpConfig


def parse_model_string(file_name: str) -> dict:
    """Parse the model file name to extract the model data.
    Args:
        file_name (str): file name (without path and extension)
    Returns:
        dict: Parameter dict
    """
    pattern = r"(?P<type>^[a-z]+)_(?P<data>[a-z0-9]+)_(?P<model>[a-zA-Z0-9]+)_(?P<batch>b\d+)_mxn(?P<xbar>\d+x\d+)_inp(?P<ifm>\d+x\d+x\d+x\d+)"
    match = re.match(pattern, file_name)
    if match:
        raw_data = match.groupdict()

        ifm_shape_str = raw_data['ifm']
        ifm_shape_array = list(map(int, ifm_shape_str.split('x')))
        raw_data['ifm'] = ifm_shape_array

        xbar_dim_str = raw_data['xbar']
        xbar_dim_array = list(map(int, xbar_dim_str.split('x')))
        raw_data['xbar'] = xbar_dim_array

        raw_data['batch'] = int(raw_data['batch'][1:])
        return raw_data
    else:
        raise ValueError("Unknown format.")


def create_experiment(cfg: dict) -> ExpConfig:
    exp = ExpConfig(
        nn_names=cfg['nn_names'],
        ifm=cfg['ifm'],
        nn_data_set=cfg['nn_data_set'],
        nn_data=cfg['nn_data'],
        batch=cfg['batch'],
        num_runs=cfg['num_runs'],
        xbar_size=cfg['xbar_size'],
        digital_only=cfg['digital_only'],
        hrs_lrs=cfg.get('hrs_lrs'),
        gmin_gmax=cfg.get('gmin_gmax'),
        adc_type=cfg['adc_type'],
        m_mode=cfg['m_mode'],
        hrs_noise=cfg['hrs_noise'],
        lrs_noise=cfg['lrs_noise'],
        verbose=cfg['verbose'],
        resolution=cfg.get('resolution'),
        adc_profile=cfg.get('adc_profile'),
        adc_calib_mode=cfg.get('adc_calib_mode'),
        adc_calib_dict=cfg.get('adc_calib_dict'),
        read_disturb=cfg.get('read_disturb'),
        V_read=cfg.get('V_read'),
        t_read=cfg.get('t_read'),
        read_disturb_update_freq=cfg.get('read_disturb_update_freq'),
        read_disturb_mitigation_strategy=cfg.get(
            'read_disturb_mitigation_strategy'),
        read_disturb_mitigation_fp=cfg.get('read_disturb_mitigation_fp'),
        read_disturb_update_tolerance=cfg.get('read_disturb_update_tolerance'),
        parasitics=cfg.get('parasitics'),
        w_res=cfg.get('w_res'),
        c2c_var=cfg.get('c2c_var'),
        mvm_profile=cfg.get('mvm_profile'))

    for key in cfg.keys():
        if not hasattr(exp, key):
            raise Exception(
                f"Config parameter {key} not supported in ExpConfig.")
    return exp


def get_model_name(cfg: str) -> str:
    if cfg['m_mode'].startswith('BNN'):
        mode = 'B'
    elif cfg['m_mode'].startswith('TNN'):
        mode = 'T'
    else:
        raise ValueError("Unknown mode.")

    # Mappable logical matrix sizes differ based on mappings
    mappable_col_scales = {
        "BNN_I": 0.5,
        "BNN_II": 0.5,
        "BNN_III": 1.0,
        "BNN_IV": 1.0,
        "BNN_V": 1.0,
        "BNN_VI": 0.5,
        "TNN_I": 0.5,
        "TNN_II": 0.5,
        "TNN_III": 0.5,
        "TNN_IV": 0.5,
        "TNN_V": 0.5,
    }

    mappable_row_scales = {
        "BNN_I": 1.0,
        "BNN_II": 1.0,
        "BNN_III": 1.0,
        "BNN_IV": 1.0,
        "BNN_V": 0.5,
        "BNN_VI": 0.5,
        "TNN_I": 0.5,
        "TNN_II": 1.0,
        "TNN_III": 1.0,
        "TNN_IV": 1.0,
        "TNN_V": 1.0,
    }

    m_matrix_mappable = int(cfg['xbar_size'][0] *
                            mappable_col_scales[cfg['m_mode']])
    n_matrix_mappable = int(cfg['xbar_size'][1] *
                            mappable_row_scales[cfg['m_mode']])

    model_name = f"{cfg['nn_data_set']}_{mode}_{cfg['nn_name']}_b{cfg['batch']}_mxn{m_matrix_mappable}x{
        n_matrix_mappable}_inp{cfg['batch']}x{cfg['ifm'][0]}x{cfg['ifm'][1]}x{cfg['ifm'][2]}.so"
    return model_name
