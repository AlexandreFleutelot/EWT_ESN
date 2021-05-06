import pandas as pd
import numpy as np
import pywt
import torch
from datetime import datetime
from ESN import DeepESN


def dateparse(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def MSE(x, y):
    return np.mean((x - y) ** 2)


def load_data(file_path: str, start_date: str, end_date: str):
    df = pd.read_csv(file_path,
                     skiprows=1,
                     parse_dates=['Date'], date_parser=dateparse)
    mask = (df['Date'] > start_date) & (df['Date'] < end_date)
    df = df[mask]
    df = df.loc[::-1].reset_index(drop=True)
    return df


def get_stochastic(df: pd.DataFrame, sto_len: int, sto_ma: int, sto_drop_first: int):
    ma_high = df['High'].rolling(sto_len).max()
    ma_low = df['Low'].rolling(sto_len).min()
    df['%K'] = ((df['Close'] - ma_low) / (ma_high - ma_low)) - 0.5
    df['%D'] = df['%K'].rolling(sto_ma).mean()
    return df['%D'].values[sto_drop_first:]


def get_roc(df: pd.DataFrame, delay: int, drop_first: int):
    roc = df['Close'].pct_change(periods=delay)
    return roc.values[drop_first:]


def get_wavelets(signal, ewt_nb_waves=5, ewt_wave_mode='periodic', ewt_wave_type='dmey'):
    w = pywt.Wavelet(ewt_wave_type)
    coeffs = pywt.wavedec(signal, w, level=ewt_nb_waves, mode=ewt_wave_mode)

    waves = np.zeros((signal.shape[0], ewt_nb_waves))
    for n in range(ewt_nb_waves):
        n_coeffs = coeffs.copy()
        for i in range(len(n_coeffs)):
            if i != n:
                n_coeffs[i] = np.zeros_like(coeffs[i])
        waves[:, n] = pywt.waverec(n_coeffs, w)
    return waves


def load_nn(nn_cfg):
    ae_size, leaky_rate, density, rho = [], [], [], []
    input_scale, inter_scale, norm = [], [], []

    nb_layers = nn_cfg['nb_layers']
    nb_units = nn_cfg['nb_units']
    dim_in = torch.ones(nb_layers, dtype=torch.int)
    reservoir_size = torch.ones(nb_layers, dtype=torch.int) * nb_units

    for layer in range(nb_layers):
        ae_size += [nn_cfg['ae_size_' + str(layer)]]
        density += [nn_cfg['density_' + str(layer)]]
        norm += [nn_cfg['norm_' + str(layer)]]
        leaky_rate += [nn_cfg['leaky_rate_' + str(layer)]]
        rho += [nn_cfg['rho_' + str(layer)]]
        input_scale += [nn_cfg['input_scale_' + str(layer)]]
        inter_scale += [nn_cfg['inter_scale_' + str(layer)]]

    nn = DeepESN(n_layer=nb_layers, dim_in=dim_in,
                 reservoir_size=reservoir_size, ae_size=ae_size,
                 density=density, leaky_rate=leaky_rate, spectral_radius=rho,
                 input_scale=input_scale, inter_scale=inter_scale, norm=norm)

    return nn
