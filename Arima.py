import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ARIMA:
    max_iter: int = 200
    d: Optional[int] = field(default=None)
    p: Optional[int] = field(default=None)
    q: Optional[int] = field(default=None)
    phi: Optional[np.ndarray] = field(default=None)
    theta: Optional[np.ndarray] = field(default=None)
    x_diff: Optional[np.ndarray] = field(default=None)
    residuals: Optional[np.ndarray] = field(default=None)

    def _difference(self, x, max_diff=10, significance=0.05):
        d = 0
        current = x.copy()
        for _ in range(max_diff):
            result = adfuller(current)
            if result[1] < significance:
                return d, current
            current = np.diff(current)
            d += 1
        return d, current

    def _inverse_difference(self, original, forecast):
        reconstructed = forecast.copy()
        for i in range(self.d):
            last_value = original[-(self.d - i)]
            reconstructed = np.cumsum(reconstructed) + last_value
        return reconstructed

    def _compute_acf(self, x, max_lag):
        x = x - np.mean(x)
        N = len(x)
        autocorr = np.zeros(max_lag + 1)
        denominator = np.dot(x, x)
        for lag in range(max_lag + 1):
            numerator = np.dot(x[:N - lag], x[lag:N])
            autocorr[lag] = numerator / (denominator * (N - lag) / N)
        return autocorr

    def _compute_pacf(self, x, max_lag):
        x = x - np.mean(x)
        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1
        full_X = [x[max_lag - i: -i] for i in range(1, max_lag + 1)]
        full_X = np.column_stack(full_X)
        y = x[max_lag:]
        for k in range(1, max_lag + 1):
            X = full_X[:, :k]
            phi, *_ = np.linalg.lstsq(X, y, rcond=None)
            pacf[k] = phi[-1]
        return pacf

    def _choose_pq(self, x):
        N = len(x)
        threshold = 1.96 / np.sqrt(N)
        acf = self._compute_acf(x, max_lag=int(N / 100))
        pacf = self._compute_pacf(x, max_lag=int(N / 100))

        def threshold_idx(data):
            for i in range(1, len(data)):
                if all(abs(data[j]) < threshold for j in range(i, min(i + 2, len(data)))):
                    return i
            return len(data) - 1

        p = threshold_idx(pacf)
        q = threshold_idx(acf)
        return p, q

    def _train_ar(self, x, p):
        x = x - np.mean(x)
        y = x[p:]
        X = np.column_stack([x[p - i - 1: len(x) - i - 1] for i in range(p)])
        phi, *_ = np.linalg.lstsq(X, y, rcond=None)
        return phi

    def _train_ma(self, x, q):
        def loss_func(theta):
            # Calc x_hat and eps
            x_hat, eps = self._predict_ma(x, theta)
            # Calc MSE
            mse = np.mean(np.square(x[q:] - x_hat[q:]))  # must skip first q
            return mse

        def callback(theta):
            pbar.update(1)

        # start theta with 0's
        init_theta = np.zeros(q)
        pbar = tqdm(total=self.max_iter, desc="Fitting MA(q)", ncols=80)

        # use minimize and loss to get theta
        # logger.info('Fitting MA model')
        result = minimize(loss_func, init_theta, method='L-BFGS-B', callback=callback,
                          options={"maxiter": self.max_iter})

        pbar.close()
        # return result
        return result.x

    def _predict_ma(self, x, theta):
        q = len(theta)
        N = len(x)
        mu = np.mean(x)
        eps = np.zeros(N)
        x_hat = np.zeros(N)
        for t in range(q, N):
            err_slice = eps[t - q:t][::-1]
            x_hat[t] = mu + np.dot(theta, err_slice)
            eps[t] = x.iloc[t] - x_hat[t]
        return x_hat, eps

    def fit(self, x):
        self.d, self.x_diff = self._difference(x)
        self.p, self.q = self._choose_pq(self.x_diff)
        self.phi = self._train_ar(self.x_diff, self.p)
        self.theta = self._train_ma(x, self.q)
        N = len(self.x_diff)
        self.residuals = np.zeros(N)
        x_hat = np.zeros(N)
        for t in range(max(self.p, self.q), N):
            ar_terms = self.x_diff[t - self.p:t][::-1]
            ma_terms = self.residuals[t - self.q:t][::-1]
            ar_pred = np.dot(self.phi, ar_terms)
            ma_pred = np.dot(self.theta, ma_terms)
            x_hat[t] = ar_pred + ma_pred
            self.residuals[t] = self.x_diff[t] - x_hat[t]

    def forecast(self, forecast_horizon):
        N = len(self.x_diff)
        x_forecast = np.concatenate([self.x_diff, np.zeros(forecast_horizon)])
        residuals = np.concatenate([self.residuals, np.zeros(forecast_horizon)])

        for t in range(1, forecast_horizon + 1):
            ar_window = x_forecast[N + t - self.p - 1:N + t - 1][::-1]
            ma_window = residuals[N + t - self.q - 1:N + t - 1][::-1]
            ar_pred = np.dot(self.phi, ar_window) if self.p else 0
            ma_pred = np.dot(self.theta, ma_window) if self.q else 0
            x_forecast[N + t - 1] = ar_pred + ma_pred

        return x_forecast[-forecast_horizon:]

    def forecast_original_scale(self, x_original, forecast_horizon):
        forecast_diff = self.forecast(forecast_horizon)
        return self._inverse_difference(x_original, forecast_diff)

    def build_forecast_dataframe(self, forecast_diff, original_series, forecast_horizon, include_inverse=True):
        forecast_diff = np.asarray(forecast_diff[-forecast_horizon:])

        if isinstance(original_series, pd.Series):
            last_index = original_series.index[-1]
            if isinstance(last_index, (pd.Timestamp, pd.DatetimeIndex)):
                forecast_index = pd.date_range(start=last_index, periods=forecast_horizon + 1, freq='h')[1:]
            else:
                forecast_index = np.arange(len(original_series), len(original_series) + forecast_horizon)
        else:
            forecast_index = np.arange(len(original_series), len(original_series) + forecast_horizon)

        df = pd.DataFrame(index=forecast_index)

        if include_inverse:
            full_original = original_series.values if isinstance(original_series, pd.Series) else original_series
            df['forecast'] = self._inverse_difference(full_original, forecast_diff)
        else:
            df['forecast'] = forecast_diff

        return df

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def summary(self):
        print(f"ARIMA(p={self.p}, d={self.d}, q={self.q})")
        print(f"phi: {self.phi}")
        print(f"theta: {self.theta}")
