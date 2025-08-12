import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from odoo import models, api
from datetime import datetime,time,date
import logging
_logger = logging.getLogger(__name__)

HISTORY_URL = "https://api.thaistock2d.com/history"
LIVE_URL = "https://api.thaistock2d.com/live"

class TwoDPredictionService(models.TransientModel):
    _name = 'twod.prediction.service'
    _description = '2D Prediction Service'

    def fetch_history(self):
        try:
            return requests.get(HISTORY_URL, timeout=10).json()
        except Exception as e:
            self.env['ir.logging'].sudo().create({
                'name': 'TwoDPredictionService',
                'type': 'server',
                'dbname': self.env.cr.dbname,
                'level': 'error',
                'message': f'Error fetching history data: {e}',
                'path': 'twod.prediction.service',
                'func': 'fetch_history',
            })
            return []

    def fetch_live(self):
        try:
            return requests.get(LIVE_URL, timeout=10).json()
        except Exception as e:
            self.env['ir.logging'].sudo().create({
                'name': 'TwoDPredictionService',
                'type': 'server',
                'dbname': self.env.cr.dbname,
                'level': 'error',
                'message': f'Error fetching live data: {e}',
                'path': 'twod.prediction.service',
                'func': 'fetch_live',
            })
            return {}

    @api.model
    def update_history_data(self):
        data = self.fetch_history()
        for day in data:
            date = day.get('date')
            children = day.get('child', [])
            for entry in children:
                set_number = (entry.get('set'))
                value = (entry.get('value'))
                twod_str = entry.get('twod')
                try:
                    twod_number = int(twod_str)
                except:
                    twod_number = 0
                self.env['twod.history'].create({
                    'date': date,
                    'time': entry.get('time'),
                    'value': value,
                    'set_number': set_number,
                    'twod_number': twod_number,
                })

    @api.model
    def update_live_data(self):
        today_weekday = datetime.today().weekday()
        if today_weekday >= 5:  # Saturday (5) and Sunday (6)
            _logger.info("update_live_data skipped on weekend")
            return
        response = self.fetch_live()
        if not response or 'live' not in response:
            return

        def to_float(value):
            try:
                return float(str(value).replace(',', ''))
            except:
                return 0.0

        live_data = response['live']
        print(live_data)
        set_number = to_float(live_data.get('set'))
        value = to_float(live_data.get('value'))
        twod_str = live_data.get('twod')
        twod_number = int(twod_str) if twod_str and twod_str.isdigit() else 0

        self.env['twod.live'].create({
            'value': value,
            'set_number': set_number,
            'twod_number': twod_number,
            'stock_date': live_data.get('date'),
            'stock_datetime': live_data.get('time'),
        })

    def _prepare_features(self, history_data):
        return np.array([[d['value'], d['set_number'], d['twod_number']] for d in history_data])

    def _evaluate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    def linear_model(self, X, y, live):
        model = LinearRegression().fit(X, y)
        return model.predict([live])[0]

    def poly_model(self, X, y, live):
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        live_poly = poly.transform([live])
        model = LinearRegression().fit(X_poly, y)
        return model.predict(live_poly)[0]

    def xgb_model(self, X, y, live):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        return model.predict([live])[0]

    @api.model
    def predict_twod(self, model_type='linear', target_time_value=None, cutoff_time_value=None):
        today = date.today()
        target_time = datetime.combine(today, target_time_value or time(12, 1))

        domain = [('stock_date', '=', today)]
        if cutoff_time_value:
            cutoff_datetime = datetime.combine(today, cutoff_time_value)
            domain.append(('stock_datetime', '<=', cutoff_datetime))

        history = self.env['twod.live'].search(domain, order='stock_datetime desc')

        live = self.env['twod.live'].search(domain, order='stock_datetime desc', limit=1)

        if not history or not live:
            _logger.warning(f"No history or live data before cutoff - {model_type} - {target_time}")
            return

        history_data = [{
            'value': float(h.value.replace(',', '')) if isinstance(h.value, str) else h.value,
            'set_number': float(h.set_number.replace(',', '')) if isinstance(h.set_number, str) else h.set_number,
            'twod_number': int(h.twod_number) if isinstance(h.twod_number, str) else h.twod_number,
        } for h in history]

        X = self._prepare_features(history_data)
        y = np.array([h['twod_number'] for h in history_data])

        live_features = [
            float(live.value.replace(',', '')) if isinstance(live.value, str) else live.value,
            float(live.set_number.replace(',', '')) if isinstance(live.set_number, str) else live.set_number,
            int(live.twod_number) if isinstance(live.twod_number, str) else live.twod_number,
        ]

        if model_type == 'poly':
            prediction = self.poly_model(X, y, live_features)
        elif model_type == 'xgboost':
            prediction = self.xgb_model(X, y, live_features)
        else:
            prediction = self.linear_model(X, y, live_features)

        mae, mse, rmse, r2 = self._evaluate_metrics(y, [prediction] * len(y))

        self.env['twod.prediction'].create({
            'target_time': target_time,
            'value': live.value,
            'set_number': live.set_number,
            'twod_number': round(prediction, 2),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'model_type': model_type
        })

    # Cron methods
    @api.model
    def cron_update_history_data(self):
        self.update_history_data()

    @api.model
    def cron_update_live_data(self):
        self.update_live_data()

    @api.model
    def cron_predict_all_models(self):
        model_types = ['linear', 'poly', 'xgboost']
        target_times_with_cutoff = [
            (time(12, 1),time(10, 30)),
            (time(16, 30),time(14, 0)),
        ]
        for t_time, cutoff in target_times_with_cutoff:
            for m_type in model_types:
                try:
                    self.predict_twod(m_type, t_time, cutoff)
                    _logger.info(f"Prediction done for model: {m_type} at {t_time} with cutoff {cutoff}")
                except Exception as e:
                    _logger.error(f"Error in prediction with {m_type} at {t_time}: {e}")



