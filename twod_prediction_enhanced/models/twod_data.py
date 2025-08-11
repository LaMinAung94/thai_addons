from odoo import models, fields, api

class TwoDHistory(models.Model):
    _name = 'twod.history'
    _description = '2D History Data'

    date = fields.Date(required=True, index=True)
    time = fields.Char(required=True)
    value = fields.Float()
    set_number = fields.Float()
    twod_number = fields.Integer()

class TwoDLive(models.Model):
    _name = 'twod.live'
    _description = '2D Live Data'

    fetch_time = fields.Datetime(default=fields.Datetime.now)
    value = fields.Float()
    set_number = fields.Float()
    twod_number = fields.Integer()
    stock_date = fields.Date(string="Stock Date")
    stock_datetime = fields.Datetime(string="Stock Datetime")

class TwoDPrediction(models.Model):
    _name = 'twod.prediction'
    _description = '2D Prediction Result'

    prediction_time = fields.Datetime(default=fields.Datetime.now)
    target_time = fields.Char()
    value = fields.Float()
    set_number = fields.Integer()
    twod_number = fields.Integer()
    mae = fields.Float()
    mse = fields.Float()
    rmse = fields.Float()
    r2 = fields.Float()

class TwoDModelConfig(models.Model):
    _name = 'twod.model.config'
    _description = 'ML Model Configuration'

    model_type = fields.Selection([
        ('linear', 'Linear Regression'),
        ('poly', 'Polynomial Regression'),
        ('xgboost', 'XGBoost')
    ], default='linear', string="Model Type")
