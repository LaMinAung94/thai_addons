from odoo import http
from odoo.http import request

class TwoDPredictAPI(http.Controller):

    @http.route('/api/twod/predict', type='json', auth='public')
    def get_prediction(self, time):
        pred = request.env['twod.prediction'].sudo().search(
            [('target_time', '=', time)], order='prediction_time desc', limit=1
        )
        if not pred:
            return {'error': 'No prediction found'}
        return {
            'time': pred.target_time,
            'twod': pred.twod_number,
            'accuracy': 100 - pred.mae,
            'model': pred._origin.env['twod.model.config'].search([], limit=1).model_type
        }
