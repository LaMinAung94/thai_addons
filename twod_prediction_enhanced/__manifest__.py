{
    'name': '2D Prediction Enhanced',
    'version': '3.0',
    'summary': '2D Prediction with Multi-Feature ML, Tailwind Dashboard, Accuracy Metrics',
    'author': 'Your Name',
    'depends': ['base', 'web', 'sale'],
    'data': [
        'security/ir.model.access.csv',
        'views/twod_prediction_views.xml',
        'views/twod_dashboard_views.xml',

        'data/ir_cron.xml',
    ],
    'installable': True,
    'application': True,
}
