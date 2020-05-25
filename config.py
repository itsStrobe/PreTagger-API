import os

class EnvVariables:
    # AMAZON AWS
    BUCKET_NAME = os.environ.get('BUCKET_NAME')
    REGION_NAME = os.environ.get('REGION_NAME')
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

    # SERVER SETTINGS
    PORT = os.environ.get('PORT')
