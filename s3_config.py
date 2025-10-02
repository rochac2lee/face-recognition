import os
from dotenv import load_dotenv

load_dotenv()

# Configurações AWS S3
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.getenv('AWS_S3_REGION', 'us-east-1')

# URL base do S3 (usar CDN se disponível)
STATIC_CDN = os.getenv('STATIC_CDN')
S3_BASE_URL = STATIC_CDN if STATIC_CDN else f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com"