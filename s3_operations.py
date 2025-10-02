import boto3
import os
import logging
import tempfile
from s3_config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME, AWS_S3_REGION

logger = logging.getLogger(__name__)

class S3Operations:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_S3_REGION
        )
        self.bucket_name = AWS_S3_BUCKET_NAME
        logger.info(f"✅ Cliente S3 inicializado para bucket: {self.bucket_name}")

    def download_image(self, filename, download_path, album_prefix="1/album"):
        """Baixar uma imagem específica do S3"""
        try:
            # Caminho no S3 (padrão: 1/album/, mas pode ser customizado)
            s3_key = f"{album_prefix}/{filename}"
            
            # Baixar arquivo
            self.s3_client.download_file(self.bucket_name, s3_key, download_path)
            logger.info(f"✅ Imagem baixada: {filename} de {album_prefix}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao baixar {filename} de {album_prefix}: {e}")
            return False

    def download_image_to_temp(self, filename, album_prefix="1/album"):
        """Baixar imagem para arquivo temporário e retornar o caminho"""
        try:
            # Criar arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
            temp_path = temp_file.name
            temp_file.close()
            
            # Baixar imagem
            if self.download_image(filename, temp_path, album_prefix):
                return temp_path
            else:
                # Limpar arquivo temporário se falhou
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao baixar imagem temporária {filename}: {e}")
            return None

    def image_exists(self, filename, album_prefix="1/album"):
        """Verificar se uma imagem existe no S3"""
        try:
            s3_key = f"{album_prefix}/{filename}"
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False

    def list_album_files(self, album_prefix="1/album"):
        """Listar todos os arquivos na pasta especificada do S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f'{album_prefix}/'
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extrair apenas o nome do arquivo (sem o prefixo)
                    filename = obj['Key'].replace(f'{album_prefix}/', '')
                    if filename:  # Ignorar se for apenas o prefixo
                        files.append(filename)
            
            logger.info(f"📁 Encontrados {len(files)} arquivos no álbum {album_prefix}")
            return files
            
        except Exception as e:
            logger.error(f"❌ Erro ao listar arquivos do S3: {e}")
            return []

    def list_all_albums(self):
        """Listar todos os álbuns disponíveis no S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Delimiter='/'
            )
            
            albums = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    album_name = prefix['Prefix'].rstrip('/')
                    albums.append(album_name)
            
            logger.info(f"📁 Encontrados {len(albums)} álbuns no S3")
            return albums
            
        except Exception as e:
            logger.error(f"❌ Erro ao listar álbuns do S3: {e}")
            return []

    def list_all_files(self):
        """Listar todos os arquivos do bucket S3"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
            
            logger.info(f"📁 Encontrados {len(files)} arquivos no bucket S3")
            return files
            
        except Exception as e:
            logger.error(f"❌ Erro ao listar arquivos do S3: {e}")
            return []

# Instância global
s3_ops = S3Operations()