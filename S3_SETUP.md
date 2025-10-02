# Configuração S3

O sistema de reconhecimento facial funciona principalmente com arquivos locais, mas o S3 está disponível como opção para casos específicos.

## Como Habilitar S3

### 1. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Configurações AWS S3
AWS_ACCESS_KEY_ID=sua_access_key_aqui
AWS_SECRET_ACCESS_KEY=sua_secret_key_aqui
AWS_S3_BUCKET_NAME=seu_bucket_name
AWS_S3_REGION=us-east-1
STATIC_CDN=https://seu-cdn.com
```

### 2. Instalar Dependências S3

```bash
pip install boto3 python-dotenv
```

### 3. Verificar Disponibilidade

Acesse o endpoint `/health` para verificar se o S3 está disponível:

```json
{
  "status": "healthy",
  "engine_available": true,
  "s3_available": true,
  "database_stats": {...}
}
```

## Funcionalidades S3

Quando habilitado, você pode usar as classes `S3Operations` e `s3_ops` para:

- Listar arquivos de álbuns específicos
- Listar todos os álbuns disponíveis
- Listar todos os arquivos do bucket
- Baixar imagens para processamento
- Verificar existência de arquivos

## Implementação Customizada

Para usar S3 em endpoints customizados:

```python
try:
    from s3_operations import s3_ops
    S3_AVAILABLE = True
except ImportError:
    s3_ops = None
    S3_AVAILABLE = False

@app.route('/meu_endpoint_s3')
def meu_endpoint():
    if not S3_AVAILABLE:
        return jsonify({'error': 'S3 não disponível'}), 503
    
    files = s3_ops.list_album_files('meu_album')
    return jsonify({'files': files})
```

## Sistema Principal

O sistema principal funciona independentemente do S3 usando:

- Upload local: Endpoint `/upload` para processar imagens
- Álbum local: Pasta `album/` para armazenar imagens de referência
- Reconhecimento: InsightFace + Faiss + Redis para busca facial

O S3 é uma opção adicional para casos específicos de uso.