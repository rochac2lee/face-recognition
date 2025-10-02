# 🎯 Sistema de Reconhecimento Facial Profissional

Sistema avançado de reconhecimento facial desenvolvido com **InsightFace**, **Faiss (Facebook)** e **Redis** para alta precisão e performance. Ideal para fotógrafos, estúdios e aplicações que precisam identificar pessoas em grandes coleções de fotos.

## ✨ Características Principais

- 🔍 **Alta Precisão**: Utiliza InsightFace com modelo Buffalo-L para detecção facial avançada
- ⚡ **Performance Otimizada**: Faiss (Facebook) para busca vetorial ultra-rápida
- 🚀 **Escalabilidade**: Redis para cache e gerenciamento de dados
- 🎨 **Interface Moderna**: Interface web responsiva e intuitiva
- 📱 **Drag & Drop**: Upload fácil de imagens
- 🔄 **Tempo Real**: Processamento instantâneo com feedback visual
- ☁️ **S3 Opcional**: Integração opcional com AWS S3 como diferencial

## 🏗️ Arquitetura do Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Interface     │    │   Flask App     │    │   Redis Cache   │
│   Web (HTML/JS) │◄──►│   (Python)      │◄──►│   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   InsightFace   │
                       │   + Faiss       │
                       │   (AI Engine)   │
                       └─────────────────┘
```

## 🚀 Instalação e Configuração

### Pré-requisitos

- Docker e Docker Compose
- Python 3.9+ (para desenvolvimento local)
- 4GB+ RAM recomendado

### 1. Clone o Repositório

```bash
git clone git@github.com:rochac2lee/face-recognition.git
cd face-recognition
```

### 2. Configuração com Docker (Recomendado)

```bash
# Iniciar o sistema
docker-compose up -d

# Verificar status
docker-compose ps

# Ver logs
docker-compose logs -f face-recognition
```

O sistema estará disponível em: **http://localhost:8080**

### 3. Configuração Local (Desenvolvimento)

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar Redis (necessário)
docker run -d -p 6379:6379 redis:7-alpine

# Executar aplicação
python app.py
```

## 📁 Estrutura do Projeto

```
face-recognition/
├── app.py                 # Aplicação principal Flask
├── face_recognition_redis.py  # Motor de reconhecimento facial
├── s3_operations.py       # Operações S3 (opcional)
├── s3_config.py          # Configuração S3 (opcional)
├── requirements.txt      # Dependências Python
├── docker-compose.yml    # Configuração Docker
├── Dockerfile           # Imagem Docker
├── album/               # Pasta com fotos de referência
├── uploads/             # Pasta temporária para uploads
├── templates/           # Templates HTML
├── static/              # Arquivos estáticos (CSS/JS)
└── S3_SETUP.md         # Guia de configuração S3
```

## 🎮 Como Usar o Sistema

### 1. Preparar o Banco de Dados

Coloque suas fotos de referência na pasta `album/`:

```bash
# Exemplo de estrutura
album/
├── pessoa1.jpg
├── pessoa2.jpg
├── evento1/
│   ├── foto1.jpg
│   └── foto2.jpg
└── evento2/
    └── foto3.jpg
```

### 2. Acessar a Interface

1. Abra o navegador em `http://localhost:8080`
2. A interface mostrará uma área de upload drag & drop

### 3. Processar Imagens

1. **Arraste e solte** uma imagem na área de upload
2. Clique em **"Buscar Fotos"**
3. O sistema irá:
   - Detectar faces na imagem
   - Buscar correspondências no banco de dados
   - Mostrar resultados com precisão

### 4. Interpretar Resultados

- **Faces detectadas**: Número de rostos encontrados na imagem
- **Correspondências**: Fotos similares encontradas
- **Precisão**: Percentual de confiança (0-100%)
- **Bounding boxes**: Caixas verdes (faces detectadas) e vermelhas (matches)

## 🔧 Endpoints da API

### Principais

- `GET /` - Interface web principal
- `POST /upload` - Upload e reconhecimento de imagem
- `GET /health` - Status do sistema
- `GET /album` - Listar imagens do álbum
- `GET /album/<filename>` - Servir imagem do álbum

### Debug e Análise

- `POST /debug_similarity` - Debug de similaridade (mostra todos os scores)
- `POST /search_client` - Busca específica por cliente
- `GET /client_stats/<client_id>` - Estatísticas de cliente

## ⚙️ Configurações Avançadas

### Parâmetros do Motor de Reconhecimento

No arquivo `app.py`, você pode ajustar:

```python
face_engine = FaceRecognitionRedis(
    model_name='buffalo_l',           # Modelo InsightFace
    similarity_threshold=0.25,        # Threshold de similaridade
    min_face_size=5,                  # Tamanho mínimo da face
    min_det_score=0.2,                # Confiança mínima de detecção
    # ... outros parâmetros
)
```

### Variáveis de Ambiente

```bash
# Configurações Flask
FLASK_ENV=production
FLASK_APP=app.py

# Configurações Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Configurações S3 (opcional)
AWS_ACCESS_KEY_ID=sua_key
AWS_SECRET_ACCESS_KEY=sua_secret
AWS_S3_BUCKET_NAME=seu_bucket
AWS_S3_REGION=us-east-1
```

## 🔍 Como Funciona o Reconhecimento

### 1. Detecção Facial
- InsightFace detecta e extrai faces das imagens
- Gera embeddings (vetores) únicos para cada face
- Aplica filtros de qualidade e tamanho

### 2. Indexação
- Faiss cria índice vetorial para busca rápida
- Armazena embeddings no Redis
- Organiza por pessoa/evento

### 3. Busca
- Extrai face da imagem de consulta
- Busca no índice vetorial
- Retorna resultados ordenados por similaridade

### 4. Filtragem
- Aplica threshold de similaridade
- Remove duplicatas
- Calcula confiança final

## 📊 Monitoramento e Logs

### Health Check

```bash
curl http://localhost:8080/health
```

Resposta:
```json
{
  "status": "healthy",
  "engine_available": true,
  "s3_available": false,
  "database_stats": {
    "total_faces": 150,
    "total_persons": 25,
    "index_size": "2.3MB"
  }
}
```

### Logs do Sistema

```bash
# Ver logs em tempo real
docker-compose logs -f face-recognition

# Logs específicos
docker-compose logs face-recognition | grep "✅\|❌\|🔍"
```

## 🚨 Solução de Problemas

### Problemas Comuns

1. **"Motor de reconhecimento não disponível"**
   - Verifique se o Redis está rodando
   - Reinicie o container: `docker-compose restart face-recognition`

2. **"Nenhuma face detectada"**
   - Verifique qualidade da imagem
   - Ajuste `min_face_size` e `min_det_score`
   - Use imagens com boa iluminação

3. **"Performance lenta"**
   - Aumente recursos do Docker
   - Verifique uso de memória do Redis
   - Considere usar GPU para InsightFace

### Debug Avançado

```bash
# Testar similaridade
curl -X POST -F "file=@teste.jpg" http://localhost:8080/debug_similarity

# Verificar estatísticas
curl http://localhost:8080/health
```

## 🔒 Segurança

- Uploads são processados em memória temporária
- Arquivos temporários são removidos automaticamente
- Não há armazenamento permanente de imagens enviadas
- Redis pode ser configurado com autenticação

## 📈 Performance

### Benchmarks Típicos

- **Detecção**: ~200ms por imagem
- **Busca**: ~50ms para 1000 faces
- **Memória**: ~2GB para 10.000 faces
- **Precisão**: >95% com threshold 0.25

### Otimizações

- Use imagens redimensionadas (max 1920px)
- Configure Redis com persistência adequada
- Monitore uso de memória
- Considere clustering para alta demanda

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 🆘 Suporte

- 📧 Email: [seu-email]
- 🐛 Issues: [GitHub Issues]
- 📖 Documentação: [Wiki do projeto]

---

**Desenvolvido com ❤️ para a comunidade de reconhecimento facial**