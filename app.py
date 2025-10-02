#!/usr/bin/env python3
"""
Sistema de Reconhecimento Facial Profissional
InsightFace + Faiss (Facebook) + Redis
"""

import os
import io
import base64
import logging
import threading
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from face_recognition_redis import FaceRecognitionRedis
from s3_operations import s3_ops

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALBUM_FOLDER'] = 'album'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Extensões permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Inicializar motor de reconhecimento facial
face_engine = None

def initialize_face_engine():
    """Inicializar o motor de reconhecimento facial"""
    global face_engine
    try:
        logger.info("🚀 Inicializando motor de reconhecimento facial...")
        face_engine = FaceRecognitionRedis(
            model_name='buffalo_l',
            similarity_threshold=0.25,  # Threshold mais flexível para encontrar mais matches
            min_face_size=5,  # Tamanho mínimo balanceado
            redis_host='redis',
            redis_port=6379,
            redis_db=0,
            # Parâmetros otimizados para fotógrafos
            adaptive_threshold=False,  # Desabilitar threshold adaptativo para encontrar todas as fotos
            majority_vote_k=1,  # Apenas 1 foto necessária para considerar match
            min_det_score=0.2,  # Confiança mínima mais permissiva
            ensemble_models=None  # Pode ser expandido no futuro
        )
        logger.info("✅ Motor de reconhecimento facial inicializado com sucesso!")
        
        # Construir banco de dados automaticamente em background apenas se não existir
        # DESATIVADO TEMPORARIAMENTE - processamento automático do álbum
        def build_database_background():
            try:
                # Verificar se o banco já existe
                stats = face_engine.get_database_stats()
                if stats['total_faces'] > 0:
                    logger.info(f"📥 Banco de dados já existe com {stats['total_faces']} faces - pulando construção automática")
                    return
                
                logger.info("🏗️ Processamento automático do álbum DESATIVADO temporariamente")
                logger.info("💡 Use o endpoint /process_s3_image para processar imagens individualmente")
                # stats = face_engine.build_database_from_folder(app.config['ALBUM_FOLDER'])
                # logger.info(f"✅ Banco construído automaticamente: {stats}")
            except Exception as e:
                logger.error(f"❌ Erro na construção automática do banco: {e}")
        
        # Executar construção em thread separada
        db_thread = threading.Thread(target=build_database_background, daemon=True)
        db_thread.start()
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar motor de reconhecimento: {e}")
        face_engine = None

# Inicializar motor
initialize_face_engine()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/album')
def album():
    """Listar imagens do álbum"""
    try:
        album_files = []
        if os.path.exists(app.config['ALBUM_FOLDER']):
            for filename in os.listdir(app.config['ALBUM_FOLDER']):
                if allowed_file(filename):
                    album_files.append(filename)
        return jsonify({'files': sorted(album_files)})
    except Exception as e:
        logger.error(f"❌ Erro ao listar álbum: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/album/<filename>')
def get_album_image(filename):
    """Servir imagem do álbum"""
    try:
        return send_from_directory(app.config['ALBUM_FOLDER'], filename)
    except Exception as e:
        logger.error(f"❌ Erro ao servir imagem: {e}")
        return jsonify({'error': str(e)}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint principal para reconhecimento facial usando InsightFace + Faiss
    Sistema profissional com alta precisão e performance
    """
    if face_engine is None:
        return jsonify({'error': 'Motor de reconhecimento não disponível'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            logger.info(f"🔍 Processando imagem com InsightFace: {filename}")
            
            # Buscar faces similares usando InsightFace + Faiss
            results = face_engine.search_faces(filepath, top_k=50)
            
            if not results:
                logger.info("❌ Nenhuma face similar encontrada")
                return jsonify({
                    'success': True,
                    'message': 'Nenhuma face similar encontrada no banco de dados',
                    'matches': [],
                    'image_with_boxes': None,
                    'album_images_with_boxes': []
                })
            
            # Processar imagem original para mostrar faces detectadas
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extrair faces da imagem de consulta para desenhar bounding boxes
            face_data = face_engine.extract_face_embeddings(filepath)
            
            # Desenhar bounding boxes nas faces detectadas (verde)
            for face_info in face_data:
                bbox = face_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Converter para base64
            pil_image = Image.fromarray(image_rgb)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Processar imagens dos matches
            album_images_with_boxes = []
            seen_persons = set()
            
            for result in results:
                person_id = result['person_id']
                if person_id in seen_persons:
                    continue
                seen_persons.add(person_id)
                
                try:
                    # Carregar imagem do match
                    match_image_path = result['image_path']
                    match_image = cv2.imread(match_image_path)
                    
                    if match_image is not None:
                        match_image_rgb = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
                        
                        # Desenhar bounding box na face que deu match (vermelho)
                        match_bbox = result['match_bbox']
                        x1, y1, x2, y2 = map(int, match_bbox)
                        cv2.rectangle(match_image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        
                        # Adicionar texto com similaridade
                        similarity_text = f"Sim: {result['similarity']:.3f}"
                        cv2.putText(match_image_rgb, similarity_text, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Converter para base64
                        match_pil_image = Image.fromarray(match_image_rgb)
                        match_buffered = io.BytesIO()
                        match_pil_image.save(match_buffered, format="JPEG")
                        match_img_str = base64.b64encode(match_buffered.getvalue()).decode()
                    
                    album_images_with_boxes.append({
                            'filename': os.path.basename(match_image_path),
                            'person_id': person_id,
                            'image_base64': match_img_str,
                            'confidence': result['confidence'],
                            'similarity': result['similarity']
                        })
                        
                except Exception as e:
                    logger.error(f"Erro ao processar imagem do match {person_id}: {e}")
                    continue
            
            # Preparar resposta
            response_data = {
                'success': True,
                'faces_detected': len(face_data),
                'matches_found': len(results),
                'matches': [
                    {
                        'person_id': r['person_id'],
                        'face_id': r.get('face_id', f"{r['person_id']}_face_{r['match_face_index']}"),
                        'filename': os.path.basename(r['image_path']),
                        'similarity': r['similarity'],
                               'confidence': r['confidence'],
                               'accuracy': r.get('accuracy', r['confidence'])  # Usar accuracy se disponível, senão confidence
                    } for r in results
                ],
                'image_with_boxes': img_str,
                'album_images_with_boxes': album_images_with_boxes
            }
            
            logger.info(f"✅ Processamento concluído: {len(results)} matches encontrados")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento InsightFace: {e}")
            return jsonify({'error': f'Erro no processamento: {str(e)}'}), 500
        
        finally:
            # Limpar arquivo temporário
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/health')
def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'engine_available': face_engine is not None,
        'database_stats': face_engine.get_database_stats() if face_engine else None
    })

@app.route('/list_s3_images')
def list_s3_images():
    """Listar imagens disponíveis no S3 (padrão: 1/album)"""
    try:
        album_prefix = request.args.get('album_prefix', '1/album')
        files = s3_ops.list_album_files(album_prefix)
        return jsonify({
            'success': True,
            'album_prefix': album_prefix,
            'total_files': len(files),
            'files': files
        })
    except Exception as e:
        logger.error(f"❌ Erro ao listar imagens S3: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/list_s3_albums')
def list_s3_albums():
    """Listar todos os álbuns disponíveis no S3"""
    try:
        albums = s3_ops.list_all_albums()
        return jsonify({
            'success': True,
            'total_albums': len(albums),
            'albums': albums
        })
    except Exception as e:
        logger.error(f"❌ Erro ao listar álbuns S3: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/list_all_s3_files')
def list_all_s3_files():
    """Listar todos os arquivos do bucket S3"""
    try:
        files = s3_ops.list_all_files()
        return jsonify({
            'success': True,
            'total_files': len(files),
            'files': files
        })
    except Exception as e:
        logger.error(f"❌ Erro ao listar arquivos S3: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/process_s3_image', methods=['POST'])
def process_s3_image():
    """Processar uma imagem do S3 e adicionar faces ao banco de dados"""
    if face_engine is None:
        return jsonify({'error': 'Motor de reconhecimento não disponível'}), 500
    
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'Nome do arquivo é obrigatório'}), 400
        
        filename = data['filename']
        album_prefix = data.get('album_prefix')
        logger.info(f"🔄 Processando imagem do S3: {filename} do álbum {album_prefix}")
        
        # Verificar se a imagem existe no S3
        if not s3_ops.image_exists(filename, album_prefix):
            return jsonify({'error': f'Imagem {filename} não encontrada no álbum {album_prefix}'}), 404
        
        # Baixar imagem para arquivo temporário
        temp_path = s3_ops.download_image_to_temp(filename, album_prefix)
        if not temp_path:
            return jsonify({'error': f'Erro ao baixar imagem {filename} do álbum {album_prefix}'}), 500
        
        try:
            # Extrair embeddings das faces
            face_data = face_engine.extract_face_embeddings(temp_path)
            
            if not face_data:
                logger.warning(f"⚠️ Nenhuma face detectada em: {filename}")
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'faces_detected': 0,
                    'faces_added': 0,
                    'message': 'Nenhuma face detectada na imagem'
                })
            
            # Usar nome do arquivo (sem extensão) como person_id
            person_id = os.path.splitext(filename)[0]
            
            # Criar caminho S3 real para salvar no banco
            s3_image_path = f"{album_prefix}/{filename}"
            
            # Adicionar faces ao banco de dados (usando caminho S3 real)
            faces_added = face_engine.add_faces_to_database_with_s3_path(
                temp_image_path=temp_path,
                s3_image_path=s3_image_path,
                person_id=person_id
            )
            
            logger.info(f"✅ Processado {filename}: {faces_added} faces adicionadas")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'person_id': person_id,
                'faces_detected': len(face_data),
                'faces_added': faces_added,
                'message': f'Sucesso: {faces_added} faces adicionadas ao banco'
            })
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"❌ Erro ao processar imagem S3: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/debug_similarity', methods=['POST'])
def debug_similarity():
    """Endpoint para debug de similaridade - mostra todos os scores"""
    if face_engine is None:
        return jsonify({'error': 'Motor de reconhecimento não disponível'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            logger.info(f"🔍 Debug Similarity: Analisando {filename}")
            
            # Buscar faces similares com top_k maior
            results = face_engine.search_faces(filepath, top_k=50)
            
            # Extrair faces da imagem de consulta
            face_data = face_engine.extract_face_embeddings(filepath)
            
            debug_info = {
                'filename': filename,
                'faces_detected': len(face_data),
                'total_results': len(results),
                'threshold': face_engine.similarity_threshold,
                'all_scores': []
            }
            
            for i, result in enumerate(results):
                debug_info['all_scores'].append({
                    'rank': i + 1,
                    'person_id': result['person_id'],
                    'similarity': float(result['similarity']),
                    'confidence': float(result['confidence']),
                    'above_threshold': result['similarity'] >= face_engine.similarity_threshold,
                    'filename': os.path.basename(result['image_path'])
                })
            
            return jsonify(debug_info)
            
        except Exception as e:
            logger.error(f"❌ Erro no debug similarity: {e}")
            return jsonify({'error': f'Erro no debug: {str(e)}'}), 500
        
        finally:
            # Limpar arquivo temporário
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/search_client', methods=['POST'])
def search_client_faces():
    """
    Endpoint específico para fotógrafos - busca faces de um cliente específico
    com threshold otimizado e votação majoritária
    """
    if face_engine is None:
        return jsonify({'error': 'Motor de reconhecimento não disponível'}), 500
    
    if 'file' not in request.files or 'client_id' not in request.form:
        return jsonify({'error': 'Arquivo e client_id são obrigatórios'}), 400
    
    file = request.files['file']
    client_id = request.form['client_id']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            logger.info(f"🔍 Busca específica do cliente {client_id}: {filename}")
            
            # Buscar faces do cliente específico
            results = face_engine.search_client_faces(filepath, client_id, top_k=20)
            
            if not results:
                logger.info(f"❌ Nenhuma face do cliente {client_id} encontrada")
                return jsonify({
                    'success': True,
                    'message': f'Nenhuma foto do cliente {client_id} encontrada',
                    'client_id': client_id,
                    'matches': [],
                    'image_with_boxes': None,
                    'album_images_with_boxes': []
                })
            
            # Processar imagem original para mostrar faces detectadas
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extrair faces da imagem de consulta para desenhar bounding boxes
            face_data = face_engine.extract_face_embeddings(filepath)
            
            # Desenhar bounding boxes nas faces detectadas (verde)
            for face_info in face_data:
                bbox = face_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Converter para base64
            pil_image = Image.fromarray(image_rgb)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Processar imagens dos matches
            album_images_with_boxes = []
            seen_persons = set()
            
            for result in results:
                person_id = result['person_id']
                if person_id in seen_persons:
                    continue
                seen_persons.add(person_id)
                
                try:
                    # Carregar imagem do match
                    match_image_path = result['image_path']
                    match_image = cv2.imread(match_image_path)
                    
                    if match_image is not None:
                        match_image_rgb = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
                        
                        # Desenhar bounding box na face que deu match (vermelho)
                        match_bbox = result['match_bbox']
                        x1, y1, x2, y2 = map(int, match_bbox)
                        cv2.rectangle(match_image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        
                        # Adicionar texto com similaridade
                        similarity_text = f"Sim: {result['similarity']:.3f}"
                        cv2.putText(match_image_rgb, similarity_text, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Converter para base64
                        match_pil_image = Image.fromarray(match_image_rgb)
                        match_buffered = io.BytesIO()
                        match_pil_image.save(match_buffered, format="JPEG")
                        match_img_str = base64.b64encode(match_buffered.getvalue()).decode()
                        
                        album_images_with_boxes.append({
                            'filename': os.path.basename(match_image_path),
                            'person_id': person_id,
                            'image_base64': match_img_str,
                            'confidence': result['confidence'],
                            'similarity': result['similarity']
                        })
                        
                except Exception as e:
                    logger.error(f"Erro ao processar imagem do match {person_id}: {e}")
                    continue
            
            # Preparar resposta
            response_data = {
                'success': True,
                'client_id': client_id,
                'faces_detected': len(face_data),
                'matches_found': len(results),
                'matches': [
                    {
                        'person_id': r['person_id'],
                        'face_id': r.get('face_id', f"{r['person_id']}_face_{r['match_face_index']}"),
                        'filename': os.path.basename(r['image_path']),
                        'similarity': r['similarity'],
                               'confidence': r['confidence'],
                               'accuracy': r.get('accuracy', r['confidence'])  # Usar accuracy se disponível, senão confidence
                    } for r in results
                ],
                'image_with_boxes': img_str,
                'album_images_with_boxes': album_images_with_boxes
            }
            
            logger.info(f"✅ Busca do cliente {client_id} concluída: {len(results)} matches encontrados")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Erro na busca do cliente {client_id}: {e}")
            return jsonify({'error': f'Erro no processamento: {str(e)}'}), 500
        
        finally:
            # Limpar arquivo temporário
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/client_stats/<client_id>')
def get_client_stats(client_id):
    """Endpoint para obter estatísticas de um cliente específico"""
    if face_engine is None:
        return jsonify({'error': 'Motor de reconhecimento não disponível'}), 500
    
    try:
        stats = face_engine.get_client_stats(client_id)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas do cliente {client_id}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Iniciando servidor de reconhecimento facial...")
    logger.info("📊 Sistema: InsightFace + Faiss (Facebook)")
    logger.info("🌐 Acesse: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)