import os
import face_recognition
import numpy as np
import math
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import base64
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALBUM_FOLDER'] = 'album'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Garantir que as pastas existam
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ALBUM_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def face_distance_to_conf(face_distance, face_match_threshold=0.4):
    """Converte distância facial em percentual de confiança"""
    if face_distance > face_match_threshold:
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)
        return linear_val
    else:
        range_val = face_match_threshold
        linear_val = 1.0 - (face_distance / (range_val * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def enhance_image_for_recognition(image_path):
    """Ajusta a imagem para ficar melhor para reconhecer faces"""
    try:
        img = Image.open(image_path)

        # Garante que está em RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Ajusta contraste (fotos muito escuras ficam melhores)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Melhora a nitidez (fotos borradas ficam mais nítidas)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)

        # Ajusta brilho um pouco
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)

        # Remove ruído da imagem
        img = img.filter(ImageFilter.SMOOTH_MORE)

        return np.array(img)

    except Exception as e:
        print(f"Erro ao melhorar {image_path}: {e}")
        # Se deu problema, carrega normal
        return face_recognition.load_image_file(image_path)


def load_album_encodings():
    """Processa todas as fotos do álbum e extrai as faces"""
    album_encodings = []
    album_filenames = []
    
    print(f"Carregando fotos do álbum...")
    
    for filename in os.listdir(app.config['ALBUM_FOLDER']):
        if allowed_file(filename):
            image_path = os.path.join(app.config['ALBUM_FOLDER'], filename)
            try:
                print(f"Processando {filename}...")
                # Melhora a qualidade da imagem antes de processar
                image = enhance_image_for_recognition(image_path)

                # Primeiro tenta HOG (mais rápido)
                face_locations = face_recognition.face_locations(image, model='hog')

                # Se não achou nada, tenta CNN
                if not face_locations:
                    try:
                        face_locations = face_recognition.face_locations(image, model='cnn')
                        print(f"  -> CNN usado para {filename}")
                    except:
                        pass

                print(f"  -> {len(face_locations)} faces encontradas em {filename}")

                if face_locations:
                    # Usa num_jitters=10 para boa precisão sem ser muito lento
                    face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=10, model='large')

                    # Salva todas as faces encontradas
                    for idx, encoding in enumerate(face_encodings):
                        album_encodings.append(encoding)
                        album_filenames.append(filename)
                        print(f"    -> Face {idx} salva para {filename}")
                else:
                    print(f"  -> Nenhuma face detectada em {filename}")

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

    print(f"Pronto: {len(album_encodings)} faces de {len(set(album_filenames))} fotos")
    return album_encodings, album_filenames

def draw_face_boxes(image_path, face_locations):
    """Desenha retângulos azuis nas faces encontradas"""
    image = Image.open(image_path)
    
    # Converte para RGB se for RGBA
    if image.mode in ('RGBA', 'LA'):
        # Cria fundo branco e cola a imagem
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    draw = ImageDraw.Draw(image)
    
    # Desenha os retângulos azuis
    for (top, right, bottom, left) in face_locations:
        draw.rectangle([left, top, right, bottom], outline='blue', width=3)
    
    return image

def draw_specific_face_box(image_path, face_location):
    """Desenha retângulo azul só na face que deu match"""
    image = Image.open(image_path)
    
    # Converte para RGB se for RGBA
    if image.mode in ('RGBA', 'LA'):
        # Cria fundo branco e cola a imagem
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    draw = ImageDraw.Draw(image)
    
    # Desenha só na face que deu match
    (top, right, bottom, left) = face_location
    draw.rectangle([left, top, right, bottom], outline='blue', width=3)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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
            # Carrega as faces do álbum
            print("=== COMEÇANDO A PROCURAR ===")
            album_encodings, album_filenames = load_album_encodings()
            
            print(f"Álbum carregado: {len(album_encodings)} faces de {len(set(album_filenames))} fotos")
            
            if not album_encodings:
                print("ERRO: Nenhuma face encontrada no álbum!")
                return jsonify({'error': 'Nenhuma face encontrada no álbum'}), 400
            
            # Melhora a foto enviada
            print(f"Processando foto enviada: {filepath}")
            uploaded_image = enhance_image_for_recognition(filepath)
            
            # Procura faces na foto (HOG primeiro, CNN se necessário)
            face_locations = face_recognition.face_locations(uploaded_image, model='hog')
            
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(uploaded_image, model='cnn')
                    print(f"  -> CNN usado para foto enviada")
                except:
                    pass
            
            print(f"Faces encontradas na foto: {len(face_locations)}")
                
            if not face_locations:
                print("ERRO: Nenhuma face detectada na foto!")
                return jsonify({'error': 'Nenhuma face detectada na foto enviada'}), 400
            
            # Usa num_jitters=10 para boa precisão sem ser muito lento
            face_encodings = face_recognition.face_encodings(uploaded_image, face_locations, num_jitters=10, model='large')
            print(f"Faces codificadas: {len(face_encodings)}")
            
            # Procura correspondências usando o método oficial compare_faces
            matches = []
            seen_combinations = set()
            
            for i, face_encoding in enumerate(face_encodings):
                print(f"\n=== PROCURANDO FACE {i} ===")
                
                # Usa o método oficial compare_faces com tolerance=0.5 (equilibrado)
                results = face_recognition.compare_faces(album_encodings, face_encoding, tolerance=0.5)
                distances = face_recognition.face_distance(album_encodings, face_encoding)
                
                print(f"Distâncias: {[f'{d:.3f}' for d in distances[:5]]}... (5 primeiras)")
                print(f"Menor distância: {min(distances):.3f}")
                
                # Processa apenas os matches válidos
                for j, (is_match, distance) in enumerate(zip(results, distances)):
                    if is_match:
                        # Calcula confiança usando a fórmula oficial da documentação
                        confidence = face_distance_to_conf(distance, face_match_threshold=0.5)
                        
                        print(f"  -> MATCH VÁLIDO: {album_filenames[j]}")
                        print(f"     Distância: {distance:.3f}")
                        print(f"     Confiança: {confidence:.3f} ({confidence*100:.1f}%)")
                        
                        # Evita duplicatas
                        combination = (i, album_filenames[j])
                        if combination not in seen_combinations:
                            seen_combinations.add(combination)
                            matches.append({
                                'face_index': i,
                                'album_filename': album_filenames[j],
                                'distance': float(distance),
                                'confidence': float(confidence)
                            })
            
            # Ordena por confiança (melhor primeiro)
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limita a 10 matches no máximo
            matches = matches[:10]
            
            # Desenha retângulos nas faces da foto original
            image_with_boxes = draw_face_boxes(filepath, face_locations)
            
            # Converte para base64
            buffered = io.BytesIO()
            image_with_boxes.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Processa as fotos do álbum que deram match
            album_images_with_boxes = []
            unique_albums = {}  # Agrupa por foto e pega o melhor match
            
            # Pega só o melhor match de cada foto
            for match in matches:
                filename = match['album_filename']
                if filename not in unique_albums or match['confidence'] > unique_albums[filename]['confidence']:
                    unique_albums[filename] = match
            
            print(f"Total de matches: {len(matches)}")
            print(f"Fotos únicas: {list(unique_albums.keys())}")
            
            if not matches:
                print("NENHUM MATCH ENCONTRADO! Verifique:")
                print("1. Se as fotos estão carregando direito")
                print("2. Se as faces estão sendo detectadas")
                print("3. Se os critérios não estão muito restritivos")
            
            # Processa cada foto que deu match
            for match in unique_albums.values():
                album_path = os.path.join(app.config['ALBUM_FOLDER'], match['album_filename'])
                try:
                    # Melhora a foto do álbum
                    album_image = enhance_image_for_recognition(album_path)
                    
                    # Procura faces na foto do álbum (HOG primeiro)
                    album_face_locations = face_recognition.face_locations(album_image, model='hog')
                    
                    # Usa a face da foto enviada para comparar
                    uploaded_face_encoding = face_encodings[match['face_index']]
                    
                    # Compara com todas as faces da foto do álbum
                    album_face_encodings = face_recognition.face_encodings(album_image, album_face_locations, num_jitters=10, model='large')
                    face_distances = face_recognition.face_distance(album_face_encodings, uploaded_face_encoding)
                    
                    # Pega a face com menor distância (mais parecida)
                    best_face_index = face_distances.argmin()
                    best_face_location = album_face_locations[best_face_index]
                    best_distance = face_distances[best_face_index]
                    
                    print(f"  -> Face escolhida: índice {best_face_index}, distância {best_distance:.3f}")
                    
                    # Só aceita se for parecido o suficiente (threshold equilibrado)
                    if best_distance >= 0.5:
                        print(f"Pulando {match['album_filename']} - muito diferente: {best_distance:.3f}")
                        continue
                    
                    # Desenha retângulo só na face que deu match
                    album_image_with_boxes = draw_specific_face_box(album_path, best_face_location)
                    album_buffered = io.BytesIO()
                    album_image_with_boxes.save(album_buffered, format="JPEG")
                    album_img_str = base64.b64encode(album_buffered.getvalue()).decode()
                    
                    album_images_with_boxes.append({
                        'filename': match['album_filename'],
                        'image_base64': album_img_str,
                        'confidence': match['confidence'],
                        'distance': match['distance']
                    })
                except Exception as e:
                    print(f"Erro ao processar {match['album_filename']}: {e}")
            
            return jsonify({
                'success': True,
                'faces_detected': len(face_locations),
                'matches': matches,
                'image_with_boxes': img_str,
                'album_images_with_boxes': album_images_with_boxes
            })
            
        except Exception as e:
            return jsonify({'error': f'Erro no processamento: {str(e)}'}), 500
        
        finally:
            # Limpar arquivo temporário
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/album')
def list_album():
    """Lista todas as imagens do álbum"""
    album_files = []
    for filename in os.listdir(app.config['ALBUM_FOLDER']):
        if allowed_file(filename):
            album_files.append(filename)
    
    return jsonify({'album_files': album_files})

@app.route('/album/<filename>')
def serve_album_image(filename):
    """Serve imagens do álbum"""
    try:
        return send_from_directory(app.config['ALBUM_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)