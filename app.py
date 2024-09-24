from flask import Flask, render_template, request, jsonify
import os
import cv2
from region_growing import region_growing
from seed_selection import select_seed_point
from visualization import display_results

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 保存上传的图像
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return jsonify({'image_path': file_path})


@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    image_path = data['image_path']
    seed_point = (data['x'], data['y'])  # 鼠标点击的位置

    # 读取图像并进行区域生长处理
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    segmented_image = region_growing(gray_image, seed_point, threshold=10)

    # 保存处理后的图像
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.png')
    cv2.imwrite(result_path, segmented_image)

    return jsonify({'result_path': result_path})


if __name__ == "__main__":

    app.run(port=5001,debug=True)

