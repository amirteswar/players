from flask import Flask, request, render_template, send_from_directory
import os
from same import find_similar_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            similar_image_path, similarity = find_similar_image(file_path)
            similar_image_path = os.path.relpath(similar_image_path)  # Relative path for HTML
            return render_template('result.html', similar_image=similar_image_path, similarity=similarity)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, render_template, send_from_directory
# import os
# from find_similarity import find_similar_image

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure the uploads directory exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return 'No file part'
#         file = request.files['file']
#         if file.filename == '':
#             return 'No selected file'
#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)
#             similar_image_path, similarity = find_similar_image(file_path)
#             similar_image_path = os.path.relpath(similar_image_path)  # Relative path for HTML
#             return render_template('result.html', similar_image=similar_image_path, similarity=similarity)
#     return render_template('upload.html')

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/<path:filename>')
# def serve_image(filename):
#     return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

# if __name__ == '__main__':
#     app.run(debug=True)
