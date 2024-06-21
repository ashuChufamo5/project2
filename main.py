# from flask import Flask, render_template, request, send_file
# import os
# import cv2
# import numpy as np
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Set the upload folder
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return 'No file part', 400

#     file = request.files['file']

#     if file.filename == '':
#         return 'No selected file', 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Load the image
#         img = cv2.imread(file_path)

#         # Convert the image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding to get the document area
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Find the contours in the thresholded image
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Find the largest contour, which should be the document
#         largest_contour = max(contours, key=cv2.contourArea)

#         # Get the bounding box of the largest contour
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # Crop the image to the document area
#         cropped = img[y:y+h, x:x+w]

#         # Save the cropped image
#         cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], f'cropped_{filename}')
#         cv2.imwrite(cropped_path, cropped)

#         return render_template('index.html', image_path=url_for('display_image', filename=f'cropped_{filename}'))
#     else:
#         return 'Invalid file type', 400

# @app.route('/image/<filename>')
# def display_image(filename):
#     return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True) 


# from app import app
# from flask import Flask, flash, request, redirect, render_template
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# import io
# from PIL import Image
# import base64
# from Helpers import *  # Assuming 'Helpers.py' contains your resize and transform functions

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def upload_form():
#     return render_template('upload.html')

# @app.route('/', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         flash('Document scan was successful')
#         filename = secure_filename(file.filename)

#         filestr = request.files['file'].read()
#         npimg = np.frombuffer(filestr, np.uint8)
#         image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
#         ratio = image.shape[0] / 500.0
#         orig = image.copy()
#         image = Helpers.resize(image, height=500)

#         # --- Convex Hull Algorithm ---
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)

#         contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         largest_contour = None
#         largest_area = 0
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > largest_area:
#                 largest_area = area
#                 largest_contour = contour

#         if largest_contour is not None:
#             hull = cv2.convexHull(largest_contour)

#             ext_left = tuple(hull[hull[:, :, 0].argmin()][0])
#             ext_right = tuple(hull[hull[:, :, 0].argmax()][0])
#             ext_top = tuple(hull[hull[:, :, 1].argmin()][0])
#             ext_bottom = tuple(hull[hull[:, :, 1].argmax()][0])

#             pts1 = np.float32([ext_left, ext_right, ext_bottom, ext_top])
#             pts2 = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
#             matrix = cv2.getPerspectiveTransform(pts1, pts2)

#             warped = cv2.warpPerspective(orig, matrix, (image.shape[1], image.shape[0]))

#             # --- End of Convex Hull Algorithm ---

#             img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
#             file_object = io.BytesIO()
#             img = Image.fromarray(Helpers.resize(img, width=500))
#             img.save(file_object, 'PNG')
#             base64img = "data:image/png;base64," + base64.b64encode(file_object.getvalue()).decode('ascii')

#             return render_template('upload.html', image=base64img)
#         else:
#             flash('No document found in image')
#             return redirect(request.url)
#     else:
#         flash('Allowed image types are -> png, jpg, jpeg')
#         return redirect(request.url)

# if __name__ == "__main__":
#     app.run(debug=True)


# import cv2
# import pytesseract

# # Load your image
# img = cv2.imread('document_top.jpg')

# # Convert the image to gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Use Tesseract to do OCR on the image
# text = pytesseract.image_to_string(gray)
# print(text)

from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image
import base64
from Helpers import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		flash('Document scan was successful')
		filename = secure_filename(file.filename)
		
		filestr = request.files['file'].read()
		npimg = np.frombuffer(filestr, np.uint8)
		image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
		ratio = image.shape[0] / 500.0
		orig = image.copy()
		image = Helpers.resize(image, height = 500)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		edged = cv2.Canny(gray, 75, 200)

		cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = Helpers.grab_contours(cnts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:
				screenCnt = approx
				break

		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

		warped = Helpers.transform(orig, screenCnt.reshape(4, 2) * ratio)

		img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
		file_object = io.BytesIO()
		img= Image.fromarray(Helpers.resize(img,width=500))
		img.save(file_object, 'PNG')
		base64img = "data:image/png;base64,"+base64.b64encode(file_object.getvalue()).decode('ascii')

		return render_template('upload.html', image=base64img )
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)