import os
from flask import Flask, render_template, request, redirect, url_for
import cv2

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Create a HOGDescriptor object
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def allowed_file(filename):
    # Check if the file extension is allowed
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Save the uploaded file to the upload folder
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        detect_pedestrians(image_path)

        # Generate the URL for the detected image
        detected_image_url = url_for('static', filename='detected/{}.jpg'.format(os.path.splitext(filename)[0]))

        return render_template('result.html', image_url=detected_image_url)
    else:
        return redirect(url_for('index'))


def detect_pedestrians(image_path):
    # Load an image
    image = cv2.imread(image_path)

    # Detect people
    (bounding_boxes, _) = hog.detectMultiScale(image,
                                               winStride=(4, 4),
                                               padding=(8, 8),
                                               scale=1.05)

    # Draw bounding boxes on the image
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image,
                      (x, y),
                      (x + w, y + h),
                      (0, 0, 255),
                      4)

    # Create the output file name
    output_filename = os.path.join('static', 'detected', os.path.basename(image_path))

    # Save the new image
    cv2.imwrite(output_filename, image)


if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Create the folder for detected images if it doesn't exist
    os.makedirs(os.path.join('static', 'detected'), exist_ok=True)

    app.run()
