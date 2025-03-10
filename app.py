from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import utlis
import io

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Ensure the file is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    
    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the uploaded image
        img_array = np.frombuffer(file.read(), np.uint8)  # Replacing np.fromstring with np.frombuffer
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode the image to OpenCV format

        # Process the image using the functions from Utils.py
        imgContours, contours = utlis.getContours(img, minArea=50000, filter=4)

        if len(contours) == 0:
            return jsonify({'error': 'No contours detected'}), 400  # No contours found
        
        biggest = contours[0][2]
        
        # Scaling factor for dimensions (as per ObjectMeasurement.py)
        scale = 3
        wP = 210 * scale  # Width of the page in pixels
        hP = 297 * scale  # Height of the page in pixels
        
        # Warp the image based on detected contours
        imgWarp = utlis.warpImg(img, biggest, wP, hP)

        # Further processing on the warped image to get contours
        imgContours2, contours2 = utlis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)
        
        result = []
        for obj in contours2:
            # Reorder points and calculate width and height of each object
            nPoints = utlis.reorder(obj[2])
            nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
            nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)

            # Visualize contours and measurements on the image
            x, y, w, h = obj[3]
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.putText(imgContours2, f'{nW}cm', (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.putText(imgContours2, f'{nH}cm', (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)

            # Append the result (width, height) for each object found
            result.append({
                'width': nW,
                'height': nH,
                'contour': obj[2].tolist(),
                'bounding_box': obj[3]
            })

        # Convert the processed image to bytes and return as response
        _, img_encoded = cv2.imencode('.jpg', imgContours2)
        img_bytes = img_encoded.tobytes()

        # Create a BytesIO stream to send the image back in the response
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg', as_attachment=True, download_name='processed_image.jpg')

    except Exception as e:
        # If there's any error during processing, return it as a response
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
