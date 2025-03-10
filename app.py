import os
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import utlis
import io

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    img_array = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    imgContours, contours = utlis.getContours(img, minArea=50000, filter=4)
    
    if len(contours) != 0:
        biggest = contours[0][2]
        scale = 3
        wP = 210 * scale
        hP = 297 * scale
        imgWarp = utlis.warpImg(img, biggest, wP, hP)
        imgContours2, contours2 = utlis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)

        result = []
        for obj in contours2:
            nPoints = utlis.reorder(obj[2])
            nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
            nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
            x, y, w, h = obj[3]
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.putText(imgContours2, f'{nW}cm', (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.putText(imgContours2, f'{nH}cm', (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            result.append({
                'width': nW,
                'height': nH,
                'contour': obj[2].tolist(),
                'bounding_box': obj[3]
            })

        _, img_encoded = cv2.imencode('.jpg', imgContours2)
        img_bytes = img_encoded.tobytes()
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg', as_attachment=True, download_name='processed_image.jpg')

    return jsonify({'error': 'No contours detected'}), 400

if __name__ == '__main__':
    # Vercel provides the PORT environment variable
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if the environment variable is not set
    app.run(host='0.0.0.0', port=port, debug=True)  # Listen on all interfaces
