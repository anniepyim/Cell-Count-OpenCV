#!/usr/bin/env python
from flask import Flask, render_template, request, jsonify, make_response, send_file, session
from flask_socketio import SocketIO, emit
import io, os
import pandas as pd
import numpy as np
from modules.process_image import generate_image, download_image
import uuid
from readlif.reader import LifFile
import cv2
import zipfile
import shutil

import base64
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

ALLOWED_EXTENSIONS = ['', 'txt', 'lif']
UPLOAD_FOLDER = "./user_files/upload"
EXPORT_FOLDER = "./user_files/export"

# EB looks for an 'application' callable by default.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER
app.config['SESSION_PERMANENT'] = False
app.secret_key = "change_your_secret!"
socketio = SocketIO(app)

CONNECTIVITY = 10
CIRCLE_RADIUS = 4
CENTERS_NO = 0

@app.errorhandler(500)
def server_error(e):
    logging.exception('some eror')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

@app.route("/")
def index():
    #only single user is allowed
    session['uid'] = "test_user"
    #session['uid'] = str(uuid.uuid1())

    logger.info(f"session set {session['uid']}")
    return render_template('index.html', invalid_feedback="", stack_list = [], stack_dict_list = [])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():

    uploaded_file = request.files['imagefile']
    if uploaded_file and allowed_file(uploaded_file.filename):
        original_name = uploaded_file.filename
        file_name = session['uid'] + "." + original_name.rsplit('.', 1)[1].lower()
        session['file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        session['export_dir'] = os.path.join(app.config['EXPORT_FOLDER'], session['uid'])

        if os.path.exists(session['file_path']):
            os.remove(session['file_path'])

        uploaded_file.save(session['file_path'])

        try:
            lif_file = LifFile(session['file_path'])
            img_list = lif_file.image_list

            stack_dict_list = []
            stack_list = []

            for img in img_list:
                img_name = ''.join(e for e in img['name'] if (e.isalnum() or e == ' '))
                stack_list.append(img_name)
                c_list = [i for i in range(img['channels'])]
                z_list = [i for i in range(img['dims'].z)]
                stack_dict_list.append({'Z_LIST': z_list, 'C_LIST': c_list})

            session['stack_list'] = stack_list
            session['stack_dict_list'] = stack_dict_list

            # session['stack_list'] = [1,2]
            # session['stack_dict_list'] = [{'Z_LIST':[1,2,3,4], 'C_LIST':[1,2,3,4]}, {'Z_LIST':[1,2], 'C_LIST':[1,2]}]

            return render_template('index.html', stack_list=session['stack_list'],
                                   stack_dict_list=session['stack_dict_list'])

        except Exception as e:
            logger.error(e)
            return render_template('index.html', invalid_feedback="Invalid file - Please check if the image file is valid", stack_list=[], stack_dict_list = [])
    else:
        return render_template('index.html', invalid_feedback="Invalid file - Please check if file exists and is in correct format", stack_list = [], stack_dict_list = [])

# accepts either deafult values or user inputs and outputs prediction
@app.route('/update_image', methods=['POST', 'GET'])
def update_image():

    global CENTERS_NO

    try:
        session['stack'] = int(request.args.get('stack'))
        session['zframe'] = int(request.args.get('zframe'))
        session['channel'] = int(request.args.get('channel'))
        session['bg_thresh'] = int(request.args.get('bg_thresh'))
        session['adaptive_thresh'] = int(request.args.get('adaptive_thresh'))
        session['erosion'] = int(request.args.get('erosion'))
        session['dilation'] = int(request.args.get('dilation'))
        session['min_dist'] = int(request.args.get('min_dist'))
        session['gamma'] = float(request.args.get('gamma'))
        session['gain'] = float(request.args.get('gain'))

        # img = cv2.imread('./test.jpg', cv2.IMREAD_GRAYSCALE)

        lif_file = LifFile(session['file_path'])
        img_list = [i for i in lif_file.get_iter_image()]
        img, CENTERS_NO = generate_image(img_list, session['stack'], session['zframe'], session['channel'],
                                         session['bg_thresh'], session['adaptive_thresh'],
                                         session['erosion'], session['dilation'],
                                         session['min_dist'], session['gamma'], session['gain'],
                                         connectivity=CONNECTIVITY, circle_radius=CIRCLE_RADIUS)


        (flag, encodedImage) = cv2.imencode(".jpg", img)

        response = base64.b64encode(encodedImage)

        return response

    except Exception as e:
        logger.error(e)
        resp = {'message': 'Failed'}
        return make_response(jsonify(resp), 400)

@app.route('/get_centers_no', methods=['POST', 'GET'])
def get_centers_no():

    global CENTERS_NO

    return jsonify({'centers_no':CENTERS_NO})

@app.route('/download', methods=['GET'])
def download_file():

    export_option = request.args.get('export_option')

    if os.path.exists(session['export_dir']):
        shutil.rmtree(session['export_dir'])
    os.makedirs(session['export_dir'])

    lif_file = LifFile(session['file_path'])
    img_list = [i for i in lif_file.get_iter_image()]
    data = download_image(export_option, session['export_dir'], img_list,
                          session['stack_list'], session['stack_dict_list'],
                          session['stack'], session['zframe'], session['channel'],
                          session['bg_thresh'], session['adaptive_thresh'],
                          session['erosion'], session['dilation'],
                          session['min_dist'], session['gamma'], session['gain'], CONNECTIVITY, CIRCLE_RADIUS)


    data_df = pd.DataFrame(data)
    export_file_path = os.path.join(session['export_dir'], "data.csv")
    data_df.to_csv(export_file_path, index=False, sep=";")

    def retrieve_file_paths(dirName):

        # setup file paths variable
        filePaths = []

        # Read all directory, subdirectories and file lists
        for root, directories, files in os.walk(dirName):
            for filename in files:
                # Create the full filepath by using os module.
                filePath = os.path.join(root, filename)
                filePaths.append(filePath)

        # return all paths
        return filePaths

    # Call the function to retrieve all files and folders of the assigned directory
    filePaths = retrieve_file_paths(session['export_dir'])

    # printing the list of all files to be zipped
    logger.info('The following list of files will be zipped:')
    for fileName in filePaths:
        logger.info(fileName)

    zip_file_path = session['export_dir'] + '.zip'
    zip_file = zipfile.ZipFile(zip_file_path, 'w')
    with zip_file:
        # writing each file one by one
        for file in filePaths:
            zip_file.write(file)

    return_data = io.BytesIO()
    with open(zip_file_path, 'rb') as fo:
        return_data.write(fo.read())
    # (after writing, cursor will be at last byte, so move it to start)
    return_data.seek(0)

    os.remove(zip_file_path)
    shutil.rmtree(session['export_dir'])

    return send_file(return_data, mimetype='application/zip',
                     attachment_filename='download.zip')

@socketio.on('disconnect')
def disconnect_user():
    logger.info("disconnectee")
    session.pop("key")
    if os.path.exists(session['export_dir']):
        shutil.rmtree(session['export_dir'])
    if os.path.exists(session['file_path']):
        os.remove(session['file_path'])

# # # when running app locally
# # if __name__ == '__main__':
# #     application.debug = False
#     application.run(host='0.0.0.0')