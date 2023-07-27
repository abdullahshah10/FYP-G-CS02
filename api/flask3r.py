import os
import subprocess
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.datastructures import  FileStorage
app = Flask(__name__)
UPLOAD_FOLDER = r'/home/ubuntu/FYP-G-CS02/api'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#@app.route('/upload')
#def upload_file():
   #return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    print('req')
    #return request
    print(type(request))
    if request.method == 'POST':
        try:
            f = request.files['file']
            #f.save(secure_filename(f.filename))
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test.wav'))
            result = subprocess.check_output(['python', 'final.py'])
            output = result.decode('utf-8')
            return {
                'status': 200,
                'response': output.rstrip()
            }   
        except BadRequestKeyError:
            print(request.files)
            return 'File key is missing in the request!', 400
if __name__ == '__main__':
   app.run(debug = True)
