import pickle
import keras
import numpy as np
import pandas as pd
import os

from sklearn import metrics

from keras.models import Model, load_model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from flask import Flask, request, redirect, url_for, Response
from flask import render_template, make_response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

#from jinja2 import Template

UPLOAD_FOLDER = './static/data/uploads'
ALLOWED_EXTENSIONS = set(['json'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_model(fn='./static/models/cnn1d_model.h5'):
    global model
    model = keras.models.load_model(fn)
    ## this call has to be made or keras does not
    ## work in flasks debug model
    model._make_predict_function()
    return(model)


def load_data(path):
    data = pd.read_json(path)
    return(data)


def load_tokenizer(fn='./static/models/tokenizer.pickle'):
    global tokenizer
    with open(fn, 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokenizer.oov_token = None ## this is a hack to handle bad serialization
    return(tokenizer)


def prepare_docs(data, maxlen=500):
    texts = data["Abstract"]
    labels = data["Label"]
    sequences = tokenizer.texts_to_sequences(texts)
    texts = pad_sequences(sequences, maxlen=maxlen)
    labels = to_categorical(np.asarray(labels))
    return([texts, labels])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def uploaded_file(filename):
    return(send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename))

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#            return redirect(url_for('uploaded_file',
#                                    filename=filename))
            return redirect(url_for('predict', filename=filename))
    return(render_template('index.html'))


@app.route("/predict/<filename>", methods=["GET"])
def predict(filename):
    data = load_data(os.path.join(app.config['UPLOAD_FOLDER'],
                             filename))
    x_test, y_test = prepare_docs(data)

    prediction = np.round(model.predict(x_test), 4)
    prediction = pd.DataFrame.from_dict({'noncancer_prob': list(prediction[:, 0]),
                                         'cancer_prob': list(prediction[:, 1]),
                                         'label': list(np.argmax(y_test, axis=1))})
    prediction = prediction[["noncancer_prob", "cancer_prob", "label"]]
    ## add cancer probability column
    ## be aware this resets the row index !
    data_concat = pd.concat([data.reset_index(drop=True),
                             prediction["cancer_prob"]], axis=1)
    data_concat = data_concat.sort_values(by="cancer_prob", ascending=False)
    data_out = data_concat[["cancer_prob", "PUI", "Title", "Source"]]
#    data_out = data_out.to_json(orient='records')
#    template = render_template("bootstrap_table.html",
#                               data=data_out,
#                               columns=columns,
#                               title='Flask Bootstrap Table')
    template = render_template('result.html', results=data_out.values.tolist())
    return(template)



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_tokenizer()
    load_model()
    app.run(debug=True, port=8086)
