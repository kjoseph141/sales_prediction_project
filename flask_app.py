from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import numpy as np

import os
import pickle

app = Flask(__name__)

#global NN_model, graph
#
#with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
     

#graph = tf.get_default_graph()

#tf_config = some_custom_config
sess = tf.Session()#config=tf_config)
graph = tf.get_default_graph()

set_session(sess)
NN_model = load_model('/Users/kevinjoseph/Python/Sales Prediction Project/NN_full_model_nb.h5')

#NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

#Load in data
X_test = pickle.load(open('X_test_nb.pickle','rb'))


@app.route('/')
def index():
    return render_template("index_v2.html")

@app.route("/", methods = ["POST"])
def prediction():
    ticker = request.form.get("ticker")
    
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        #Generate prediction from most recent data
        pred = int(NN_model.predict(X_test.iloc[0:1,:], batch_size=32))
        
    return render_template('index_v2.html', ticker=ticker, pred=pred*10)
      


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)