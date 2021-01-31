import flask
import numpy as np
import pickle

model= pickle.load(open('model/rfclassifier_jfam-v03.pkl', 'rb'))
Tfidf_vect = pickle.load(open('model/Tfidf_vect.pkl', 'rb'))
Encoder = pickle.load(open('model/Encoder.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('main.html'))

@app.route('/predict', methods=['POST'])
def predict():    
    features = [str(x) for x in flask.request.form.values()] 
    # final_features = [np.array(int_features)]
    prediction = Encoder.inverse_transform(model.predict(Tfidf_vect.transform(features)))
    
    # model.predict(final_features)

    # Encoder.inverse_transform(load_rfclassifier_jfam.predict(Tfidf_vect.transform(['leadership'])))

    # output = {0: 'not placed', 1:'placed'}

    return flask.render_template('main.html', prediction_text=prediction[0])

    # prediction_text='Student must be {} to workplace'.format([prediction[0]]))
    # prediction_text='Student must be {} to workplace'.format(output[prediction[0]]))

if __name__ == '__main__':
    app.run()