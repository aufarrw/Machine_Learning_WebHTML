from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# fitting scaler using pandas
filepath = "tabel_tembak.csv"
data_tembak = pd.read_csv(filepath)
data_tembak_sudut = data_tembak.drop(['tipe', 'waktu_lintas_det', 'sudut_elevasi_der', 'sudut_elevasi_rad'], axis=1)
data_tembak_sudut["jarak_km"] = data_tembak_sudut["jarak_m"] / 1000
data_tembak_sudut["jarak_cm"] = data_tembak_sudut["jarak_m"] * 100
scaler = StandardScaler()
scaler.fit(data_tembak_sudut)

# constant
pi = 22 / 7
t=np.arange(0,60,0.1)

# load models
model = load('model.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # getting data from post
    isian = request.form.get('isian')
    isian = int(isian)

    jarak_m = request.form.get('jarak_m')
    jarak_m = int(jarak_m)
    jarak_km = 0
    jarak_cm = 0

    input_query = (isian, jarak_m, jarak_km, jarak_cm)

    # convert meter to km and cm
    input_list = list(input_query)
    converted_km = input_list[1] / 1000
    converted_cm = input_list[1] * 100
    input_list[2] = converted_km
    input_list[3] = converted_cm
    input_query = tuple(input_list)

    # convert to array and reshape
    input_data_as_numpy = np.asarray(input_query)
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)

    # standardize data
    std_data = scaler.transform(input_data_reshaped)

    # predict
    result = model.predict(std_data)
    result = float(result)

    # convert result(derajat) to radian
    radian = result * (pi / 180)
    radian_not_converted = radian
    radian = round(radian * 1000)

    result = "{:.2f}".format(result)

    #matplotlib
    input_list_mat = list(input_query)
    isian_val = input_list_mat[0]
    angle = radian_not_converted
    angle_list = []

    if isian_val == 0:
        V = 67
        plt.ylim([0, 300])
        plt.xlim([0, 600])

    elif isian_val == 1:
        V = 89
        plt.ylim([0, 500])
        plt.xlim([0, 1000])

    elif isian_val == 2:
        V = 116
        plt.ylim([0, 1000])
        plt.xlim([0, 1500])

    elif isian_val == 3:
        V = 161
        plt.ylim([0, 1500])
        plt.xlim([0, 3000])

    elif isian_val == 4:
        V = 192
        plt.ylim([0, 2000])
        plt.xlim([0, 4000])

    elif isian_val == 5:
        V = 216
        plt.ylim([0, 2500])
        plt.xlim([0, 5000])

    elif isian_val == 6:
        V = 236
        plt.ylim([0, 3000])
        plt.xlim([0, 6000])

    elif isian_val == 7:
        V = 250
        plt.ylim([0, 3000])
        plt.xlim([0, 6500])

    elif isian_val == 8:
        V = 255
        plt.ylim([0, 3200])
        plt.xlim([0, 7000])

    x = V * np.cos(angle) * t
    y = V * np.sin(angle) * t + (0.5 * -9.8 * t ** 2)

    plt.plot(x, y, color="red")
    angle_list.append(rf'{np.rad2deg(angle):.1f}$\degree$')
    plt.xlabel("range (m)")
    plt.ylabel("height (m)")
    plt.title('Projectile Motion: 'rf'{np.rad2deg(angle):.1f}$\degree$')
    plt.legend(angle_list)
    plt.savefig('static/foo.png', transparent=True)
    plt.close()
    plt.clf()

    # send as json
    return jsonify({'sudut_elevasi_der': str(result), 'sudut_elevasi_rad': str(radian)})


@app.route('/predicthtml', methods=['POST'])
def predicthtml():
    # getting data from post
    isian = request.form.get('isian')
    isian = int(isian)

    jarak_m = request.form.get('jarak_m')
    jarak_m = int(jarak_m)
    jarak_km = 0
    jarak_cm = 0

    input_query = (isian, jarak_m, jarak_km, jarak_cm)

    # convert meter to km and cm
    input_list = list(input_query)
    converted_km = input_list[1] / 1000
    converted_cm = input_list[1] * 100
    input_list[2] = converted_km
    input_list[3] = converted_cm
    input_query = tuple(input_list)

    # convert to array and reshape
    input_data_as_numpy = np.asarray(input_query)
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)

    # standardize data
    std_data = scaler.transform(input_data_reshaped)

    # predict
    result = model.predict(std_data)
    result = float(result)

    # convert result(derajat) to radian
    radian = result * (pi / 180)
    radian_not_converted = radian
    radian = round(radian * 1000)

    result = "{:.2f}".format(result)

    #matplotlib
    input_list_mat = list(input_query)
    isian_val = input_list_mat[0]
    angle = radian_not_converted
    angle_list = []

    if isian_val == 0:
        V = 67
        plt.ylim([0, 300])
        plt.xlim([0, 600])

    elif isian_val == 1:
        V = 89
        plt.ylim([0, 500])
        plt.xlim([0, 1000])

    elif isian_val == 2:
        V = 116
        plt.ylim([0, 1000])
        plt.xlim([0, 1500])

    elif isian_val == 3:
        V = 161
        plt.ylim([0, 1500])
        plt.xlim([0, 3000])

    elif isian_val == 4:
        V = 192
        plt.ylim([0, 2000])
        plt.xlim([0, 4000])

    elif isian_val == 5:
        V = 216
        plt.ylim([0, 2500])
        plt.xlim([0, 5000])

    elif isian_val == 6:
        V = 236
        plt.ylim([0, 3000])
        plt.xlim([0, 6000])

    elif isian_val == 7:
        V = 250
        plt.ylim([0, 3000])
        plt.xlim([0, 6500])

    elif isian_val == 8:
        V = 255
        plt.ylim([0, 3200])
        plt.xlim([0, 7000])

    x = V * np.cos(angle) * t
    y = V * np.sin(angle) * t + (0.5 * -9.8 * t ** 2)

    plt.plot(x, y, color="red")
    angle_list.append(rf'{np.rad2deg(angle):.1f}$\degree$')
    plt.xlabel("range (m)")
    plt.ylabel("height (m)")
    plt.title('Projectile Motion: 'rf'{np.rad2deg(angle):.1f}$\degree$')
    plt.legend(angle_list)
    plt.savefig('static/foo.png', transparent=True)
    plt.close()

    path="/"
    #Open Graph Image
    PlotGraph = os.path.join(path, "/static", "foo.png")
    plt.clf()

    # send as html
    return render_template('index.html', sudut_elevasi_der='Sudut Derajat: {} derajat'.format(result),
                           sudut_elevasi_rad='Sudut Radian: {} rad'.format(radian), MortarGraph= PlotGraph)


if __name__ == '__main__':
    app.run(debug=True)
