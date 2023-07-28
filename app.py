from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

pipe = pickle.load(open('pipe.pkl','rb'))

@app.route('/')
def home():
    return 'Hello World'

@app.route('/api', methods=['POST'])
def api():
    batting_team = request.form.get('batting_team')
    bowling_team = request.form.get('bowling_team')
    city = request.form.get('city')
    runs_left = request.form.get('runs_left')
    balls_left = request.form.get('balls_left')
    wickets = request.form.get('wickets')
    total_runs_x = request.form.get('total_runs_x')
    crr = request.form.get('crr')
    rrr = request.form.get('rrr')

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[total_runs_x],'crr':[crr],'rrr':[rrr]})

    print(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    batting_team_ = batting_team + "- " + str(round(win*100)) + "%"
    bowling_team_ = bowling_team + "- " + str(round(loss*100)) + "%"

    response = jsonify({'batting team': batting_team_, 'bowling team' : bowling_team_})
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
