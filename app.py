from flask import Flask,request,render_template
import pandas as pd
import pickle

app = Flask(__name__)

model1 = pickle.load(open('model1.pkl', 'rb')) 

@app.route('/')

def  index():
    return render_template('ui.html') 

@app.route('/predict', methods=['POST', 'GET']) 

def predict():
    
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    
    df1 = pd.DataFrame([{'1': int(data1) ,'2': int(data2) ,'3': int(data3) ,'4': int(data4) ,'5': int(data5) ,'6': int(data6) ,'7': int(data7) ,'8': int(data8) ,'9': int(data9) ,'10': float(data10) ,'11': int(data11) ,'12': int(data12) ,'13': int(data13)}])

    prediction = model1.predict(df1)
    
    if prediction==0:
        prediction= " dont have any heart related Disease"
    else:
        prediction=" have some Heart_ Disease"
     
    print(prediction) 
    
   
    return render_template('op.html', pred='you{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
