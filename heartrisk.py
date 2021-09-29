from flask import Flask,render_template,request
import joblib
import numpy as np

model=joblib.load('heart_risk_regression.sav')
model_poly=joblib.load('model_poly.sav')
model_qntl_data=joblib.load('model_qunt_data.sav')
model_qntl_target=joblib.load('model_qunt_target.sav')

app=Flask(__name__) #application


@app.route('/')
def index():

	return render_template('indexheart.html')
	
@app.route('/Predicte')
def Predicte():

	return render_template('Detailsheart.html')

@app.route('/getresults',methods=['POST'])
def getresults():

	gender_dict={'female':0,'male':1}
	smoke_dict={'no':0,'yes':1}
	bmp_dict={'no':0,'yes':1}
	diab_dict={'no':0,'yes':1}


	result=request.form 

	name=result['name']
	gender=result['gender']
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	sbp=float(result['sbp'])
	smoke=result['smoke']
	bpm=result['bpm']
	diab=result['diab']

	test_data=np.array([gender_dict[gender],age,tc,hdl,smoke_dict[smoke],bmp_dict[bpm],diab_dict[diab]]).reshape(1,-1)

	test_data=model_qntl_data.transform(test_data)
	test_data=model_poly.transform(test_data)

	prediction=model.predict(test_data)

	prediction=model_qntl_target.inverse_transform(prediction)

	resultDict={"name":name,"risk":round(prediction[0][0],2)}

	return render_template('patient_result_thivanka_new.html',results=resultDict)

app.run(debug=True)