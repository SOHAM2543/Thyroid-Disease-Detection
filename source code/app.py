from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from DatabaseConnection.database_conn import DBConn

application = Flask(__name__)
app = application

def convert_bool(value):
    return value == 't'

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        # Capture form data
        age = int(request.form.get('age'))
        on_thyroxine = request.form.get('on_thyroxine')
        goitre = request.form.get('goitre')
        TSH = float(request.form.get('TSH'))
        TSH_measured = request.form.get('TSH_measured')
        T3 = float(request.form.get('T3'))
        TT4_measured = request.form.get('TT4_measured')
        TT4 = float(request.form.get('TT4'))
        FTI = float(request.form.get('FTI'))
        referral_source = request.form.get('referral_source')

        # Debugging output
        #print(f"TT4 Value: {TT4}")  # Check if TT4 is being captured correctly

        data = CustomData(
            age=age,
            on_thyroxine=on_thyroxine,
            goitre=goitre,
            TSH=TSH,
            TSH_measured=TSH_measured,
            T3=T3,
            TT4_measured=TT4_measured,
            TT4=TT4,
            FTI=FTI,
            referral_source=referral_source
        )

        # Converting data to dataframe for prediction
        final_new_data = data.get_data_as_dataframe()
        # print(final_new_data.columns)  # Check if TT4 is present in the DataFrame
        # print(final_new_data) #Check data

        # Making prediction
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        result = pred[0]
        output = ""

        if result == 0:
            output = "compensated hypothyroid"
        elif result == 1:
            output = "negative"
        elif result == 2:
            output = "primary hypothyroid"
        else:
            output = "secondary hypothyroid"
        
        final_new_data['output'] = output

        collected_data_dict = final_new_data.to_dict()
        # Removing the key `0` from the dictionary
        cleaned_data_dict = {key: value[0] for key, value in collected_data_dict.items()}

        #print(cleaned_data_dict)
        db = DBConn()
        db.insert_data(cleaned_data_dict)
        return render_template('form.html', final_result=output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
