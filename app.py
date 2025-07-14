from flask import Flask, request, render_template,redirect, url_for
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier 
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load model from URL if local load fails
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if download fails
    return pickle.loads(response.content)

try:
    # Try loading from local file
    with open("model_rft.pkl", "rb") as rf:
        model = pickle.load(rf)
    print("✅ Model loaded")
except Exception as e:
    print(f"⚠️ Local model load failed: {e}")
    print("⏬ Trying to load model from DropBox")
    model_url = "https://www.dropbox.com/scl/fi/qxtyk2obxs8gn1kq5dse6/model_rft.pkl?rlkey=s085ulecnnjzrmhpyvsqgzol4&st=s5hdb0kq&dl=1"
    model = load_model_from_url(model_url)
    print("✅ Model loaded from remote source.")



# model = pickle.load(open("model_rft.pkl", "rb"))


#Risk Category 
Risk_cat = ["Very low","Low","Moderate","High","Very High"]

    
# Grade assignment based on interest rate using array
grade_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def assign_grade(int_rate):
    index = int((int_rate - 6) / 2.5)
    index = max(0, min(index, len(grade_array) - 1))  # Clamp to 0–6
    return grade_array[index]

# --- Derived Calculations ---
def calculate_fields(income, debt, loan_amt, int_rate, term):
    annual_inc = income * 12
    dti = (debt / income) * 100 if income != 0 else 0
    if int_rate > 0:
        monthly_rate = int_rate / 1200
        installment = (loan_amt * monthly_rate) / (1 - (1 + monthly_rate) ** -term)
    else:
        installment = loan_amt / term
    return annual_inc, dti, installment

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric inputs
        income = float(request.form['month_income'])
        debt = float(request.form['debt'])
        fico = float(request.form['fico_range'])
        int_rate = float(request.form['int_rate'])
        loan_amt = float(request.form['loan_amnt'])
        term = int(request.form['term'])
        revol_bal = float(request.form['revol_bal'])
        revol_util = float(request.form['revol_util'])
        open_acc = float(request.form['open_acc'])
        pub_rec_bankruptcies = float(request.form['pub_rec_bankruptcies'])

        # Derived fields
        annual_inc, dti, installment = calculate_fields(income, debt, loan_amt, int_rate, term)
        grade = assign_grade(int_rate)
        print("grade: ",grade)

        # Categorical inputs
        app_type = request.form['application_type']
        home_own = request.form['home_ownership']
        title = request.form['title']
        verification = request.form['verification_status']

        # One-hot encode
        app_joint = app_type == "Joint App"
        home_rent = home_own == "RENT"
        home_own_ = home_own == "OWN"
        verification_verified = verification == "Verified"

        grade_levels = ['B', 'C', 'D', 'E', 'F', 'G']
        grade_encoded = [grade == g for g in grade_levels]

        title_list = [
            'Car financing', 'Credit card refinancing', 'Debt consolidation', 'Green loan',
            'Home buying', 'Home improvement', 'Major purchase', 'Medical expenses',
            'Moving and relocation', 'Other', 'Vacation'
        ]
        title_encoded = [title == t for t in title_list]

        # Final input array
        final_input = np.array([
            annual_inc, dti, fico, installment, int_rate,
            loan_amt, open_acc, pub_rec_bankruptcies,
            revol_bal, revol_util, term,
            app_joint, home_own_, home_rent,
            *grade_encoded,
            *title_encoded,
            verification_verified
        ], dtype=np.float32).reshape(1, -1)

        #print(final_input)
        prediction = model.predict(final_input)[0]
        result = "Loan Default" if prediction == 1 else "No Default"

        # Make prediction (probability)
        probabilities = model.predict_proba(final_input)[0]  # Get first (and only) row
        # Extract default probability
        prob_default = probabilities[1]  # class '1' is usually "default"
        print(int(prob_default*100),"% Default")

        Risk = Risk_cat[int(prob_default*2)+1]

        return f"""
        <h3> This is your predicted score </h3> \n
          <h2>Prediction Result: {result}</h2> \n 
          <h2>Probability of Default: {prob_default:.2%}</h2> \n 
          <h2>Risk Category: {Risk}</h2>\n
          <form action="/" method="get">
            <button type="submit"><h2>🏠 Home</h2></button>
        </form>
        """

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80,debug=True)
