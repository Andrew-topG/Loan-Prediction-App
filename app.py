#import psycopg2
from flask import Flask, request, render_template, redirect, session
import numpy as np
import pickle
import requests
#import threading
#import queue
from datetime import datetime

app = Flask(__name__)
app.secret_key = "dev-test"  # Needed for session

# -------------------- Model Loading --------------------
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.loads(response.content)

try:
    with open("model_rft.pkl", "rb") as rf:
        model = pickle.load(rf)
    print("✅ Model loaded locally")
except Exception as e:
    print(f"⚠️ Local model load failed: {e}")
    model_url = "https://www.dropbox.com/scl/fi/qxtyk2obxs8gn1kq5dse6/model_rft.pkl?rlkey=s085ulecnnjzrmhpyvsqgzol4&st=s5hdb0kq&dl=1"
    model = load_model_from_url(model_url)
    print("✅ Model loaded from remote source")

# -------------------- Risk + Grade --------------------
Risk_cat = ["Very low", "Low", "Moderate", "High", "Very High"]
grade_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def assign_grade(int_rate):
    index = int((int_rate - 6) / 2.5)
    return grade_array[max(0, min(index, len(grade_array) - 1))]

def calculate_fields(income, debt, loan_amt, int_rate, term):
    annual_inc = income * 12
    dti = (debt / income) * 100 if income != 0 else 0
    if int_rate > 0:
        monthly_rate = int_rate / 1200
        installment = (loan_amt * monthly_rate) / (1 - (1 + monthly_rate) ** -term)
    else:
        installment = loan_amt / term
    return annual_inc, dti, installment

# -------------------- DB Config + Queue --------------------
#db_config = {
#    'dbname': 'loandb',
#    'user': 'postgres',
#    'password': 'Neha@1123',
#    'host': 'localhost',
#    'port': '5432'
#}

#db_queue = queue.Queue()

#def db_worker():
#    conn = psycopg2.connect(**db_config)
#    cur = conn.cursor()
#    while True:
#        data = db_queue.get()
#        if data is None: break  # Sentinel to end thread
#        try:
#            cur.execute("""
#                INSERT INTO predictions (
#                    income, debt, fico, int_rate, loan_amt, term,
#                    revol_bal, revol_util, open_acc, pub_rec_bankruptcies,
#                    prediction, prob_default, risk_category,
#                    application_type, home_ownership, loan_purpose, verification_status
#                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#            """, data)
#            conn.commit()
#        except Exception as e:
#            print("❌ DB Error:", e)
#    cur.close()
#    conn.close()

# Start DB thread
#threading.Thread(target=db_worker, daemon=True).start()

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs
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

        app_type = request.form['application_type']
        home_own = request.form['home_ownership']
        title = request.form['title']
        verification = request.form['verification_status']

        # Derived
        annual_inc, dti, installment = calculate_fields(income, debt, loan_amt, int_rate, term)
        grade = assign_grade(int_rate)

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

        final_input = np.array([
            annual_inc, dti, fico, installment, int_rate,
            loan_amt, open_acc, pub_rec_bankruptcies,
            revol_bal, revol_util, term,
            app_joint, home_own_, home_rent,
            *grade_encoded, *title_encoded, verification_verified
        ], dtype=np.float32).reshape(1, -1)

        prediction = model.predict(final_input)[0]
        probabilities = model.predict_proba(final_input)[0]
        prob_default = round(probabilities[1] * 100, 2)
        risk_category = Risk_cat[int(probabilities[1]*2.5)+1]
        result = "Loan Default" if prediction == 1 else "No Default"

        # Send to DB queue (non-blocking)
        #db_queue.put(tuple(map(lambda x: x.item() if hasattr(x, "item") else x, ( #converts np to python scalars to add in db
        #income, debt, fico, int_rate, loan_amt, term,
        #revol_bal, revol_util, open_acc, pub_rec_bankruptcies,
        #int(prediction), float(prob_default), risk_category,
        #app_type, home_own, title, verification
        #))))


        session['result_data'] = { # Store prediction in session
                'result': result,
                'prob_default': prob_default,
                'risk_category': risk_category
            }
        return redirect('/result')

    except Exception as e:
        return f"<h3>❌ Error: {str(e)}</h3>"
    
@app.route('/result')
def show_result():
    data = session.pop('result_data', None)
    if not data:
        return redirect('/')

    return f"""
    <link rel="stylesheet" href="/static/style.css">
    <main>
        <div class="container">
            <div style="text-align: center;">
                <h2>The Default Prediction for Borrower's Data:</h2>
                <h2>Result: {data['result']}</h2>
                <h2>Probability of Default: {data['prob_default']:.2f}%</h2>
                <h2>Risk Category: {data['risk_category']}</h2>
                <form action="/" method="get">
                    <button type="submit"><h2>🏠 Home</h2></button>
                </form>
            </div>
        </div>
    </main>
    """


# -------------------- Run Server --------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
