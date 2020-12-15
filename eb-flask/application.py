from flask import Flask, render_template, url_for, flash, redirect
from forms import ModelInputForm
from model_engine import run_prediction
application = app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route("/")
@app.route("/home")
def home():
    return render_template('about.html', title='About')

@app.route("/model", methods=['GET', 'POST'])
def model():
    form = ModelInputForm()
    if form.validate_on_submit():
        flash(f'The model predicts your movie may be a {run_prediction(form.data)}.', 'success')
        return redirect(url_for('model'))
    return render_template('model.html', title='Movie Profit Prediction', form=form)
    

if __name__ == '__main__':
    app.run(debug=True)
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()