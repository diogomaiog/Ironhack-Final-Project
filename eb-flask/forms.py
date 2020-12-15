from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, SelectMultipleField, SelectField, IntegerField
from wtforms.validators import InputRequired, Length, NumberRange
import pickle

# populate list of valid actors/actresses
valid_actors = sorted(pickle.load(open('valid_actors.pkl',"rb")))
valid_directors = sorted(pickle.load(open('valid_directors.pkl',"rb")))

class ModelInputForm(FlaskForm):
    year = IntegerField('Year the movie will come out:', 
                        validators=[InputRequired(), NumberRange(min=1980, max=2025, message='enter number between 1980 and 2025')])
    
    runtimeMinutes = IntegerField('Movie duration in minutes:', 
                        validators=[InputRequired(), NumberRange(min=60, max=300,message='enter number beween 60 and 300')])

    genres = SelectMultipleField('Movie genres:', 
                        choices=[('action','Action'),('adventure','Adventure'),('animation','Animation')
                                 ,('biography','Biography'),('comedy','Comedy'),('crime','Crime')
                                 ,('documentary','Documentary'),('drama','Drama'),('family','Family')
                                 ,('fantasy','Fantasy'),('history','History'),('horror','Horror')
                                 ,('music','Music'),('musical','Musical'),('mistery','Mistery')
                                 ,('news','News'),('romance','Romance'),('sci-fi','Sci-fi')
                                 ,('sport','Sport'),('thriller','Thriller'),('war','War')
                                 ,('western','Western')], validators=[InputRequired()])

    rating = SelectField(u'Movie Rating (age restrictions):', choices=[('G', 'All ages admitted'), ('PG', 'Parental Guidance Suggested'), 
                                             ('PG-13', 'Parents Strongly Cautioned'),('R','Restricted'),
                                             ('NC-17','Adults only')],
                                             validators=[InputRequired()])

    story = TextAreaField(u'Briefly describe the movie plot (around 150 words).',
                          validators=[InputRequired(), Length(max=1000)])               
       
    director = SelectField(u'Movie director:', choices = valid_directors,
                          validators=[InputRequired(), Length(max=50)])

    actor1 = SelectField(u'First actor/actress:', choices = valid_actors,
                          validators=[InputRequired(), Length(max=50)])      

    actor2 = SelectField(u'Second actor/actress:', choices = valid_actors,
                          validators=[Length(max=50)])     

    actor3 = SelectField(u'Third actor/actress:', choices = valid_actors,
                          validators=[Length(max=50)])        

    submit = SubmitField('Make Prediction')
