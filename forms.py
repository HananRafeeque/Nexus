from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, HiddenField
from wtforms.validators import DataRequired, InputRequired


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()]) 

    submit = SubmitField('Login')

class UploadForm(FlaskForm):
    file = FileField('File to Upload' , validators=[InputRequired()])
    csrf_token = HiddenField()
    submit = SubmitField('Upload')