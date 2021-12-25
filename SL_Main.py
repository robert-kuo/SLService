from flask import Flask

myapp = Flask(__name__)


@myapp.route("/")
def hello():
    return 'SL Service! ...   ??? !!!!!!!!!!!!!!!!!!!'

@myapp.route("/test")
def test():
    return 'SL test !!  ...'
