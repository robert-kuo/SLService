from flask import Flask
from flask import abort

import os, subprocess
import Opt_func
import Opt_SL_Func

myapp = Flask(__name__)
if os.name == 'nt':
    mainpath = 'd:\\opt_web'
    ip = Opt_func.GetIP()
else:
    mainpath = '/aidata/DIPS'
    ip = ''

subprocess.Popen(['python', 'Opt_test.py'])

@myapp.route("/")
def hello():
    return 'sL Service...'

@myapp.route('/SL/v0.1/<string:taskname>/<string:stagename>/RUN',  methods = ['GET'])
def SL_RunStage(taskname, stagename):
    spath = os.path.join(os.path.join(mainpath, taskname), stagename)
    ret = 200
    print(os.path.join(os.path.join(mainpath, taskname), 'Dataset'))
    if os.path.isdir(os.path.join(os.path.join(mainpath, taskname), 'Dataset')):
        if os.path.isdir(spath):
            Opt_SL_Func.Run_Stage(mainpath, taskname, stagename)
        else:
            abort(404)
    else:
        abort(404)
    return '', ret

@myapp.route('/SL/v0.1/<string:taskname>/<string:stagename>/STOP',  methods = ['GET'])
def SL_StopStage(taskname, stagename):
    spath = os.path.join(os.path.join(mainpath, taskname), stagename)
    ret = 200
    if os.path.isdir(spath):
        Opt_SL_Func.Stop_Stage(mainpath, taskname, stagename)
    else:
        abort(404)
    return '', ret