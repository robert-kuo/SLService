import datetime as dt
import warnings
import time
import Opt_SL_Run
import os
import threading

def getdiff(date1, date2):
      date_format = '%Y-%m-%d %H:%M:%S'
      a = dt.datetime.strptime(date1, date_format)
      b = dt.datetime.strptime(date2, date_format)
      delta = b - a
      return delta.days * 86400 +delta.seconds

checktime = dt.datetime.strftime(dt.datetime.now(),  '%Y-%m-%d %H:%M:%S')
warnings.filterwarnings('ignore')

if os.name == 'nt':
    mainpath = 'd:\\opt_web'
else:
    mainpath = '/aidata/DIPS'
sfile = os.path.join(mainpath, 'stagerun.txt')
i = 1
while True:
    if i % 6 == 0: print('=====  Watching ', dt.datetime.strftime(dt.datetime.now(),  '%Y-%m-%d %H:%M:%S'), '  =====')
    time.sleep(5)
    if os.path.isfile(sfile):
        fn = open(sfile, 'r')
        lst_data = fn.readlines()
        fn.close()
        bchange = False
        for s in lst_data:
            lst_ret = s.replace('\n', '').split(',')
            if getdiff(checktime, lst_ret[0]) > 0:
                print(lst_ret[1], lst_ret[2], lst_ret[3])
                t = threading.Thread(target=Opt_SL_Run.Run_Learning, args = (lst_ret[1], lst_ret[2], lst_ret[3],))
                t.start()
                bchange = True
        if bchange: checktime = dt.datetime.strftime(dt.datetime.now(),  '%Y-%m-%d %H:%M:%S')
    i += 1


