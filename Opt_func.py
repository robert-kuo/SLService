import os
import datetime as dt
import numpy as np
import json, math
import shutil, pathlib
import pandas as pd
import functools

import socket

from openpyxl.chart import (LineChart, Reference,)
from flask import send_file

def getchar(i):
    return chr(65 + int(i / 10)) + str(i % 10)

def getdate(sday, dday):
    end_date = dt.datetime.strptime(sday, '%Y-%m-%d') + dt.timedelta(days=int(dday))
    return str(end_date.year) + '-' + '{0:02d}'.format(end_date.month) + '-' + '{0:02d}'.format(end_date.day)

def GetIP():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ret = s.getsockname()[0]
    s.close()
    return ret

def gethours(date1, date2):
    diff = date2 - date1
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    # seconds = seconds % 60
    return hours + minutes / 60

def getdays(date1, date2):
    delta = dt.datetime.strptime(date2, '%Y-%m-%d') - dt.datetime.strptime(date1, '%Y-%m-%d')
    return delta.days

def list_index(lst, s):
    try:
        ret = lst.index(s)
    except:
        ret = -1
    return ret

def sortlist_bynum(lst):
    for i in range(len(lst)):
        lst[i] = lst[i][0] + '{0:02d}'.format(int(lst[i][1:]))
    lst.sort()
    for i in range(len(lst)):
        lst[i] = lst[i][0] + str(int(lst[i][1:]))
    return lst

def GetLineID_sorted(Linename, df_lines):
    lst_copy = sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())  #['D2', 'Y6', 'Y4', 'Y7', 'Y9', 'Y8', 'Y10']  #lst.copy()
    return list_index(lst_copy, Linename)

def GetLineName_sorted(index, df_lines):
    lst_copy = sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())  #['D2', 'Y6', 'Y4', 'Y7', 'Y9', 'Y8', 'Y10']  #lst.copy()
    return lst_copy[index]

def Get_blockcount(index, lstblk):
    return len(list(filter(None, lstblk[index])))

def readlines(df_linedata, df_products):
    df_lines = df_linedata.reset_index()
    df_lines.drop(columns=[df_lines.columns[0]], axis=1, inplace=True)
    df_lines.columns = ['line_name', 'Usable', 'Line Begin', 'Line_Status', 'mold_no', 'mfg_width', 'part_no', 'type', 'material', 'composition', 'lenti_pitch', 'roller_position', 'width', 'thickness']
    for i in range(df_linedata.shape[0]):
        # smold = df_linedata.iloc[i,4]
        spart = df_linedata.iloc[i,6].strip()
        if spart != '':
            df_tmp = df_products[df_products['part_no'] == spart]
            if df_tmp.shape[0] > 0:
                df_lines.loc[i, 'part_no'] = spart
                df_lines.loc[i, 'type'] = df_tmp.loc[df_tmp.index[0], 'type']
                df_lines.loc[i, 'lenti_pitch'] = df_tmp.loc[df_tmp.index[0], 'lenti_pitch']
                df_lines.loc[i, 'roller_position'] = df_tmp.loc[df_tmp.index[0], 'roller_position']
                df_lines.loc[i, 'width'] = df_tmp.loc[df_tmp.index[0], 'width']
                df_lines.loc[i, 'thickness'] = df_tmp.loc[df_tmp.index[0], 'height']
                df_lines.loc[i, 'composition'] = df_tmp.loc[df_tmp.index[0], 'composition']
                df_lines.loc[i, 'material'] = df_tmp.loc[df_tmp.index[0], 'material']
            else:
                df_lines.loc[i, 'Line_Status'] = 'Shutdown'
                df_lines.loc[i, 'material'] = 'PMMA'
                df_lines.loc[i, 'part_no'] = spart
    return df_lines

# mold data
def readmolds(df_lines, df_molddata):
    n2 = df_molddata.shape[0]
    df_mold = df_molddata.drop(0)
    if df_mold.columns[1] != 'mold_no': df_mold.insert(1, 'mold_no', '')
    n1 = df_mold[df_mold['模頭號碼'] == ''].index[0]
    df_mold.drop([*range(n1, n2)], inplace=True)

    df_mold.columns = ['mold_code', 'mold_no', 'width_max', 'width_min', 'thickness_max', 'thickness_min', 'lip']
    df_mold.sort_values(by=['width_max', 'thickness_max', 'mold_no'], inplace=True)
    df_mold.reset_index(inplace=True)
    df_mold['mold_no'] = df_mold['mold_code']
    df_mold['mold_code'] = [chr(x) + str(df_mold.loc[x - 65, 'width_max']) + chr(64 + int(df_mold.loc[x - 65, 'thickness_max'])) for x in range(65, 65 + df_mold.shape[0])]
    lst_LineName = [df_lines[df_lines['mold_no'] == df_mold.loc[x, 'mold_no']] for x in range(df_mold.shape[0])]
    df_mold['Usage'] = ['' if x.shape[0] == 0 else x.loc[x.index[0], 'line_name'] for x in lst_LineName]   #df_lines[df_lines['mold_no'] == df_mold['mold_no']]['line_name']
    df_mold.drop([df_mold.columns[0]], axis=1, inplace=True)
    return df_mold

# add mold_code to lines
def updatelines(df_lines, df_molds, shiftday):
    df_lines.insert(4, 'mold_code', '', True)
    df_lines['mold_code'] = ['' if df_lines.loc[x, 'mold_no'] == '' else df_molds.loc[df_molds[df_molds['mold_no'] == df_lines.loc[x, 'mold_no']].index[0], 'mold_code'] for x in range(df_lines.shape[0])]
    if shiftday !=0:
        for i in range(df_lines.shape[0]):
            if df_lines.loc[i, 'Line Begin'] != '': df_lines.loc[i, 'Line Begin'] = df_lines.loc[i, 'Line Begin'] + dt.timedelta(days=shiftday)
    return df_lines

def OrderLost_Trendchart(df_order_lost, df_lines, end_day):
    lst_lines = sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())
    lst_lines.append('AVERAGE')

    n_lines = len(lst_lines)
    mindate = df_lines[df_lines['Usable'] == 'YES']['Line Begin'].min()
    n_days = getdays(dt.datetime.strftime(mindate, '%Y-%m-%d'), end_day)
    data_array = np.full((n_lines, n_days + 1), 0, dtype=float)
    for i in range(df_order_lost.shape[0]):
        lst_dolines = df_order_lost.loc[i, 'Do_Lines'].split(';')
        p_hour = df_order_lost.loc[i, 'Production_Hours'] / len(lst_dolines)
        n_a_o = dt.datetime.strptime(df_order_lost.loc[i, 'not_after'], '%Y-%m-%d %H:%M:%S')
        for litem in lst_dolines:
            n_b_o = dt.datetime.strptime(df_order_lost.loc[i, 'not_before'], '%Y-%m-%d %H:%M:%S')
            index = list_index(lst_lines, litem)
            if index >= 0:
                n_b_l = df_lines.loc[df_lines[df_lines['line_name'] == litem].index[0], 'Line Begin']
                if n_b_l > n_b_o: n_b_o = n_b_l
                ba_day = getdays(dt.datetime.strftime(n_b_o, '%Y-%m-%d'), dt.datetime.strftime(n_a_o, '%Y-%m-%d')) + 1
                x = getdays(dt.datetime.strftime(mindate, '%Y-%m-%d'), dt.datetime.strftime(n_b_o, '%Y-%m-%d'))
                for v in range(x, x + ba_day):
                    data_array[index, v] += p_hour / ba_day

    for i in range(data_array.shape[1]):
        sum = 0
        count = 0
        for k in range(data_array.shape[0] - 1):
            sum += data_array[k, i]
            count += 1
            # if data_array[k, i] > 0: count += 1
        data_array[data_array.shape[0] - 1, i] = 0 if count == 0 else sum / count

    columns = ['Date']
    for i in range(data_array.shape[0]):
        columns.append(lst_lines[i])
    df_trendlost = pd.DataFrame(columns=columns)
    for k in range(data_array.shape[1]):
        lst_tmpdata = [dt.datetime.strftime(mindate + dt.timedelta(days=k), '%m-%d')]
        for i in range(data_array.shape[0]):
            lst_tmpdata.append(data_array[i, k])
        df_trendlost.loc[k] = lst_tmpdata
    return df_trendlost

def FileList(mainpath, taskname, dirname):
    spath = os.path.join(mainpath, taskname)
    dpath = os.path.join(spath, dirname)
    lst_file = []
    if os.path.isdir(dpath):
        ret = 200
        lst_dir = os.listdir(dpath)
        for x in lst_dir:
            if os.path.isfile(os.path.join(dpath, x)): lst_file.append(x)
    else:
        ret = 404
    return {"Files": lst_file}, ret

def DirList(dirpath, dirFilter, label):
    lst_folder = []
    if os.path.isdir(dirpath):
        ret = 200
        lst_dir = os.listdir(dirpath)
        for x in lst_dir:
            if os.path.isdir(os.path.join(dirpath, x)):
                if dirFilter == '':
                    lst_folder.append(x)
                elif x[:len(dirFilter)] == dirFilter:
                    lst_folder.append(x)
    else:
        ret = 404
    return {label: lst_folder}, ret

def get_dolines(lstline, lstproduct):
    if ''.join(lstproduct).strip() == '':
        ret = ';'.join(lstline)
    else:
        lstret = []
        for v in range(len(lstline)):
            for u in range(len(lstproduct)):
                if lstline[v] == lstproduct[u]: lstret.append(lstline[v])
        ret = ';'.join(lstret)
    return ret

def get_domolds(df_products, df_molds, product_code):
    ret = ''
    if df_products[df_products['product_code'] == product_code].shape[0] > 0:
        i = df_products[df_products['product_code'] == product_code].index[0]
        width = df_products[df_products['product_code'] == product_code].loc[i, 'width']
        height = df_products[df_products['product_code'] == product_code].loc[i, 'height']
        if width < 1000: width = 1000
        mask = functools.reduce(np.logical_and, (df_molds['width_min'] <= width, df_molds['width_max'] >= width, df_molds['thickness_min'] <= height, df_molds['thickness_max'] >= height))
        if df_molds[mask].shape[0] > 0:
            lstdata = df_molds[mask]['mold_code'].to_list()
            ret = ';'.join([str(elem) for elem in lstdata])
    return ret

# generate orders
def readorders(df_orderdata, df_products, df_lines, df_molds):
    begin_day = getdate(str(df_orderdata.columns[15])[:10], -30)
    end_day = str(df_orderdata.columns[len(df_orderdata.columns) - 1])[:10]

    columns = ['order_code', 'product_code', 'part_no', 'type', 'material', 'composition', 'width', 'length', 'height', 'density',
               'not_before', 'not_after', 'quantity', 'Production_Hours', 'Do_Lines', 'Do_Molds']
    df_orders = pd.DataFrame(columns=columns)
    m = 0
    lstdata = []
    for k in range(15, len(df_orderdata.columns)):
        for i in range(df_orderdata.shape[0]):
            if df_orderdata.loc[i, df_orderdata.columns[k]] != '' and df_orderdata.loc[i, df_orderdata.columns[k]] > 0:
                s = ('L' if df_orderdata.loc[i, df_orderdata.columns[5]] == '結構板' or df_orderdata.loc[i, df_orderdata.columns[5]] == 'lenti' else 'P') + '{0:03d}'.format(i)
                ctype = 'lenti' if s[0] == 'L' else 'plate'

                # part_no
                product_code = df_orderdata.loc[i, df_orderdata.columns[0]]
                lst_productlines = df_products[df_products['product_code'] == product_code]['assigned_lines'].to_list()
                lst_productlines = ':'.join(lst_productlines).split(':')
                df_tmp = df_products[df_products['product_code'] == product_code].reset_index()

                day_quantity = 0
                part_no = ''
                composition = ''
                material = df_orderdata.loc[i, 'material']
                if df_tmp.shape[0] > 0:
                    n_day = 42 if df_tmp.loc[0, 'LT'] == '' else df_tmp.loc[0, 'LT']
                    not_before = getdate(str(df_orderdata.columns[k])[:10], - n_day - 1)
                    product_code = df_tmp.loc[0, 'product_code']
                    part_no = df_tmp.loc[0, 'part_no']
                    day_quantity = df_tmp.loc[0, 'throughput']
                    composition = df_tmp.loc[0, 'composition']  # str(int(df_tmp.loc[0, 'composition'] * 100)) + '%'
                else:
                    not_before = getdate(str(df_orderdata.columns[k])[:10], - 43)
                if getdays(begin_day, not_before) < 0: not_before = begin_day
                not_after = getdate(str(df_orderdata.columns[k])[:10], -1) + ' 23:59:59'

                if day_quantity == 0:
                    print(s + getchar(m) + ' 日產量=0')
                else:
                    lst_alllines = df_lines[np.logical_and(df_lines['Usable'] == 'YES', True)]['line_name'].to_list()
                    do_lines = get_dolines(lst_alllines, lst_productlines)
                    do_molds = get_domolds(df_products, df_molds, product_code)
                    Prod_hour = math.ceil(df_orderdata.loc[i, df_orderdata.columns[k]] * 48 / day_quantity) / 2
                    n = lstdata.count(df_tmp.loc[0, 'product_code'])
                    lstdata.append(df_tmp.loc[0, 'product_code'])
                    df_orders.loc[m] = [df_tmp.loc[0, 'product_code'] + getchar(n), product_code, part_no, ctype, material, composition, df_orderdata.loc[i, 'width'], df_orderdata.loc[i, 'length'],
                                        df_orderdata.loc[i, 'height'], df_orderdata.loc[i, 'density'], not_before, not_after, df_orderdata.loc[i, df_orderdata.columns[k]], Prod_hour, do_lines, do_molds]
                    m += 1

    df_orders.sort_values('not_before', inplace=True)
    df_orders['O_Status'] = 'Waiting'
    df_orders['not_before'] = df_orders['not_before'].astype('datetime64[ns]')
    df_orders['not_after'] = df_orders['not_after'].astype('datetime64[ns]')
    df_orders.reset_index(inplace=True)
    return begin_day, end_day, df_orders

#drop columns
def order_dropcolumns(df_orderdata, demand_start, demand_end):
    s1 = 'part_no'
    s2 = 'MFG產出'
    s3 = str(demand_start + dt.timedelta(days=-1))[:10]
    s4 = str(demand_end)[:10]
    lst_column = []
    bAppend = True
    for i in range(len(df_orderdata.columns)):
        #print(df_orderdata.columns[i])
        sdata = str(df_orderdata.columns[i])
        if sdata == s1 or sdata[:5] == s2 or sdata[:len(s3)] == s3: bAppend = False
        if bAppend: lst_column.append(df_orderdata.columns[i])
        if sdata == s1 or sdata[:5] == s2 or sdata[:len(s4)] == s4: bAppend = True
    df_orderdata.drop(lst_column, axis=1, inplace=True)
    lst_column = []
    for i in range(2, len(df_orderdata.columns)):
        sdata = str(df_orderdata.columns[i])
        if sdata.count('-') != 2 and sdata[:5] != s2: lst_column.append(df_orderdata.columns[i])
    df_orderdata.drop(lst_column, axis=1, inplace=True)
    return df_orderdata

# drop rows
def order_droprows(df_orderdata):
    lst_row = []
    for i in range(df_orderdata.shape[0]):
        if df_orderdata.iloc[i,0] == '':
            lst_row.append(i)
        elif df_orderdata.iloc[i,1] != 'Demand':
            lst_row.append(i)
    df_orderdata.drop(lst_row, inplace=True)
    df_orderdata.reset_index(inplace=True)
    df_orderdata.drop(columns=[df_orderdata.columns[0], df_orderdata.columns[2]], inplace=True)
    return df_orderdata

def updateorderdata(df_products, df_orderdata):
    df_orderdata.insert(0, 'Product_ID', '', True)
    df_orderdata.insert(0, 'product_code', '', True)
    lst_partno = []
    for i in range(df_orderdata.shape[0]):
        lst_partno.append(df_orderdata.loc[i, 'part_no'])
        df_orderdata.iloc[i, 1] = lst_partno.count(df_orderdata.loc[i, 'part_no'])
    for i in range(df_orderdata.shape[0]):
        prodid = df_orderdata.loc[i, 'Product_ID']
        partno = df_orderdata.loc[i, 'part_no']
        df_tmp = df_products[df_products['part_no'] == partno]
        #print(i, prodid, partno)
        df_orderdata.loc[i, 'product_code'] = '' if df_tmp.shape[0] == 0 else df_products.loc[df_tmp.index[prodid - 1], 'product_code']
        #df_orderdata.loc[i, 'product_code'] = '' if df_tmp.shape[0] < prodid else df_products.loc[df_tmp.index[prodid - 1], 'product_code']
    df_orderdata.drop(columns=['Product_ID'], inplace=True)
    return df_orderdata

def GetEvery_PHs(lst_blk, n_count, df_lines):
    min_day = lst_blk[0][0][1][:10]
    for i in range(1, len(lst_blk)):
        t = lst_blk[i][0][1][:10]
        if t < min_day: min_day = t

    tx = dt.datetime.strptime(min_day, '%Y-%m-%d')
    lst_lines = sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())
    lst_PHs = [0] * n_count
    for i in range(len(lst_lines)):
        m = Get_blockcount(i, lst_blk)
        for k in range(1, m):
            if lst_blk[i][k][0] == 'Production' or lst_blk[i][k][0] == 'Tunning-Production':
                start_time = dt.datetime.strptime(lst_blk[i][k][1], '%Y-%m-%d %H:%M:%S')
                end_time = start_time + dt.timedelta(hours=lst_blk[i][k][2])
                w1 = (start_time - tx).days
                w2 = (end_time - tx).days
                for p in range(w1, w2 + 1):
                    t1 = tx + dt.timedelta(days=p)
                    if t1 < start_time: t1 = start_time
                    t2 = tx + dt.timedelta(days=p+1)
                    if t2 > end_time: t2 = end_time
                    v = gethours(t1, t2)
                    if v > 0: lst_PHs[p] += v
    return lst_PHs

def Get_orders_WaitandNotWait(df_orders):
    df_orders_notWait = df_orders[np.logical_and(df_orders['O_Status'] != 'Waiting', True)]
    df_orders_notWait.reset_index(inplace=True)
    df_orders_notWait.drop(columns=[df_orders_notWait.columns[0]], inplace=True)
    df_orders_Wait = df_orders[np.logical_and(df_orders['O_Status'] == 'Waiting', True)]
    df_orders_Wait.reset_index(inplace=True)
    df_orders_Wait.drop(columns=[df_orders_Wait.columns[0]], inplace=True)
    return df_orders_Wait, df_orders_notWait

def Get_TaskName(mainpath, taskname):
    i = 0
    ret = taskname
    s = os.path.join(mainpath, ret)
    while os.path.isdir(s):
        i += 1
        ret = taskname + '_' + '{0:03d}'.format(i)
        s = os.path.join(mainpath, ret)
    os.makedirs(s)
    return ret

def Create_TaskConfig(mainpath, taskname):
    s1 = {'Name': taskname, 'CreatedTime': dt.datetime.strftime(dt.datetime.now(),  '%Y-%m-%d %H:%M:%S'), 'Status': '', 'StatusNotes': ''}
    s2 = {'Preprocess': 'Yes', 'Manipulation': 'None', 'Lambda': 0, 'SplitQuantity': 0}
    s = {'Task': s1, 'Order': s2}
    with open(os.path.join(os.path.join(mainpath, taskname), 'TaskConfig.json'), 'w') as fn_json:
        json.dump(s, fn_json, ensure_ascii=False)
    return s

def OpenJsonFile(mainpath, taskname, jfile, DefaultTitle):
    s = os.path.join(os.path.join(mainpath, taskname), jfile)
    if os.path.isfile(s):
        fn_json = open(s, 'r')
        json_data = json.load(fn_json)
        fn_json.close()
        ret = 200
    else:
        json_data = {DefaultTitle: ''}
        ret = 404
    return json_data, ret

def DeleteTask(mainpath, taskname, dirname):
    ret = 200
    try:
        if dirname == '':
            spath = os.path.join(mainpath, taskname)
        else:
            spath = os.path.join(os.path.join(mainpath, taskname), dirname)
        if os.path.isdir(spath):
            #json_file, ret = {"Files", [dirname]}, 200 if dirname != '' else FileList(mainpath, taskname, 'StageLearning')
            if dirname == '':
                json_file, ret = FileList(mainpath, taskname, 'StageLearning')
                ret = 200
                lst_stage = json_file["Files"]
            else:
                lst_stage = [dirname]
            isrun = False
            for x in lst_stage:
                isrun = StageisProcessing(mainpath, taskname, x)
                if isrun: break
            if isrun:
                ret = 403
            else:
                shutil.rmtree(spath)
        else:
            ret = 404
    except:
        ret = 403
    return ret

def StageisProcessing(mainpath, taskname, dirname):
    spath = os.path.join(os.path.join(mainpath, taskname), dirname)
    sfile = os.path.join(spath, 'StageParameter.json')
    ret = False
    if os.path.isdir(spath) and os.path.isfile(sfile):
        fn = open(sfile, 'r')
        json_data = json.load(fn)
        fn.close()
        for x in json_data:
            if json_data[x]['Practice'] == 'Processing':
                ret = True
                break
    return ret

def Get_LastLearningNo(mainpath, taskname):
    lst = os.listdir(os.path.join(mainpath, taskname))
    i = 0
    for x in lst:
        if x[:13] == 'StageLearning':
            m = int(x[14:].replace(')', ''))
            if i < m: i = m
    return i

def NewStage(mainpath, taskname):
    n = Get_LastLearningNo(mainpath, taskname)
    sName = 'StageLearning(' + str(n+1) + ')'
    return sName

def savefile(mainpath, taskname, file, newfilename, attribvalue):
    filename = newfilename if newfilename != '' else file.filename
    if taskname == '':
        sfile = os.path.join(mainpath, filename)
    else:
        sfile = os.path.join(os.path.join(mainpath, taskname), filename)
    file.save(sfile)
    foldername = os.path.dirname(sfile)[len(mainpath) + 1:]
    if attribvalue == 'none':
        ret = 201
    else:
        ret = TaskConfig_SaveFileAttrib(mainpath, taskname, filename, attribvalue, foldername)
    return ret

def Download_EXCELFile(spath, sfile):
    result = send_file(os.path.join(spath, sfile), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", attachment_filename=sfile, conditional=False)
    result.headers['x-filename'] = sfile
    result.headers["Access-Control-Expose-Headers"] = 'x-filename'
    return result

def TaskConfig_SaveFileAttrib(mainpath, taskname, filename, attrib, foldername):
    fn = open(os.path.join(os.path.join(mainpath, taskname), 'TaskConfig.json'), 'r')
    json_data = json.load(fn)
    fn.close()
    lst_file = [] if 'Files' not in json_data else json_data['Files']
    if attrib != 'Calendar' and attrib != 'Setting' and attrib != 'Evaluation' and attrib != 'Dataset': attrib = ''
    ret = 201
    for x in lst_file:
        if x['Attribute'] == 'Calendar' and attrib == 'Calendar':
            x['Attribute'] = ''
            ret = 205
        elif x['Attribute'] == 'Setting' and attrib == 'Setting':
            x['Attribute'] = ''
            ret = 205
        elif x['Attribute'] == 'Evaluation' and attrib == 'Evaluation':
            x['Attribute'] = ''
            ret = 205
        elif x['Attribute'] == 'Dataset' and attrib == 'Dataset':
            x['Attribute'] = ''
            ret = 205
        else:
            continue
    fname = pathlib.Path(os.path.join(os.path.join(mainpath, taskname), filename))
    ttime = dt.datetime.strftime(dt.datetime.fromtimestamp(fname.stat().st_mtime),  '%Y-%m-%d %H:%M:%S')
    lst_file.append({'Attribute': attrib, 'FileName': filename, 'FolderName': foldername, 'ModifiedTime': ttime})
    json_data['Files'] = lst_file
    with open(os.path.join(os.path.join(mainpath, taskname), 'TaskConfig.json'), 'w') as fn_json:
        json.dump(json_data, fn_json, ensure_ascii=False)
    return ret

def EVRFile(mainpath, taskname):
    fn = open(os.path.join(os.path.join(mainpath, taskname), 'TaskConfig.json'), 'r')
    json_data = json.load(fn)
    fn.close()
    lst_file = [] if 'Files' not in json_data else json_data['Files']
    efile = ''
    for x in lst_file:
        if x['Attribute'] == 'Evaluation':
            efile = os.path.join(os.path.join(mainpath, taskname), x['FileName'])
            if not os.path.isfile(efile): efile = ''
        else:
            continue
    return efile

def SaveData_toDataset(mainpath, taskname, demand_start, demand_end, begin_day, end_day, df_orders, df_products, df_lines, df_molds, df_orderdata):
    dirname = os.path.join(os.path.join(mainpath, taskname), 'Dataset')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        TaskConfig_SaveFileAttrib(mainpath, taskname, '', 'Dataset', 'Dataset')
    df_orderdata.to_csv(os.path.join(dirname, 'orderdata.csv'), index=False)
    df_products.to_csv(os.path.join(dirname, 'products.csv'), index=False)
    df_lines.to_csv(os.path.join(dirname, 'lines.csv'), index=False)
    df_molds.to_csv(os.path.join(dirname, 'molds.csv'), index=False)
    df_orders.to_csv(os.path.join(dirname, 'orders.csv'), index=False)
    para_data = {'demand_start': str(demand_start), 'demand_end': str(demand_end), 'begin_day': begin_day, 'end_day': end_day}
    with open(os.path.join(dirname, 'parameter.json'), 'w') as fn_json:
        json.dump(para_data, fn_json, ensure_ascii=False)

def LoadData_FromDataset(mainpath, taskname):
    dirname = os.path.join(os.path.join(mainpath, taskname), 'Dataset')
    if os.path.isdir(dirname):
        df_orderdata = pd.read_csv(os.path.join(dirname, 'orderdata.csv'))
        df_products = pd.read_csv(os.path.join(dirname, 'products.csv'))
        df_lines = pd.read_csv(os.path.join(dirname, 'lines.csv'))
        df_molds = pd.read_csv(os.path.join(dirname, 'molds.csv'))
        df_orders = pd.read_csv(os.path.join(dirname, 'orders.csv'))
        df_orderdata.fillna('', inplace=True)
        df_products.fillna('', inplace=True)
        df_lines.fillna('', inplace=True)
        df_molds.fillna('', inplace=True)
        df_orders.fillna('', inplace=True)

        df_orders['not_before'] = df_orders['not_before'].astype('datetime64[ns]')
        df_orders['not_after'] = df_orders['not_after'].astype('datetime64[ns]')
        df_lines['Line Begin'] = df_lines['Line Begin'].astype('datetime64[ns]')

        f = open(os.path.join(dirname, 'parameter.json'), 'r')
        para_data = json.load(f)
        f.close()

        demand_start = dt.datetime.strptime(para_data['demand_start'], '%Y-%m-%d %H:%M:%S')
        demand_end = dt.datetime.strptime(para_data['demand_end'], '%Y-%m-%d %H:%M:%S')
        begin_day = para_data['begin_day']
        end_day = para_data['end_day']
        return 'OK', demand_start, demand_end, begin_day, end_day, df_orders, df_products, df_lines, df_molds, df_orderdata
    else:
        return 'files not exist.', '', '', '', '', pd.DataFrame({'id' : []}), pd.DataFrame({'id' : []}), pd.DataFrame({'id' : []}), pd.DataFrame({'id' : []}), pd.DataFrame({'id' : []})

# set orders to on_stock
def Orders_On_Stock(df_products, df_orders):
    lst_onstock_prodcode = []
    lst_onstock_qty = []
    df_tmp_prod = df_products[df_products['on_hand_stock'] != ''][['product_code', 'on_hand_stock', 'throughput']]
    df_tmp_prod.reset_index(inplace=True)
    for i in range(df_tmp_prod.shape[0]):
        prod_code = df_tmp_prod.loc[i, 'product_code']
        stock_qty = df_tmp_prod.loc[i, 'on_hand_stock']
        day_quantity = df_tmp_prod.loc[i, 'throughput']
        df_tmp_order = df_orders[df_orders['product_code'] == prod_code]
        for k in range(df_tmp_order.shape[0]):
            index = df_tmp_order.index[k]
            qty = df_tmp_order.loc[index, 'quantity']
            if stock_qty >= qty:
                stock_qty -= qty
                df_orders.loc[index, 'O_Status'] = 'On_Stock'
                df_orders.loc[index, 'Production_Hours'] = 0
            elif stock_qty > 0:
                df_orders.loc[df_orders.shape[0]] = df_orders.loc[index]
                df_orders.loc[df_orders.shape[0] - 1, 'quantity'] = qty - stock_qty
                df_orders.loc[df_orders.shape[0] - 1, 'Production_Hours'] = math.ceil((qty - stock_qty) * 48 / day_quantity) / 2
                df_orders.loc[index, 'O_Status'] = 'On_Stock'
                df_orders.loc[index, 'order_code'] = df_orders.loc[index, 'order_code'] + 'D'
                df_orders.loc[index, 'quantity'] = stock_qty
                df_orders.loc[index, 'Production_Hours'] = 0
                stock_qty = 0
            else:
               break
        if stock_qty > 0:
            lst_onstock_prodcode.append(prod_code)
            lst_onstock_qty.append(stock_qty)
    return lst_onstock_prodcode, lst_onstock_qty, df_orders

def Orders_Overdue(df_lines, df_orders):
    # set orders to overdue
    df_orders['Do_Lines Count'] = pd.DataFrame([len(x.split(';')) for x in df_orders['Do_Lines'].tolist()])  # count the length of do_lines to compare with the count in the condition
    df_tmp_order = df_orders[df_orders['O_Status'] == 'Waiting']
    df_tmp_order['count'] = 0

    # each line must satifies the condition, that is, the count must equal lst_dolines_count
    for i in range(df_lines.shape[0]):
        if df_lines.loc[i, 'Usable'] == 'YES':
            mask = np.logical_and(df_tmp_order['Do_Lines'].str.contains(df_lines.loc[i, 'line_name']), df_tmp_order['not_after'] - pd.to_timedelta(df_tmp_order['Production_Hours'], 'h') < df_lines.loc[i, 'Line Begin'])
            lst_index = df_tmp_order[mask].index.to_list()
            if len(lst_index) > 0: df_tmp_order.loc[lst_index, 'count'] = df_tmp_order.loc[lst_index, 'count'] + 1

    lst_index = df_tmp_order[df_tmp_order['count'] == df_tmp_order['Do_Lines Count']].index.to_list()
    if len(lst_index) > 0: df_orders.loc[lst_index, 'O_Status'] = 'Overdue'
    lst_index = df_tmp_order[df_tmp_order['not_after'] - df_tmp_order['not_before'] < pd.to_timedelta(df_tmp_order['Production_Hours'], 'h')].index.to_list()
    if len(lst_index) > 0: df_orders.loc[lst_index, 'O_Status'] = 'Overdue'
    df_orders.sort_values(by=['not_before', 'order_code'], inplace=True)
    df_orders.reset_index(inplace=True)
    df_orders.drop(columns=[df_orders.columns[0], 'Do_Lines Count'], inplace=True)
    return df_orders

def GetWorkHour(df_lines, lst_lines, maxline_count, i, m):
    lst_workhour = []
    for k in range(len(lst_lines) - m):
        d1 = df_lines.loc[df_lines[df_lines['line_name'] == lst_lines[k]].index[0], 'Line Begin']
        d2 = dt.datetime(d1.year, d1.month, d1.day, 0, 0, 0) + dt.timedelta(days=i+1)
        work_diff = gethours(d1, d2)
        if work_diff >= 24:
            work_diff = 24
        elif work_diff <= 0:
            work_diff = 0
        lst_workhour.append(work_diff)
    lst_workhour.sort(reverse = True)
    return sum(lst_workhour[:maxline_count])

def Production_Trendchart(end_day, df_lines, df_orders_Wait, maxline_count):
    lst_lines = sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())
    lst_lines.append('AVERAGE')
    lst_lines.append('Total Work Hours')
    lst_lines.append('Total Request Hours')

    n_lines = len(lst_lines)
    mindate = df_lines[df_lines['Usable'] == 'YES']['Line Begin'].min()
    n_days = getdays(dt.datetime.strftime(mindate, '%Y-%m-%d'), end_day)
    data_array = np.full((n_lines, n_days + 1), 0, dtype=float)

    for i in range(df_orders_Wait.shape[0]):
        lst_dolines = df_orders_Wait.loc[i, 'Do_Lines'].split(';')
        p_hour = df_orders_Wait.loc[i, 'Production_Hours'] / len(lst_dolines)
        n_a_o = df_orders_Wait.loc[i, 'not_after']
        for litem in lst_dolines:
            n_b_o = df_orders_Wait.loc[i, 'not_before']
            index = list_index(lst_lines, litem)
            if index >= 0:
                n_b_l = df_lines.loc[df_lines[df_lines['line_name'] == litem].index[0], 'Line Begin']
                if n_b_l > n_b_o: n_b_o = n_b_l
                ba_day = getdays(dt.datetime.strftime(n_b_o, '%Y-%m-%d'), dt.datetime.strftime(n_a_o, '%Y-%m-%d')) + 1
                x = getdays(dt.datetime.strftime(mindate, '%Y-%m-%d'), dt.datetime.strftime(n_b_o, '%Y-%m-%d'))
                for v in range(x, x + ba_day):
                    data_array[index, v] += p_hour / ba_day

    for i in range(data_array.shape[1]):
        sum = 0
        count = 0
        for k in range(data_array.shape[0] - 1):
            sum += data_array[k, i]
            if data_array[k, i] > 0: count += 1
        data_array[data_array.shape[0] - 3, i] = 0 if count == 0 else sum / count
        data_array[data_array.shape[0] - 2, i] = GetWorkHour(df_lines, lst_lines, maxline_count, i, 3)
        data_array[data_array.shape[0] - 1, i] = sum

    columns = ['Date']
    for i in range(data_array.shape[0]):
        columns.append(lst_lines[i])
        # columns.append(dt.datetime.strftime(mindate + dt.timedelta(days=i), '%m-%d'))
    df_trendchart = pd.DataFrame(columns=columns)

    for k in range(data_array.shape[1]):
        lst_tmpdata = [dt.datetime.strftime(mindate + dt.timedelta(days=k), '%m-%d')]
        for i in range(data_array.shape[0]):
            lst_tmpdata.append(data_array[i, k])
        df_trendchart.loc[k] = lst_tmpdata
    return df_trendchart

def Evaluation_Report(mainpath, taskname, demand_start, demand_end, end_day, maxline_count, lst_onstock_prodcode, lst_onstock_qty, df_orders, df_lines, df_molds):
    n = 1
    sfile = os.path.join(os.path.join(mainpath, taskname), taskname + '_EVR.xlsx')
    while os.path.isfile(sfile):
        n += 1
        sfile = os.path.join(os.path.dirname(sfile), taskname + '_EVR(' + str(n) + ').xlsx')

    ret = 205 if os.path.isfile(sfile) else 201
    df_summary, df_LCRR, df_orders_Wait, df_orders_notWait = DemandDiffSheet(demand_start, demand_end, end_day, lst_onstock_prodcode, lst_onstock_qty, df_orders, df_lines)
    df_trendchart = Production_Trendchart(end_day, df_lines, df_orders_Wait, maxline_count)

    writer = pd.ExcelWriter(sfile, engine='openpyxl')
    df_summary.to_excel(writer, sheet_name='需求訂單分析', startrow=0, startcol=0, index=False)
    df_orders_Wait.to_excel(writer, sheet_name='排程訂單組', startrow=0, startcol=0, index=False)

    df_moldtable = GetMoldTable(df_molds)
    df_moldtable.to_excel(writer, sheet_name='模頭編號清單', startrow=0, startcol=0, index=False)

    df_LCRR.to_excel(writer, sheet_name='產能需求差異表', startrow=0, startcol=0, index=False)
    df_trendchart.to_excel(writer, sheet_name='生產線時數需求', startrow=0, startcol=0, index=False)

    worksheet = writer.sheets['生產線時數需求']
    for i in range(2):
        values = Reference(worksheet, min_col=2 if i == 0 else df_trendchart.shape[1] - 1, min_row=1, max_col=df_trendchart.shape[1] - (2 if i == 0 else 0), max_row=df_trendchart.shape[0] + 1)
        chart = LineChart()
        chart.add_data(values, titles_from_data=True)
        dates_title = Reference(worksheet, min_col=1, min_row=2, max_row=df_trendchart.shape[0] + 1)
        chart.set_categories(dates_title)

        chart.title = ' 生產線時數需求趨勢圖 ' if i == 0 else ' 總需求與產能趨勢圖 '
        chart.x_axis.title = ' Date '
        chart.y_axis.title = ' Request Hours ' if i == 0 else 'Hours'
        chart.width = 17
        chart.height = 12
        worksheet.add_chart(chart, chr(66 + df_trendchart.shape[1]) + ('1' if i == 0 else '25'))
    writer.save()
    return sfile, ret

def GetMoldTable(df_molds):
    columns = ['模頭編號 (mold_coode)', '模頭號碼 (mold_no)', '板材寬度範圍最大值 (width_max)', '板材寬度範圍最小值 (width_min)', '板材厚度範圍最大值 (thickness_max)', '板材厚度範圍最小值 (thickness_min)', '目前用於 (Usage)']
    df_moldtable = pd.DataFrame(columns=columns)
    df_moldtable['模頭編號 (mold_coode)'] = df_molds['mold_code']
    df_moldtable['模頭號碼 (mold_no)'] = df_molds['mold_no']
    df_moldtable['板材寬度範圍最大值 (width_max)'] = df_molds['width_max']
    df_moldtable['板材寬度範圍最小值 (width_min)'] = df_molds['width_min']
    df_moldtable['板材厚度範圍最大值 (thickness_max)'] = df_molds['thickness_max']
    df_moldtable['板材厚度範圍最小值 (thickness_min)'] = df_molds['thickness_min']
    df_moldtable['目前用於 (Usage)'] = df_molds['Usage']
    return df_moldtable

def DemandDiffSheet(demand_start, demand_end, end_day, lst_onstock_prodcode, lst_onstock_qty, df_orders, df_lines):
    columns =['編製項目', '訂單數', '數量(片數)', '對應產品料號的數目', '備註']
    df_summary = pd.DataFrame(columns=columns)
    df_summary = df_summary.astype({'數量(片數)': int})

    #lst_onstock_prodcode, lst_onstock_qty, df_orders= Orders_On_Stock(df_products, df_orders)
    df_orders_Wait, df_orders_notWait = Get_orders_WaitandNotWait(df_orders)
    n_stock = df_orders_notWait[df_orders_notWait['O_Status'] == 'On_Stock'].shape[0]
    n_stock_part = df_orders_notWait[df_orders_notWait['order_code'].str[-1] == 'D'].shape[0]
    sum_stock = df_orders_notWait[df_orders_notWait['O_Status'] == 'On_Stock']['quantity'].sum()
    n_orders = df_orders_Wait.shape[0]
    tot_days = getdays(dt.datetime.strftime(demand_start, '%Y-%m-%d'), dt.datetime.strftime(demand_end, '%Y-%m-%d'))
    df_orders_Overdue = df_orders[df_orders['O_Status'] == 'Orverdue']

    df_summary.loc[0] = ['總需求', n_orders + n_stock - n_stock_part, df_orders['quantity'].sum(), len(df_orders['product_code'].unique()), '需求日期天數: ' + str(tot_days) + '天']
    df_summary.loc[1] = ['庫存可抵扣', n_stock, sum_stock, len(df_orders_notWait[df_orders_notWait['O_Status'] == 'On_Stock']['product_code'].unique()), '抵扣不全訂單數: ' + str(n_stock_part)]
    df_summary.loc[2] = ['庫存無法抵扣', 0, sum(lst_onstock_qty), len(lst_onstock_prodcode), '產品料號: ' + ','.join(lst_onstock_prodcode)]
    df_summary.loc[3] = ['逾期的需求', df_orders_Overdue.shape[0], df_orders_Overdue['quantity'].sum(), len(df_orders_Overdue['product_code'].unique()), '產品料號: ' + '無' if df_orders_Overdue.shape[0] == 0 else ','.join(df_orders_Overdue['product_code'])]
    df_summary.loc[4] = ['進入排程', n_orders, df_orders_Wait['quantity'].sum(), len(df_orders_Wait['product_code'].unique()), '包含抵扣不全的訂單數']
    df_summary = df_summary.astype({'數量(片數)': int})

    # LCRR
    grouped = df_orders_Wait.groupby('Do_Lines')
    lst_lines = []
    for name, group in grouped:
        lst_tmp = name.split(';')
        for lst_item in lst_tmp:
            if list_index(lst_lines, lst_item) < 0: lst_lines.append(lst_item)

    lst_lines = sortlist_bynum(lst_lines)
    lst_prod_hour = [0] * len(lst_lines)
    for name, group in grouped:
        lst_tmp = name.split(';')
        #print(name, group['Production_Hours'].sum(), len(lst_tmp))
        ret = group['Production_Hours'].sum() / len(lst_tmp)
        for lst_item in lst_tmp:
            p = list_index(lst_lines, lst_item)
            lst_prod_hour[p] += ret

    columns = ['Line', 'Line Begin', 'Request Production Hours', 'Distributed Work Hours', 'LCRR']
    df_LCRR = pd.DataFrame(columns=columns)
    for i in range(len(lst_lines)):
        line_name = lst_lines[i]
        index = df_lines[df_lines['line_name'] == line_name].index[0]
        work_hour = gethours(df_lines.loc[index, 'Line Begin'], dt.datetime.strptime(end_day, '%Y-%m-%d'))
        lc_ratio = '{0:0.2%}'.format(lst_prod_hour[i] / work_hour)
        df_LCRR.loc[i] = [line_name, df_lines.loc[index, 'Line Begin'], '{0:0.2f}'.format(lst_prod_hour[i]), work_hour, lc_ratio]
    return df_summary, df_LCRR, df_orders_Wait, df_orders_notWait

def Read_OriginData(calendarfile, settingfile):
    data = pd.read_excel(settingfile, skiprows=5, usecols='B:O', sheet_name=None)  # "data" are all sheets as a dictionary
    df_productdata = data.get('料號清單')
    df_productdata.fillna('', inplace=True)

    data = pd.read_excel(settingfile, skiprows=5, usecols='B:O', sheet_name=None)  # "data" are all sheets as a dictionary
    df_linedata = data.get('排程前的生產線參數')
    df_linedata.fillna('', inplace=True)

    data = pd.read_excel(settingfile, skiprows=5, usecols='B:G', sheet_name=None)  # "data" are all sheets as a dictionary
    df_molddata = data.get('模頭參數')
    df_molddata.fillna('', inplace=True)

    data = pd.read_excel(settingfile, skiprows=6, usecols='B:E', sheet_name=None)
    df_DemandData = data.get('需求排程期間')
    dm_start = dt.datetime.strptime(str(df_DemandData.loc[0, '需求起始日']), '%Y-%m-%d %H:%M:%S')
    dm_end = dt.datetime.strptime(str(df_DemandData.loc[0, '需求結束日']), '%Y-%m-%d %H:%M:%S')

    data = pd.read_excel(calendarfile, skiprows=0, sheet_name=None)  # "data" are all sheets as a dictionary
    df_orderdata = data.get('出貨需求(demand)')  # get a specific sheet to DataFrame
    df_orderdata.rename(columns={'Part No.': 'part_no'}, inplace=True)
    df_orderdata = order_dropcolumns(df_orderdata, dm_start, dm_end)
    #df_orderdata = order_droprows(df_orderdata)
    df_orderdata.fillna('', inplace=True)
    return df_productdata, df_linedata, df_molddata, df_orderdata, dm_start, dm_end

# products
def readproducts(df_stockdata, df_productdata):
    df_productdata.columns = ['part_no', 'width', 'length', 'height', 'type', 'density', 'material', '材料型號', 'composition', 'lenti_pitch', 'roller_position', 'throughput', 'assigned_lines', 'LT']
    df_products = df_productdata.sort_values('part_no').reset_index()

    # insert rows if stockdata duplicate
    for i in range(df_stockdata.shape[0]):
        #print('a1', df_stockdata.loc[i, 'Product_ID'])
        if df_stockdata.loc[i, 'Product_ID'] > 1:
            partno = df_stockdata.loc[i, 'part_no']
            #print('a2', partno)
            for k in range(df_products.shape[0]):
                if df_products.loc[k, 'part_no'] == partno:
                    lstdata = []
                    for x in range(df_products.shape[1]):
                        lstdata.append(df_products.iloc[k, x])
                    df_products = pd.DataFrame(np.insert(df_products.values, k, lstdata, axis=0))
                    break
    df_products.columns = ['index', 'part_no', 'width', 'length', 'height', 'type', 'density', 'material', '材料型號', 'composition', 'lenti_pitch', 'roller_position', 'throughput', 'assigned_lines', 'LT']

    # generate a column['product_code']
    df_products.insert(1, 'product_code', '', True)
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    for i in range(df_products.shape[0]):
        if df_products.loc[i, 'type'] == '結構板':
            if df_products.loc[i, 'material'] == 'PMMA':
                n1 += 1
                df_products.loc[i, 'product_code'] = 'J{0:03d}'.format(n1)
            else:
                n2 += 1
                df_products.loc[i, 'product_code'] = 'K{0:03d}'.format(n2)
        else:
            if df_products.loc[i, 'material'] == 'PMMA':
                n3 += 1
                df_products.loc[i, 'product_code'] = 'N{0:03d}'.format(n3)
            else:
                n4 += 1
                df_products.loc[i, 'product_code'] = 'O{0:03d}'.format(n4)
        composition = str(int(df_products.loc[i, 'composition'] * 100)) + '%'
        if df_products.loc[i, '材料型號'] == '':
            df_products.loc[i, 'composition'] = composition
        else:
            df_products.loc[i, 'composition'] = df_products.loc[i, '材料型號'] + '-' + composition
        df_products.loc[i, 'assigned_lines'] = df_products.loc[i, 'assigned_lines'].strip()

    df_products['type'] = np.where(df_products['type'] == '結構板', 'lenti', 'plate')
    df_products.drop(['index', '材料型號'], axis=1, inplace=True)
    return df_products

def readstock(df_orderdata, demand_start):
    s1 = 'part_no'
    s2 = 'MFG產出'
    onstock_day = str(demand_start + dt.timedelta(days=-1))[:10]

    columns = ['product_code', 'Product_ID', 'part_no', 'stock', 'on_hand_stock']
    df_stockdata = pd.DataFrame(columns=columns)
    for i in range(len(df_orderdata.columns)):
        sdata = str(df_orderdata.columns[i])
        if sdata == s1: df_stockdata['part_no'] = df_orderdata[df_orderdata.columns[i]]
        if sdata[:5] == s2: df_stockdata['stock'] = df_orderdata[df_orderdata.columns[i]]
        if sdata[:len(onstock_day)] == onstock_day: df_stockdata['on_hand_stock'] = df_orderdata[df_orderdata.columns[i]]
    df_stockdata['on_hand_stock'] = [0 if df_stockdata.loc[x, 'on_hand_stock'] == '' else (0 if df_stockdata.loc[x, 'on_hand_stock'] < 0 else df_stockdata.loc[x, 'on_hand_stock']) for x in range(df_stockdata.shape[0])]
    lst_partno = []
    for i in range(df_stockdata.shape[0]):
        if df_stockdata.iloc[i, 3] == 'On hand Stock':
            df_stockdata.iloc[i, 4] += df_stockdata.iloc[i + 1, 4]
            lst_partno.append(df_stockdata.loc[i, 'part_no'])
            df_stockdata.iloc[i, 1] = lst_partno.count(df_orderdata.loc[i, 'part_no'])

    mask = functools.reduce(np.logical_and, (df_stockdata['part_no'] != '', df_stockdata['stock'] == 'On hand Stock', df_stockdata['on_hand_stock'] >= 0))
    df_stockdata = df_stockdata[mask]
    #print('stock shape', df_stockdata.shape[0])

    df_stockdata.reset_index(inplace=True)
    df_stockdata.fillna('', inplace=True)
    df_stockdata.drop(columns=[df_stockdata.columns[0], df_stockdata.columns[4]], inplace=True)
    return df_stockdata

def updatestock(df_stockdata, df_products):
    for i in range(df_stockdata.shape[0]):
        prodid = df_stockdata.loc[i, 'Product_ID']
        partno = df_stockdata.loc[i, 'part_no']
        df_tmp = df_products[df_products['part_no'] == partno]
        if df_tmp.shape[0] > 0: df_stockdata.loc[i, 'product_code'] = df_products.loc[df_tmp.index[prodid - 1], 'product_code']
    df_stockdata.drop(columns=['Product_ID'], inplace=True)
    return df_stockdata