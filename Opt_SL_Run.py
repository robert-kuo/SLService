import Opt_SL_Func
import Opt_func
import Opt_RS_Report
import json, copy
import os, math, random, functools
import datetime as dt
import pandas as pd
import numpy as np

def Data_Append(lst_theloss, jtime, episode, learn_weight, orders_learning_ref):
    for i in range(len(lst_theloss)):
        scode = lst_theloss[i]
        m = -1
        for k in range(len(orders_learning_ref)):
            if scode == orders_learning_ref[k][0]:
                orders_learning_ref[k] = (scode, orders_learning_ref[k][1] + ',' + str(episode), jtime, orders_learning_ref[k][3] + learn_weight)
                m = k
                break
        if m == -1: orders_learning_ref.append((scode, str(episode), jtime, learn_weight))
    return orders_learning_ref

def Show_learning_ref(ln_ref_file, Ln_No, orders_learning_ref):
    if ln_ref_file != '':
        fn = open(ln_ref_file, 'a')
        fn.write('Learning No. ' + str(Ln_No) + '\n')
        fn.write('    Learning Reference:\n')
        for i in range(len(orders_learning_ref)):
            fn.write('      ' + str(orders_learning_ref[i]) + '\n')
        fn.close()
    print('    Learning Reference:')
    for i in range(len(orders_learning_ref)):
        print('      ' + str(orders_learning_ref[i]))

def read_learning_ref(sfile):
    orders_learning_ref = []
    if sfile != '':
        fn = open(sfile)
        data = fn.readlines()
        fn.close()
        for i in range(len(data)):
            if data[i].strip() != '':
                lst_ref = data[i].replace('\n', '').split(',')
                orders_learning_ref.append((lst_ref[0], str(lst_ref[1]).replace(';', ','), dt.datetime.strptime(lst_ref[2], '%Y-%m-%d %H:%M:%S'), int(lst_ref[3])))
    return orders_learning_ref

def GetExtraDay(lostfile, TransHour, begin_day, demand_start, json_data):
    df_order_result = pd.read_csv(lostfile)
    df_LostOrder = df_order_result[df_order_result['O_Status'] == 'LOST']
    df_LostOrder.reset_index(inplace=True)
    df_LostOrder.drop(columns=[df_LostOrder.columns[0], df_LostOrder.columns[1]], inplace=True)
    df_LostOrder['Lind-Add Days'] = [math.ceil(df_LostOrder.loc[x, 'Production_Hours'] / TransHour) for x in range(df_LostOrder.shape[0])]
    df_LostOrder['Line-Add Begin'] = [str(dt.datetime.strptime(df_LostOrder.loc[x, 'not_after'], '%Y-%m-%d %H:%M:%S') - dt.timedelta(days=int(df_LostOrder.loc[x, 'Lind-Add Days'])))[:10] for x in range(df_LostOrder.shape[0])]
    df_LostOrder['Line-Add End'] = [str(dt.datetime.strptime(df_LostOrder.loc[x, 'not_after'], '%Y-%m-%d %H:%M:%S') - dt.timedelta(days=1))[:10] for x in range(df_LostOrder.shape[0])]

    lst_OOD = json_data['Trial Stage']['OOD']
    print('OOD', lst_OOD)
    n_day = Opt_func.getdays(begin_day, str(demand_start)[:10])
    for i in range(df_LostOrder.shape[0]):
        t1 = df_LostOrder.loc[i, 'Line-Add Begin']
        t2 = df_LostOrder.loc[i, 'Line-Add End']
        x1 = Opt_func.getdays(begin_day, t1)
        x2 = Opt_func.getdays(begin_day, t2)
        for k in range(x1, x2 + 1):
            the_date = str(dt.datetime.strptime(begin_day, '%Y-%m-%d') + dt.timedelta(days=k))[:10]
            d_step = -1
            while Opt_func.list_index(lst_OOD, the_date) >= 0:
                the_date = dt.datetime.strftime(dt.datetime.strptime(the_date, '%Y-%m-%d') + dt.timedelta(days=d_step), '%Y-%m-%d')
                if Opt_func.getdays(begin_day, the_date) < n_day: d_step = 1
            lst_OOD.append(the_date)
    return lst_OOD

def Calc_OOD(ordercsv_file, ths_index, maxline_count, begin_day, end_day, json_data, df_lines):
    df_order_result = pd.read_csv(ordercsv_file)
    df_order_lost = df_order_result[df_order_result['O_Status'] == 'LOST']
    df_order_lost.reset_index(inplace=True)
    df_order_lost.drop(columns=[df_order_lost.columns[0], df_order_lost.columns[1]], axis=1, inplace=True)

    df_lossPH = Opt_func.OrderLost_Trendchart(df_order_lost, df_lines, end_day)
    ncol = df_lossPH.shape[1]

    m_linearray = [maxline_count] * (Opt_func.getdays(begin_day, end_day) + 1)
    ths = json_data['Trial Stage']['OOD GM']['transfer hours'][ths_index]
    print('OOD', ths_index, ths)
    itot1 = 0
    itot2 = 9
    i1 = 0
    for i in range(df_lossPH.shape[0]):
        for k in range(1, ncol - 1):
            itot2 += df_lossPH.iloc[i, k]
        df_lossPH.loc[i, 'Accumulated'] = itot2
        i2 = int(itot2 / ths)
        if i1 != i2 or itot1 == 0 and itot2 > 0:
            df_lossPH.loc[i, str(ths)] = '@'
            n_date = begin_day[:5] + df_lossPH.loc[i, 'Date']
            m_linearray[Opt_func.getdays(begin_day, n_date)] += 1
        else:
            df_lossPH.loc[i, str(ths)] = ''
        i1 = i2
        itot1 = itot2

    lst_OOD = json_data['Trial Stage']['OOD']
    n_day = Opt_func.getdays(begin_day, begin_day[:5] + df_lossPH.loc[0, 'Date'])
    for i in range(n_day, len(m_linearray)):
        if m_linearray[i] > maxline_count:
            the_date = begin_day[:5] + df_lossPH.loc[i - n_day, 'Date']
            d_step = -1
            while Opt_func.list_index(lst_OOD, the_date) >=0:
                the_date = dt.datetime.strftime(dt.datetime.strptime(the_date, '%Y-%m-%d') + dt.timedelta(days=d_step), '%Y-%m-%d')
                if Opt_func.getdays(begin_day, the_date) < n_day: d_step = 1
            lst_OOD.append(the_date)
    m_linearray = [maxline_count] * (Opt_func.getdays(begin_day, end_day) + 1)
    for i in range(len(lst_OOD)):
        m_linearray[Opt_func.getdays(begin_day, lst_OOD[i])] += 1
    return m_linearray, lst_OOD

def Calculate_waste(o_type, linename, product_code, mfg_width, order_width, length, height, density, qty, composition, df_products, df_lines):
    p = df_products[df_products['product_code'] == product_code].index[0]
    lentipitch = df_products.loc[p, 'lenti_pitch']
    rollerposition = df_products.loc[p, 'roller_position']
    v = df_lines[df_lines['line_name'] == linename].index[0]
    df_lines.loc[v, 'mfg_width'] = mfg_width
    df_lines.loc[v, 'width'] = order_width
    df_lines.loc[v, 'thickness'] = height
    df_lines.loc[v, 'composition'] = composition
    df_lines.loc[v, 'type'] = o_type
    if o_type == 'lenti':
        df_lines.loc[v, 'roller_position'] = rollerposition
        df_lines.loc[v, 'lenti_pitch'] = lentipitch
    else:
        df_lines.loc[v, 'roller_position'] = '-'
        df_lines.loc[v, 'lenti_pitch'] = '-'
    return ((mfg_width - order_width) * length * height * density * qty) / 1000000

def initial_jtp(df_lines):
    columns = ['line_name', 'date', 'Status', 'id', 'sub id']
    df_jtp = pd.DataFrame(columns=columns)
    for i in range(df_lines.shape[0]):
        if df_lines.loc[i, 'Usable'] == 'YES':
            linename = df_lines.loc[i, 'line_name']
            df_jtp.loc[df_jtp.shape[0]] = [linename, df_lines.loc[i, 'Line Begin'], '', 0, 0]
    df_jtp['date'] = df_jtp['date'].astype('datetime64[ns]')
    return df_jtp

def initial_block(n, df_lines, lst_lines):
    lst_block = []
    for i in range(len(lst_lines)):
        lstsub = []
        for k in range(n):
            lstsub.append([])
        lst_block.append(lstsub)

    for i in range(len(lst_lines)):
        df_tmp = df_lines[df_lines['line_name'] == lst_lines[i]]
        id = df_tmp.index[0]
        sdate1 = str(df_tmp.loc[id, 'Line Begin'] - dt.timedelta(hours=24))
        lst_block[Opt_func.GetLineID_sorted(lst_lines[i], df_lines)][0] = set_blockdata(df_tmp.loc[id, 'Line_Status'], sdate1, 24, '', '', 0, 0, [''] * 7, [''] * 6)
    return lst_block

def set_blockdata(blocktype, begintime, duration, ordercode, productcode, waste, quantity, lst_turn, lst_dispstep):
    lst = [blocktype, begintime, duration, ordercode, productcode, waste, quantity]
    lst.extend(lst_turn)
    lst.extend(lst_dispstep)
    return lst

def get_jtp_linelist(i_period, jtp_now, df_jtp):
    df_tmp = df_jtp[functools.reduce(np.logical_and, (df_jtp['Status'] == '', df_jtp['date'] >= jtp_now, df_jtp['date'] <= jtp_now + dt.timedelta(hours=i_period - 1) + dt.timedelta(minutes=59) + dt.timedelta(seconds=59)))]
    df_tmp.sort_values(by='date', inplace=True)
    grouped = df_tmp.groupby('date')
    lst_jtp_linename = []
    lst_jtp_date = []
    for name, group in grouped:
        lstname = group['line_name'].to_list()
        lstdate = group['date'].to_list()
        for i in random.sample(range(len(lstname)), len(lstname)):
            lst_jtp_linename.append(lstname[i])
            lst_jtp_date.append(lstdate[i])
    return lst_jtp_linename, lst_jtp_date

def get_jtptime(begin_day, end_day, i_period, df_jtp):
    if df_jtp.shape[0] == 0:
       jtp_min = dt.datetime.strptime(begin_day, '%Y-%m-%d')
    else:
       jtp_min = df_jtp[df_jtp['Status'] == '']['date'].min()
    jtp_max = jtp_min + dt.timedelta(hours=i_period-1) + dt.timedelta(minutes=59) + dt.timedelta(seconds=59)
    hour_endday = Opt_func.gethours(jtp_min, dt.datetime.strptime(end_day, '%Y-%m-%d'))
    return jtp_min, jtp_max, hour_endday

def combine_dataframe(df_random, df_sort):
    df_combine = pd.DataFrame(columns=df_random.columns)
    n = 0
    for v in range(df_random.shape[0]):  #random.sample(range(df_random.shape[0]), df_random.shape[0]):
        df_combine.loc[n] = df_random.loc[df_random.index[v]]
        n += 1
    df_sort = df_sort.sort_values(by='waste_hour').reset_index(drop=True)
    for v in range(df_sort.shape[0]):
        df_combine.loc[n] = df_sort.loc[v]
        n += 1
    df_combine.reset_index(drop=True, inplace=True)
    return df_combine

def Get_ProductionCount(All_Lines, LineID, Line_time, lst_block):
    n = 0
    for k in range(len(All_Lines)):
        if k != LineID:
            m = Opt_func.Get_blockcount(k, lst_block)
            for x in range(m):
                btime = dt.datetime.strptime(lst_block[k][x][1], '%Y-%m-%d %H:%M:%S')
                etime = btime + dt.timedelta(hours=lst_block[k][x][2])
                blocktype = lst_block[k][x][0]
                if btime <= Line_time < etime:
                    if blocktype == 'Production' or blocktype == 'Tunning-Production' or blocktype == 'Tunning & Bootup' or blocktype == 'Shutdown':
                        if blocktype != 'Shutdown':
                            n += 1
                        elif lst_block[k][x + 1][0] == 'Tunning & Bootup':
                            n += 1
                        break
    return n

def GetMinJtpTime_notme(All_Lines, LineID, lst_block):
    m = Opt_func.Get_blockcount(LineID, lst_block)
    otime = dt.datetime.strptime(lst_block[LineID][m - 1][1], '%Y-%m-%d %H:%M:%S') + dt.timedelta(hours=lst_block[LineID][m - 1][2])
    rtime = otime
    t_min = 10000
    for k in range(len(All_Lines)):
        if k != LineID:
            m = Opt_func.Get_blockcount(k, lst_block)
            btime = dt.datetime.strptime(lst_block[k][m - 1][1], '%Y-%m-%d %H:%M:%S') + dt.timedelta(hours=lst_block[k][m - 1][2])
            if btime > otime:
                t_diff = Opt_func.gethours(otime, btime)
                if t_diff < t_min:
                    t_min = t_diff
                    rtime = btime
    return rtime

def get_lossorders(jtp_now, df_orders):
    ret = df_orders[np.logical_and(df_orders['not_after'] - pd.to_timedelta(df_orders['Production_Hours'] - 24, 'h') < jtp_now, df_orders['O_Status'] == 'Waiting')].order_code.to_list()
    return ret

def get_orders_ph(df_orders):
    ret = df_orders[df_orders['O_Status'] == 'Done'].Production_Hours.sum()
    return ret

def get_allloss(df_orders):
    ret = df_orders[np.logical_or(df_orders['O_Status'] == 'Waiting', df_orders['O_Status'] == 'LOST')].order_code.to_list()
    return ret


# =====  Set Blocks  =====
def setblock_instantorder(index, block_id, lst_block, line_id, lst_lines, jtp_now, lst_step, lst_stname, lst_parameter, lst_hour, orders_learning_ref, df_products,df_lines, df_order_instant, episode):
    i = -1
    if len(orders_learning_ref) > 0 and df_order_instant.shape[0] > 1:
        min_order = 1000
        for v in range(len(orders_learning_ref)):
            if df_order_instant[df_order_instant['order_code'] == orders_learning_ref[v][0]].shape[0] > 0:
                m = df_order_instant[df_order_instant['order_code'] == orders_learning_ref[v][0]].index[0]
                if m - orders_learning_ref[v][3] < min_order:
                    min_order = m - orders_learning_ref[v][3]
                    i = m
        Priority_result = 'Instant orders: ' + str(df_order_instant['order_code'].to_list()) + '\n'
        if min_order > 0:
            i = -1
            if episode <= 2: Priority_result += 'Priority Score: no order over top order.'
        else:
            Priority_result += 'Priority Score: find order over top order. ' + df_order_instant.loc[i, 'order_code']
    else:
        Priority_result = 'Instant orders: ' + str(df_order_instant['order_code'].to_list()) + '\nPriority Score: None.'

    dp_id = df_order_instant.index[0 if i <= 0 else i]
    disptype = df_order_instant.loc[dp_id, 'dispatch_type']
    lst_dispstep = df_order_instant.loc[dp_id, 'dispatch_step'].split(',')
    # print('step', lst_dispstep)

    duration = df_order_instant.loc[dp_id, 'Production_Hours']

    df_tmp = df_lines[df_lines['line_name'] == lst_lines[index]]
    l_width = df_tmp.loc[df_tmp.index[0], 'width']
    l_composition = df_tmp.loc[df_tmp.index[0], 'composition']
    l_thickness = df_tmp.loc[df_tmp.index[0], 'thickness']
    mfg_width = df_tmp.loc[df_tmp.index[0], 'mfg_width']

    waste = Calculate_waste(df_order_instant.loc[dp_id, 'type'], lst_lines[index], df_order_instant.loc[dp_id, 'product_code'], mfg_width, df_order_instant.loc[dp_id, 'width'], df_order_instant.loc[dp_id, 'length'],
                            df_order_instant.loc[dp_id, 'height'], df_order_instant.loc[dp_id, 'density'], df_order_instant.loc[dp_id, 'quantity'], df_order_instant.loc[dp_id, 'composition'], df_products, df_lines)
    tune_w = 0
    if disptype == 'IT':
        df_order_instant.loc[dp_id, 'tune_step'], df_order_instant.loc[dp_id, 'tune_x'], df_order_instant.loc[dp_id, 'tune_w'] = Opt_SL_Func.TIP_result(df_order_instant.loc[dp_id, 'dispatch_type'], df_order_instant.loc[dp_id, 'dispatch_step'], df_order_instant.loc[dp_id, 'type'],
                                                                                                                                                       df_order_instant.loc[dp_id, 'width'], df_order_instant.loc[dp_id, 'height'], df_order_instant.loc[dp_id, 'composition'], l_width, l_thickness, l_composition, lst_step, lst_hour)
        tune_x = float(df_order_instant.loc[dp_id, 'tune_x'])  # df_order_instant.loc[dp_id, 'tune_x'].astype(float)
        tune_w = df_order_instant.loc[dp_id, 'tune_w']
        duration += tune_x
        for v in range(len(lst_dispstep)):
            if Opt_func.list_index(lst_step, lst_dispstep[v][:3]) >= 0:
                lst_dispstep[v] += lst_stname[Opt_func.list_index(lst_step, lst_dispstep[v][:3])] + ',' + Opt_SL_Func.Get_Paradata(df_order_instant, dp_id, 0, '', lst_parameter[Opt_func.list_index(lst_step, lst_dispstep[v])].replace(';', ','), df_products)
        if len(lst_dispstep) < 6: lst_dispstep.extend([''] * (6 - len(lst_dispstep)))
        lst_block[line_id][block_id] = set_blockdata('Tunning-Production', str(jtp_now), duration, df_order_instant.loc[dp_id, 'order_code'], df_order_instant.loc[dp_id, 'product_code'],
                                                     waste + tune_w, df_order_instant.loc[dp_id, 'quantity'], ['調機時間=' + str(tune_x) + '小時', '估計廢料量=' + str(tune_w) + 'kg', '', '', '', '', ''], lst_dispstep)
    else:  # 'DP'
        lst_block[line_id][block_id] = set_blockdata('Production', str(jtp_now), duration, df_order_instant.loc[dp_id, 'order_code'], df_order_instant.loc[dp_id, 'product_code'],
                                                     waste, df_order_instant.loc[dp_id, 'quantity'], [''] * 7, [''] * 6)
    if jtp_now + dt.timedelta(hours=duration) >= df_order_instant.loc[dp_id, 'not_after']: print('Overtime Order code. ' + df_order_instant.loc[dp_id, 'order_code'] + ' ' + disptype)
    return dp_id, duration, waste + tune_w, Priority_result

def setblock_neworder(show, index, block_id, lst_block, line_id, lst_lines, jtp_id, jtp_now, learn_weight, lst_step, lst_stname, lst_parameter, lst_hour, df_products, df_lines, df_molds, orders_learning_ref, df_order_ns):
    # new orders weights
    lst_neworders = [*range(df_order_ns.shape[0])]
    lst_weights = df_order_ns.shape[0] * [learn_weight]
    for v in range(len(orders_learning_ref)):
        if df_order_ns[df_order_ns['order_code'] == orders_learning_ref[v][0]].shape[0] > 0:
            m = df_order_ns[df_order_ns['order_code'] == orders_learning_ref[v][0]].index[0]
            lst_weights[m] += orders_learning_ref[v][3]

    # ns_ok
    # -1: no new order to add
    # 0: a new order to add
    # 1 to: searching new order
    ns_ok = 1
    rand_id = -1
    ns_id = 0
    dura_prod = 0
    dura_tune = 0
    dura_shutdown = 0
    waste_tune = 0
    waste_prod = 0
    Weights_Result = 0
    while ns_ok > 0:
        if rand_id >= 0: lst_weights[rand_id] = 0
        if lst_weights.count(0) < len(lst_weights):
            rand_id = random.choices(lst_neworders, lst_weights)[0]
            ns_id = df_order_ns.index[rand_id]

            # get most early not_after order
            pd_code = df_order_ns.loc[ns_id, 'product_code']
            df_order_tmp = df_order_ns[df_order_ns['product_code'] == pd_code]
            for r1 in range(df_order_tmp.shape[0]):
                rx_id = -1
                for r2 in range(df_order_ns.shape[0]):
                    if df_order_tmp.index[r1] == df_order_ns.index[r2] and lst_weights[r2] != 0:
                        rx_id = r2
                        break
                if rx_id >= 0:
                    rand_id = rx_id
                    ns_id = df_order_ns.index[rand_id]
                    break
            # print('Early not_after', od_code, df_order_ns.loc[ns_id, 'order_code'])

            Weights_Result = 'Weight Modification:' + str(df_order_ns['order_code'].to_list()) + ' ' + str(lst_weights) + '\n'
            Weights_Result += 'Choose order:' + df_order_ns['order_code'].to_list()[ns_id] + ' (' + str(lst_weights[ns_id]) + ')'
            if Opt_func.list_index(lst_weights, 2) >= 0 or Opt_func.list_index(lst_weights, 3) >= 0:
                print('    JTP Cycle:', jtp_id, ', JTP time', jtp_now)
                print('    Weight Modification', df_order_ns['order_code'].to_list(), lst_weights)
                print('    Choose order:', df_order_ns['order_code'].to_list()[ns_id] + ' (' + str(lst_weights[ns_id]) + ')')

            df_tmp = df_lines[df_lines['line_name'] == lst_lines[index]]
            l_width = df_tmp.loc[df_tmp.index[0], 'width']
            l_composition = df_tmp.loc[df_tmp.index[0], 'composition']
            l_thickness = df_tmp.loc[df_tmp.index[0], 'thickness']
            # mfg_width = df_tmp.loc[df_tmp.index[0], 'mfg_width']

            df_order_ns.loc[ns_id, 'tune_step'], df_order_ns.loc[ns_id, 'tune_x'], df_order_ns.loc[ns_id, 'tune_w'] = Opt_SL_Func.TIP_result(df_order_ns.loc[ns_id, 'dispatch_type'],  df_order_ns.loc[ns_id, 'dispatch_step'], df_order_ns.loc[ns_id, 'type'],
                                                                                                                                            df_order_ns.loc[ns_id, 'width'], df_order_ns.loc[ns_id, 'height'], df_order_ns.loc[ns_id, 'composition'], l_width, l_thickness, l_composition, lst_step, lst_hour)
            disptype = df_order_ns.loc[ns_id, 'dispatch_type']
            lst_dispstep = df_order_ns.loc[ns_id, 'dispatch_step'].split(',')
            lst_tunestep = df_order_ns.loc[ns_id, 'tune_step'].split(',')
            if show: print(ns_id, disptype, lst_dispstep, lst_tunestep)
            wait_Y = Opt_func.list_index(lst_tunestep, '5.0')
            waste_tune = df_order_ns.loc[ns_id, 'tune_w']
            dura_tune = df_order_ns.loc[ns_id, 'tune_x'].astype(float)
            # mfg_width = df_lines.loc[df_lines[df_lines['line_name'] == lst_lines[index]].index[0], 'mfg_width']

            dura_shutdown = 0
            Y_hour = 0
            if wait_Y >= 0:  # wait Y hours
                t1 = jtp_now + dt.timedelta(hours=dura_tune - 3)
                t2 = dt.datetime(t1.year, t1.month, t1.day, 11, 0, 0) + dt.timedelta(days=0 if t1.hour <= 8 else 1)
                Y_hour = dura_tune
                dura_tune = Opt_func.gethours(jtp_now, t2)
                Y_hour = dura_tune - Y_hour
                if block_id > 0 and lst_block[line_id][block_id - 1][0] != 'No Bootup': Y_hour -= 0.5

            # add tunning & bootup
            new_moldcode = ''
            for v in range(len(lst_dispstep)):
                if Opt_func.list_index(lst_step, lst_dispstep[v]) >= 0 and lst_parameter[Opt_func.list_index(lst_step, lst_dispstep[v])].find('=%3') >= 0:  # change mold
                    lst_mold_canuse = df_molds[df_molds['Usage'] == ''].mold_code.to_list()
                    new_moldcode = random.choice(list(set(lst_mold_canuse) & set(df_order_ns.loc[ns_id, 'Do_Molds'].split(';'))))
                if Opt_func.list_index(lst_step, lst_dispstep[v][:3]) >= 0:
                    lst_dispstep[v] += lst_stname[Opt_func.list_index(lst_step, lst_dispstep[v][:3])] + ',' + Opt_SL_Func.Get_Paradata(df_order_ns, ns_id, Y_hour, new_moldcode, lst_parameter[Opt_func.list_index(lst_step, lst_dispstep[v])].replace(';', ','), df_products)
            if len(lst_dispstep) < 6: lst_dispstep.extend([''] * (6 - len(lst_dispstep)))

            for v in range(len(lst_tunestep)):
                if Opt_func.list_index(lst_step, lst_tunestep[v]) >= 0:
                    lst_tunestep[v] += lst_stname[Opt_func.list_index(lst_step, lst_tunestep[v])] + ',' + Opt_SL_Func.Get_Paradata(df_order_ns, ns_id, Y_hour, new_moldcode, lst_parameter[Opt_func.list_index(lst_step, lst_tunestep[v])], df_products)  # .replace(';', ',').replace('=%7', str(Y_hour)))
            if len(lst_tunestep) < 7: lst_tunestep.extend([''] * (7 - len(lst_tunestep)))

            lst_tmpblock = []
            startid = 0
            if wait_Y >= 0:  # wait Y hours
                if block_id > 0:
                    if lst_block[line_id][block_id - 1][0] == 'No Bootup':
                        jtp_pre = dt.datetime.strptime(lst_block[line_id][block_id - 1][1], '%Y-%m-%d %H:%M:%S')
                        dura_pre = lst_block[line_id][block_id - 1][2]
                        dura_pday = int(dura_pre / 24)
                        dura_nday = int(dura_tune / 24)
                        dura_day = min(dura_pday, dura_nday)
                        startid = -1
                        if dura_pre == dura_day * 24:
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_pre), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                            dura_tune -= dura_pre
                        else:
                            lst_tmpblock.append(['No Bootup', str(jtp_pre), dura_pre - dura_day * 24, '', '', '', '', [''] * 7, [''] * 6])
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_now - dt.timedelta(days=dura_day)), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                            dura_tune -= dura_day * 24
                    else:
                        if lst_block[line_id][block_id - 1][0] == 'Shutdown':
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_now), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                        else:
                            dura_tune -= 0.5
                            dura_shutdown = 0.5
                            lst_tmpblock.append(['Shutdown', str(jtp_now), dura_shutdown, '', '', 150, '', [''] * 7, [''] * 6])
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_now + dt.timedelta(hours=dura_shutdown)), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                            waste_tune += 150
                else:
                    lst_tmpblock.append(['Tunning & Bootup', str(jtp_now), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
            else:
                if block_id > 0:
                    if lst_block[line_id][block_id - 1][0] == 'No Bootup':
                        jtp_pre = dt.datetime.strptime(lst_block[line_id][block_id - 1][1], '%Y-%m-%d %H:%M:%S')
                        dura_pre = lst_block[line_id][block_id - 1][2]
                        startid = -1
                        if dura_tune >= dura_pre:  # kill NB, add TB
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_pre), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                            dura_tune -= dura_pre
                        else:  # small NB
                            lst_tmpblock.append(['No Bootup', str(jtp_pre), dura_pre - dura_tune, '', '', '', '', [''] * 7, [''] * 6])
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_pre + dt.timedelta(hours=dura_pre - dura_tune)), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                            dura_tune = 0
                    else:
                        if lst_block[line_id][block_id - 1][0] == 'Shutdown':
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_now), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                        else:
                            dura_shutdown = 0.5
                            lst_tmpblock.append(['Shutdown', str(jtp_now), dura_shutdown, '', '', 150, '', [''] * 7, [''] * 6])
                            lst_tmpblock.append(['Tunning & Bootup', str(jtp_now + dt.timedelta(hours=dura_shutdown)), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])
                            waste_tune += 150
                else:
                    lst_tmpblock.append(['Tunning & Bootup', str(jtp_now), dura_tune, '', '', waste_tune, '', lst_tunestep, lst_dispstep])

            dura_prod = df_order_ns.loc[ns_id, 'Production_Hours']
            mfg_width = (1000 if df_order_ns.loc[ns_id, 'width'] < 1000 else df_order_ns.loc[ns_id, 'width']) + (70 if df_order_ns.loc[ns_id, 'type'] == 'lenti' else 50)
            waste_prod = Calculate_waste(df_order_ns.loc[ns_id, 'type'], lst_lines[index], df_order_ns.loc[ns_id, 'product_code'], mfg_width, df_order_ns.loc[ns_id, 'width'], df_order_ns.loc[ns_id, 'length'],
                                         df_order_ns.loc[ns_id, 'height'], df_order_ns.loc[ns_id, 'density'], df_order_ns.loc[ns_id, 'quantity'], df_order_ns.loc[ns_id, 'composition'], df_products, df_lines)

            if jtp_now + dt.timedelta(hours=dura_shutdown + dura_tune + dura_prod) >= df_order_ns.loc[ns_id, 'not_after']:
                print('Overtime Order code. ' + df_order_ns.loc[ns_id, 'order_code'] + ' new ' + str(ns_ok) + ' ' + str(lst_weights) + ' ' + str(rand_id))
                ns_ok += 1
            else:
                if new_moldcode != '': df_lines, df_molds = Opt_SL_Func.changemold(lst_lines[index], new_moldcode, df_lines, df_molds)
                for v in range(len(lst_tmpblock)):
                    lst_block[line_id][block_id + v + startid] = set_blockdata(lst_tmpblock[v][0], lst_tmpblock[v][1], lst_tmpblock[v][2], lst_tmpblock[v][3], lst_tmpblock[v][4], lst_tmpblock[v][5], lst_tmpblock[v][6], lst_tmpblock[v][7], lst_tmpblock[v][8])
                lst_block[line_id][Opt_func.Get_blockcount(line_id, lst_block)] = set_blockdata('Production', str(jtp_now + dt.timedelta(hours=dura_shutdown + dura_tune)), dura_prod, df_order_ns.loc[ ns_id, 'order_code'], df_order_ns.loc[ns_id, 'product_code'], waste_prod, df_order_ns.loc[ns_id, 'quantity'], [''] * 7, [''] * 6)
                ns_ok = 0
        else:
            ns_ok = -1
    return ns_id, dura_prod, dura_tune, dura_shutdown, waste_tune + waste_prod, Weights_Result, ns_ok

def setblock_noorder(jtp_next, block_id, lst_block, line_id, jtp_now):
    waste_nb = 0
    if block_id > 0:
        if lst_block[line_id][block_id - 1][0] == 'No Bootup':  # extend NB
            jtp_pre = dt.datetime.strptime(lst_block[line_id][block_id - 1][1], '%Y-%m-%d %H:%M:%S')
            lst_block[line_id][block_id - 1] = set_blockdata('No Bootup', str(jtp_pre), Opt_func.gethours(jtp_pre, jtp_next), '', '', '', '', [''] * 7, [''] * 6)
        else:
            if lst_block[line_id][block_id - 1][0] == 'Shutdown':
                lst_block[line_id][block_id] = set_blockdata('No Bootup', str(jtp_now), Opt_func.gethours(jtp_now, jtp_next), '', '', '', '', [''] * 7, [''] * 6)
            else:
                lst_block[line_id][block_id] = set_blockdata('Shutdown', str(jtp_now), 0.5, '', '', 150, '', [''] * 7, [''] * 6)
                lst_block[line_id][block_id + 1] = set_blockdata('No Bootup', str(jtp_now + dt.timedelta(hours=0.5)), Opt_func.gethours(jtp_now, jtp_next) - 0.5, '', '', '', '', [''] * 7, [''] * 6)
                waste_nb = 150
    else:
        lst_block[line_id][block_id] = set_blockdata('No Bootup', str(jtp_now), Opt_func.gethours(jtp_now, jtp_next), '', '', '', '', [''] * 7, [''] * 6)
    return waste_nb

def save_jsonfile(jfile, jdata):
    with open(jfile, 'w') as fn_json:
        json.dump(jdata, fn_json, ensure_ascii=False)

def Set_RLData(df_RLCycle, jtp_id, inner_id, linename, jtp_now, id, lst_RLOrders, df_orders, df_choose, lst_lossorders):
    lst_RLData = [str(jtp_id + 1) + '.' + str(inner_id), linename, jtp_now, df_choose.loc[id, 'dispatch_type'], df_choose.loc[id, 'order_code']]
    lst_RLData.extend(df_orders.O_Status.to_list())
    for w in range(df_choose.shape[0]):
        #if w != id:
        id_w = Opt_func.list_index(lst_RLOrders, df_choose.loc[w, 'order_code'])
        lst_RLData[id_w + 5] = df_choose.loc[w, 'dispatch_type']
    for w in lst_lossorders:
        lst_RLData[Opt_func.list_index(lst_RLOrders, w) + 5] = 'Loss'
    df_RLCycle.loc[df_RLCycle.shape[0]] = lst_RLData
    return df_RLCycle

# =====  Run  ======
def Run_Learning(mainpath, taskname, stagename):
    ret, demand_start, demand_end, begin_day, end_day, df_orders, df_products, df_lines, df_molds, df_orderdata = Opt_func.LoadData_FromDataset(mainpath, taskname)

    lst_onstock_prodcode, lst_onstock_qty, df_orders = Opt_func.Orders_On_Stock(df_products, df_orders)
    df_orders = Opt_func.Orders_Overdue(df_lines, df_orders)
    df_orders_Wait, df_orders_notWait = Opt_func.Get_orders_WaitandNotWait(df_orders)

    maxline_count = 4
    max_epi = df_orders_Wait.shape[0] * 30 / 184
    lst_step = '0.1,0.2,0.3,0.4,1.1,1.2,2.1,2.2,2.3,2.4,2.5,3.1,4.1,4.2,4.3,4.4,4.5,5.0,5.1,5.2'.split(',')
    lst_stname = '生產中調整_產品寬度切換,生產中調整_產品厚度改變,直接切換料_提高比例配方,生產中換料(T6-0%),更換平板滾輪,更換結構板滾輪,更換模頭,平板模頭_提高製造寬度,結構板模頭_提高製造寬度,平板模頭_降低製造寬度,結構板模頭_降低製造寬度,清料桶_降低比例配方,開機前準備,產品寬度切換,產品厚度改變,提高比例配方,換料(T6-0%),等待AM8:00開機,平板開機(bootup),結構板開機(bootup)'.split(',')
    # lst_waste = '150,300,0,300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1500,2000'.split(',')
    lst_hour = '0.5,1,0,1,48,48,24,0,0,0,0,12,4,0,0,0,0,-1,3,3'.split(',')
    lst_parameter = '新寬度=%8,新厚度=%9,新比例配方=%6,新比例配方=%6,調機48小時,調機48小時;結構輪位置=%1;結構間距=%2,模頭號碼=%3;製造寬度=%4;製造厚度=%5,製造寬度=%4;製造厚度=%5,製造寬度=%4;製造厚度=%5,製造寬度=%4;製造厚度=%5,製造寬度=%4;製造厚度=%5,新比例配方=%6,4小時,新寬度=%8,新厚度=%9,新比例配方=%6,新比例配方=%6,等=%7小時,3小時,3小時'.split(',')
    lst_weekday = ['週一','週二','週三','週四','週五','週六','週日']
    Start_Run(mainpath, taskname, stagename, maxline_count, lst_step, lst_stname, lst_parameter, lst_weekday, lst_hour, begin_day, end_day, demand_start, demand_end, max_epi, df_orders_Wait, df_products, df_orderdata, '', '')

def Start_Run(mainpath, taskname, stagename, maxline_count, lst_step, lst_stname, lst_parameter, lst_weekday, lst_hour, begin_day, end_day, demand_start, demand_end, max_epi, df_orders_Wait, df_products, df_orderdata, df_linedata, df_molddata):
    ret_file = os.path.join(os.path.join(os.path.join(mainpath, taskname), stagename), 'ret.txt')
    if os.path.isfile(ret_file): os.remove(ret_file)
    stage_file = os.path.join(os.path.join(os.path.join(mainpath, taskname), stagename), '\StageRecord.xlsx')
    if os.path.isfile(stage_file): os.remove(stage_file)
    json_data, ret = Opt_func.OpenJsonFile(mainpath, os.path.join(taskname, stagename), 'StageParameter.json', '')

    # dirname = '1125_Normal_Accumalate/'
    data_path = os.path.join(os.path.join(mainpath, taskname), stagename)
    if not os.path.isdir(data_path): os.makedirs(data_path)
    # print(json_data)
    lst_RLOrders = df_orders_Wait.order_code.to_list()
    # print(lst_RLOrders)
    for Stage_name in json_data:
        if json_data[Stage_name]['Practice'].lower() == 'no': continue
        if Stage_name == 'Trial Stage':
            max_linearray = [maxline_count] * (Opt_func.getdays(begin_day, end_day) + 1)
            lst_ths = json_data[Stage_name]['OOD GM']['transfer hours']
            n_stage = len(lst_ths) + 1
        else:
            if json_data['Trial Stage']['Practice'].lower() == 'done': json_data[Stage_name]['OOD'] = json_data['Trial Stage']['OOD']
            max_linearray = [maxline_count] * (Opt_func.getdays(begin_day, end_day) + 1)
            for i in range(len(json_data[Stage_name]['OOD'])):
                max_linearray[Opt_func.getdays(begin_day, json_data[Stage_name]['OOD'][i])] += 1
            n_stage = 1
        max_epi_noloss = 0
        # max_epi_set = 0
        for ths_index in range(n_stage):
            json_data[Stage_name]['Practice'] = 'Processing'
            save_jsonfile(os.path.join(os.path.join(os.path.join(mainpath, taskname), stagename), 'StageParameter.json'), json_data)
            print(Stage_name + (' Trial' + str(ths_index + 1) if Stage_name == 'Trial Stage' else ''))
            min_waste = 10000
            min_losscount = 1000
            max_ph = 0
            if Stage_name == 'Trial Stage':
                lst_epicount = json_data[Stage_name]['ME ratio']
            else:
                lst_epicount = [json_data[Stage_name]['ME']] * json_data[Stage_name]['PLT']

            learn_weight = json_data[Stage_name]['Priority Score']
            Learning_No = 1
            Learning_max = len(lst_epicount)  # 20
            minWaste_at = -1

            date_no = ('_TR' + str(ths_index + 1)) if Stage_name == 'Trial Stage' else '_TG'
            # stage_path = '\\PSR' + date_no
            # if not os.path.isdir(data_path + stage_path): os.makedirs(data_path + stage_path)

            learn_file = os.path.join(data_path,  'history_learn' + date_no + '.txt')
            # episode_file = data_path + stage_path + '\\history_epi' + date_no + '.txt'
            # ordercsv_file = ''  # dirname +  'order' + date_no + 'L' + str(Learning_No) '.csv'
            # jtpcsv_file = ''  # dirname +  'result_jtp' + date_no + 'L' + str(Learning_No) + '.csv'
            # data_file = ''  # dirname +  'data_array' + date_no + 'L' + str(Learning_No)
            # loss_file = data_path + stage_path + '\\lossfile' + date_no + '.txt'
            learn_ref_file = os.path.join(data_path, 'learn_ref' + date_no + '.txt')
            # cycle_file = data_path + stage_path + '\\cycle' + date_no + '.csv'
            min_Ordercsv = ''

            fn = open(learn_file, 'w')
            fn.writelines('Learning No.,Start Time,Time Cost,Episode Count,Production hours,Extra PHs,Waste,Loss Orders,Report\n')
            fn.close()
            fn = open(learn_ref_file, 'w')
            fn.close()

            save_jsonfile(os.path.join(data_path, 'StagePara' + date_no + '.json'), json_data)
            # json_file = data_path + '\\StagePara' + date_no + '.json'
            # with open(json_file, 'w') as fn_json:
            #     json.dump(json_data, fn_json, ensure_ascii=False)

            Trial_loss = 1
            columns = ['Learning Number', 'Episode Number', 'Cycle Number', 'Date', '派工清單種類', '派工訂單', '學習參考紀錄']
            df_cycle = pd.DataFrame(columns=columns)
            df_lines = pd.DataFrame({'id': []})
            df_molds = pd.DataFrame({'id': []})
            while Learning_No <= Learning_max:
                learn_name = date_no + 'L' + str(Learning_No)
                learn_path = os.path.join(data_path, 'PSR' + learn_name)
                if not os.path.isdir(learn_path): os.makedirs(learn_path)
                episode_file = os.path.join(learn_path, 'history_epi.txt')  # + learn_name + '.txt'
                ordercsv_file = os.path.join(learn_path, 'OrderList.csv')   # + learn_name + '.csv'
                jtpcsv_file = os.path.join(learn_path, 'result_jtp.csv')  # + learn_name + '.csv'
                data_file = os.path.join(learn_path, 'PracticeResult')   # + learn_name
                # cycle_file = learn_path + '\\CycleRunning.log'   # + learn_name + '.csv'

                fn = open(episode_file, 'w')
                fn.writelines('Learning No.,episode,cycle number,time,time cost,Production hours,waste,loss orders\n')
                fn.close()

                orders_learning_ref = read_learning_ref('')
                episode = 1
                epi_max = math.floor(lst_epicount[Learning_No - 1] * (1 if Stage_name == 'Target Stage' else max_epi))

                t_learn_start = dt.datetime.now()
                print('Learning No. ' + str(Learning_No) + ' ' + str(dt.datetime.now()))
                print('--------', epi_max)
                loss_count = 1
                showLearnRef = False
                lst_lossorders = []
                # to_output = False

                df_orders = pd.DataFrame({'id' : []})
                df_jtp = pd.DataFrame({'id' : []})
                lst_block = []
                # jtp_now = dt.datetime.now()
                jtp_next = dt.datetime.now()
                tot_waste = 0
                ph_sum = 0

                while episode <= epi_max and loss_count > 0:
                    t_epi_start = dt.datetime.now()

                    cycle_file = os.path.join(learn_path, learn_name[1:] + 'E' + str(episode) + '.log')
                    RLC_file = os.path.join(learn_path, learn_name[1:] + 'E' + str(episode) + '.epd')
                    columns = ['Cycle No.', 'Line No.', 'JTP Time', 'Selected Order Type', 'Selected Order']
                    columns.extend(lst_RLOrders)
                    df_RLCycle = pd.DataFrame(columns=columns)

                    print('  Episode ' + str(episode) + ' Start:')
                    if os.path.isfile(os.path.join(os.path.join(os.path.join(mainpath, taskname), 'Dataset'), 'lines.csv')):
                        df_lines = pd.read_csv(os.path.join(os.path.join(os.path.join(mainpath, taskname), 'Dataset'), 'lines.csv'))
                        df_lines['Line Begin'] = df_lines['Line Begin'].astype('datetime64[ns]')
                        df_molds = pd.read_csv(os.path.join(os.path.join(os.path.join(mainpath, taskname), 'Dataset'), 'molds.csv'))
                        df_lines.fillna('', inplace=True)
                        df_molds.fillna('', inplace=True)
                    else:
                        df_lines = Opt_func.readlines(df_linedata, df_products)
                        df_molds = Opt_func.readmolds(df_lines, df_molddata)
                        Opt_func.updatelines(df_lines, df_molds, 0)

                        # df_products = readproducts()
                    # begin_day, end_day, df_orders = readorders()
                    df_orders = df_orders_Wait.copy(deep=True)

                    # initialize blocks
                    All_Lines = Opt_func.sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())
                    lst_lines = Opt_func.sortlist_bynum(df_lines[df_lines['Usable'] == 'YES'].line_name.to_list())
                    lst_block = initial_block(500, df_lines, lst_lines)

                    i_period = 7 * 24
                    jtp_id = 0
                    tot_waste = 0
                    ph_sum = 0
                    df_jtp = initial_jtp(df_lines)
                    jtp_min, jtp_max, hour_endday = get_jtptime(begin_day, end_day, i_period, df_jtp)
                    show = False
                    df_orders['O_Status'] = 'Waiting'
                    while jtp_id < 1000 and hour_endday > 0:
                        jtp_now = jtp_min
                        lst_lossorders = get_lossorders(jtp_now, df_orders)
                        ph_sum = get_orders_ph(df_orders)
                        loss_count = len(lst_lossorders)
                        # print('loss:', loss_count)
                        orders_learning_ref = sorted(orders_learning_ref, key=lambda s: s[3])
                        if episode < epi_max and loss_count == 0 or episode >= epi_max:
                            lst_lines, lst_dates = get_jtp_linelist(i_period, jtp_now, df_jtp)  # order by date, random if same date
                            # lst_molds = [df_lines.loc[df_lines[df_lines['line_name'] == lst_lines[x]].index[0], 'mold_code'] for x in range(len(lst_lines))]  # df_lines[df_lines['Usable'] == 'on'].mold_code.to_list()
                            # lst_mold_canuse = df_molds[df_molds['Usage'] == ''].mold_code.to_list()
                            if show:
                                print('line list:', lst_lines)
                                print('')

                            pd.set_option('expand_frame_repr', False)
                            for index in range(1):  # len(lst_lines)):
                                lst_molds = [df_lines.loc[df_lines[df_lines['line_name'] == lst_lines[x]].index[0], 'mold_code'] for x in range(len(lst_lines))]  # df_lines[df_lines['Usable'] == 'on'].mold_code.to_list()
                                lst_mold_canuse = df_molds[df_molds['Usage'] == ''].mold_code.to_list()

                                jtp_now = lst_dates[index]
                                line_id = Opt_func.GetLineID_sorted(lst_lines[index], df_lines)
                                block_id = Opt_func.Get_blockcount(line_id, lst_block)
                                # print('OLM Start')
                                df_ret = Opt_SL_Func.OLM_Orders(index, jtp_now, False, block_id, lst_block, line_id, lst_lines, lst_molds, lst_mold_canuse, df_lines, df_products, df_orders)

                                df_order_dp = df_ret[df_ret['dispatch_type'] == 'DP']
                                df_order_it = df_ret[df_ret['dispatch_type'] == 'IT']
                                df_order_ns = df_ret[df_ret['dispatch_type'] == 'NS']
                                df_order_instant = combine_dataframe(df_order_dp, df_order_it)  # pd.concat([df_order_dp, df_order_it])  # to do: random order for dp and order by waste for it
                                if show: print('(' + str(index + 1) + ')', lst_lines[index], 'order count', df_ret.shape[0], ':  Orders_DP count', df_order_dp.shape[0], ', Orders_IT count', df_order_it.shape[0], ', Orders_NS count', df_order_ns.shape[0])
                                n_job = True
                                # print(jtp_now, All_Lines[line_id], 'order count', df_order_dp.shape[0], df_order_it.shape[0], df_order_ns.shape[0])
                                if df_ret.shape[0] > 0:
                                    if show: print(df_ret[['order_code', 'type', 'width', 'dispatch_type', 'dispatch_step', 'tune_step', 'tune_x', 'tune_w', 'waste_hour']])

                                    nProdcount = Get_ProductionCount(All_Lines, line_id, jtp_now, lst_block)
                                    if nProdcount < max_linearray[Opt_func.getdays(begin_day, str(jtp_now)[:10])]:
                                        if df_order_instant.shape[0] > 0:
                                            inner_id = 1
                                            psum_ret = ''
                                            while df_order_instant.shape[0] > 0:
                                                dp_id, duration, dp_waste, Priority_ret = setblock_instantorder(index, block_id, lst_block, line_id, lst_lines, jtp_now, lst_step, lst_stname, lst_parameter, lst_hour, orders_learning_ref, df_products,df_lines, df_order_instant, episode)
                                                tot_waste += dp_waste
                                                jtp_next = jtp_now + dt.timedelta(hours=duration)
                                                df_orders.loc[df_orders['order_code'] == df_order_instant.loc[dp_id, 'order_code'], 'O_Status'] = 'Done'
                                                df_cycle.loc[df_cycle.shape[0]] = [Learning_No, episode, str(jtp_id + 1) + '.' + str(inner_id), jtp_now, 'Orders_Instant', lst_lines[0] + ':' + df_order_instant.loc[dp_id, 'order_code'], Priority_ret + psum_ret]

                                                # lst_RLData = []
                                                # lst_RLData.append(str(jtp_id + 1) + '.' + str(inner_id))
                                                # lst_RLData.append(lst_lines[0])
                                                # lst_RLData.append(jtp_now)
                                                # lst_RLData.append(df_order_instant.loc[dp_id, 'dispatch_type'])
                                                # lst_RLData.append(df_order_instant.loc[dp_id, 'order_code'])
                                                # lst_RLData.extend(df_orders.O_Status.to_list())
                                                # for w in range(df_order_instant.shape[0]):
                                                #     if w != dp_id:
                                                #         id_w = Opt_func.list_index(lst_RLOrders, df_order_instant.loc[w, 'order_code'])
                                                #         lst_RLData[id_w + 5] = df_order_instant.loc[w, 'dispatch_type']
                                                # df_RLCycle.loc[df_RLCycle.shape[0]] = lst_RLData
                                                df_RLCycle = Set_RLData(df_RLCycle, jtp_id, inner_id, lst_lines[0], jtp_now, dp_id, lst_RLOrders, df_orders, df_order_instant, lst_lossorders)

                                                df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'id'] = jtp_id + 1
                                                df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'sub id'] = inner_id
                                                df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'Status'] = 'Done'
                                                df_jtp.loc[df_jtp.shape[0]] = [lst_lines[index], jtp_next, '', 0, 0]
                                                if show: print('    Next Job Time Point is', jtp_next)
                                                inner_id += 1

                                                jtp_now = jtp_next
                                                block_id = Opt_func.Get_blockcount(line_id, lst_block)
                                                df_ret2 = Opt_SL_Func.OLM_Orders(index, jtp_now, False, block_id, lst_block, line_id, lst_lines, lst_molds, lst_mold_canuse, df_lines, df_products, df_orders)

                                                df_order_dp2 = df_ret2[df_ret2['dispatch_type'] == 'DP']
                                                df_order_it2 = df_ret2[df_ret2['dispatch_type'] == 'IT']
                                                df_order_ns2 = df_ret2[df_ret2['dispatch_type'] == 'NS']
                                                df_order_instant = combine_dataframe(df_order_dp2, df_order_it2)

                                                if df_order_instant.shape[0] > 0:
                                                    psum = 0
                                                    for w in range(len(orders_learning_ref)):
                                                        if df_order_ns2[df_order_ns2['order_code'] == orders_learning_ref[w][0]].shape[0] > 0: psum += orders_learning_ref[w][3]
                                                    ncount = df_order_instant.shape[0]
                                                    # if psum > 0: print('    Psum:', 'Outer ID', jtp_id, 'Inner ID', inner_id, 'Psum', psum, 'New Orders count', ncount, 'Deduction', ncount - psum)
                                                    psum_ret = '\nList Deduction:' + str(ncount - psum)
                                                    if df_order_instant.shape[0] <= psum:
                                                        df_order_instant = df_order_instant[:-df_order_instant.shape[0]]
                                                        print('    End of inner circle.')

                                                # print(jtp_id + 1, inner_id - 1, df_ret2.shape[0], df_order_instant.shape[0])
                                            n_job = False
                                        elif df_order_ns.shape[0] > 0:
                                            ns_id, dura_prod, dura_tune, dura_shutdown, ns_waste, Weights_ret, ns_ok = setblock_neworder(show, index, block_id, lst_block, line_id, lst_lines, jtp_id, jtp_now, learn_weight, lst_step, lst_stname, lst_parameter, lst_hour, df_products, df_lines, df_molds, orders_learning_ref, df_order_ns)
                                            if ns_ok == 0:  # ns_ok return value only 0 or -1.
                                                tot_waste += ns_waste
                                                jtp_next = jtp_now + dt.timedelta(hours=dura_shutdown + dura_tune + dura_prod)
                                                df_orders.loc[df_orders['order_code'] == df_order_ns.loc[ns_id, 'order_code'], 'O_Status'] = 'Done'
                                                df_cycle.loc[df_cycle.shape[0]] = [Learning_No, episode, str(jtp_id + 1) + '.0', jtp_now, 'Orders_New', lst_lines[0] + ':' + df_order_ns.loc[ns_id, 'order_code'], Weights_ret]

                                                # lst_RLData = []
                                                # lst_RLData.append(str(jtp_id + 1) + '.0')
                                                # lst_RLData.append(lst_lines[0])
                                                # lst_RLData.append(jtp_now)
                                                # lst_RLData.append(df_order_ns.loc[ns_id, 'dispatch_type'])
                                                # lst_RLData.append(df_order_ns.loc[ns_id, 'order_code'])
                                                # lst_RLData.extend(df_orders.O_Status.to_list())
                                                # for w in range(df_order_ns.shape[0]):
                                                #     if w != ns_id:
                                                #         id_w = Opt_func.list_index(lst_RLOrders, df_order_ns.loc[w, 'order_code'])
                                                #         lst_RLData[id_w + 5] = df_order_ns.loc[w, 'dispatch_type']
                                                # df_RLCycle.loc[df_RLCycle.shape[0]] = lst_RLData
                                                df_RLCycle = Set_RLData(df_RLCycle, jtp_id, 0, lst_lines[0], jtp_now, ns_id, lst_RLOrders, df_orders, df_order_ns, lst_lossorders)

                                                df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'id'] = jtp_id + 1
                                                df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'Status'] = 'Done'
                                                df_jtp.loc[df_jtp.shape[0]] = [lst_lines[index], jtp_next, '', 0, 0]
                                                if show: print('    Next Job Time Point is', jtp_next)

                                                inner_id = 0
                                                while inner_id == 0 or inner_id > 0 and df_order_instant.shape[0] > 0:
                                                    inner_id += 1
                                                    jtp_now = jtp_next
                                                    block_id = Opt_func.Get_blockcount(line_id, lst_block)
                                                    df_ret2 = Opt_SL_Func.OLM_Orders(index, jtp_now, False, block_id, lst_block, line_id, lst_lines, lst_molds, lst_mold_canuse, df_lines, df_products, df_orders)

                                                    df_order_dp2 = df_ret2[df_ret2['dispatch_type'] == 'DP']
                                                    df_order_it2 = df_ret2[df_ret2['dispatch_type'] == 'IT']
                                                    df_order_ns2 = df_ret2[df_ret2['dispatch_type'] == 'NS']
                                                    df_order_instant = combine_dataframe(df_order_dp2, df_order_it2)

                                                    psum_ret = ''
                                                    if df_order_instant.shape[0] > 0:
                                                        psum = 0
                                                        for w in range(len(orders_learning_ref)):
                                                            if df_order_ns2[df_order_ns2['order_code'] == orders_learning_ref[w][0]].shape[0] > 0: psum += orders_learning_ref[w][3]
                                                        ncount = df_order_instant.shape[0]
                                                        # if psum > 0: print('    Psum:', 'Outer ID', jtp_id, 'Inner ID', inner_id, 'Psum', psum, 'New Orders count', ncount, 'Deduction', ncount - psum)
                                                        psum_ret = '\nList Deduction:' + str(ncount - psum)
                                                        if df_order_instant.shape[0] <= psum:
                                                            df_order_instant = df_order_instant[:-df_order_instant.shape[0]]
                                                            print('    End of inner circle.')
                                                    if df_order_instant.shape[0] > 0:
                                                        dp_id, duration, dp_waste, Priority_ret = setblock_instantorder(index, block_id, lst_block, line_id, lst_lines, jtp_now, lst_step, lst_stname, lst_parameter, lst_hour, orders_learning_ref, df_products,df_lines, df_order_instant, episode)
                                                        tot_waste += dp_waste
                                                        jtp_next = jtp_now + dt.timedelta(hours=duration)
                                                        df_orders.loc[df_orders['order_code'] == df_order_instant.loc[dp_id, 'order_code'], 'O_Status'] = 'Done'
                                                        df_cycle.loc[df_cycle.shape[0]] = [Learning_No, episode, str(jtp_id + 1) + '.' + str(inner_id), jtp_now, 'Orders_Instant', lst_lines[0] + ':' + df_order_instant.loc[dp_id, 'order_code'], Priority_ret + psum_ret]

                                                        # lst_RLData = []
                                                        # lst_RLData.append(str(jtp_id + 1) + '.' + str(inner_id))
                                                        # lst_RLData.append(lst_lines[0])
                                                        # lst_RLData.append(jtp_now)
                                                        # lst_RLData.append(df_order_instant.loc[dp_id, 'dispatch_type'])
                                                        # lst_RLData.append(df_order_instant.loc[dp_id, 'order_code'])
                                                        # lst_RLData.extend(df_orders.O_Status.to_list())
                                                        # for w in range(df_order_instant.shape[0]):
                                                        #     if w != dp_id:
                                                        #         id_w = Opt_func.list_index(lst_RLOrders, df_order_instant.loc[w, 'order_code'])
                                                        #         lst_RLData[id_w + 5] = df_order_instant.loc[w, 'dispatch_type']
                                                        # df_RLCycle.loc[df_RLCycle.shape[0]] = lst_RLData
                                                        df_RLCycle = Set_RLData(df_RLCycle, jtp_id, inner_id, lst_lines[0], jtp_now, dp_id, lst_RLOrders, df_orders, df_order_instant, lst_lossorders)

                                                        df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'id'] = jtp_id + 1
                                                        df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'sub id'] = inner_id
                                                        df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'Status'] = 'Done'
                                                        df_jtp.loc[df_jtp.shape[0]] = [lst_lines[index], jtp_next, '', 0, 0]
                                                        if show: print('    Next Job Time Point is', jtp_next)
                                            n_job = False
                                        else:
                                            n_job = False
                                    else:  # Production Line Count >=4
                                        jtp_next = GetMinJtpTime_notme(All_Lines, line_id, lst_block)
                                        # print(All_Lines[line_id], jtp_next)
                                        df_cycle.loc[df_cycle.shape[0]] = [Learning_No, episode, str(jtp_id + 1) + '.0', jtp_now, 'NB', lst_lines[0], '']
                                        nb_waste = setblock_noorder(jtp_next, block_id, lst_block, line_id, jtp_now)
                                        tot_waste += nb_waste
                                else:  # no orders
                                    mask1 = functools.reduce(np.logical_and, (
                                    df_orders['not_before'] > jtp_now, df_orders['O_Status'] == 'Waiting',
                                    df_orders['Do_Lines'].str.contains(lst_lines[index])))
                                    mask2 = df_orders['Do_Molds'].str.contains(lst_molds[index])
                                    df_tmp = df_orders[np.logical_and(mask1, mask2)]

                                    # jtp_next = df_orders[df_orders['not_before'] > jtp_now].not_before.min() if df_tmp.shape[0] > 0 else (dt.datetime.strptime(end_day, '%Y-%m-%d'))
                                    jtp_next = (dt.datetime.strptime(end_day, '%Y-%m-%d') + dt.timedelta(days=1)) if \
                                    df_tmp.shape[0] == 0 else df_tmp[df_tmp['not_before'] > jtp_now].not_before.min()
                                    df_cycle.loc[df_cycle.shape[0]] = [Learning_No, episode, str(jtp_id + 1) + '.0', jtp_now, 'NB', lst_lines[0], '']
                                    nb_waste = setblock_noorder(jtp_next, block_id, lst_block, line_id, jtp_now)
                                    tot_waste += nb_waste

                                if n_job:
                                    df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'id'] = jtp_id + 1
                                    df_jtp.loc[np.logical_and(df_jtp['line_name'] == lst_lines[index], df_jtp['Status'] == ''), 'Status'] = 'Done'
                                    df_jtp.loc[df_jtp.shape[0]] = [lst_lines[index], jtp_next, '', 0, 0]
                                    if show: print('    Next Job Time Point is', jtp_next)
                                if show: print('')

                            # df_jtp.loc[df_jtp['Status'] == 'new', 'Status']= ''
                            jtp_min, jtp_max, hour_endday = get_jtptime(begin_day, end_day, i_period, df_jtp)
                            jtp_id += 1
                            if loss_count > 0 and episode >= epi_max and not showLearnRef:
                                orders_learning_ref = Data_Append(lst_lossorders, jtp_min, episode, learn_weight, orders_learning_ref)
                                Show_learning_ref(learn_ref_file, Learning_No, orders_learning_ref)
                                showLearnRef = True
                        else:
                            # df_jtp.loc[df_jtp['Status'] == 'new', 'Status']= ''
                            jtp_min, jtp_max, hour_endday = get_jtptime(begin_day, end_day, i_period, df_jtp)
                            jtp_id += 1
                            if episode < epi_max or episode == epi_max and hour_endday <= 0:
                                # print('Episode ' + str(episode) + ' end at ', jtp_min)
                                orders_learning_ref = Data_Append(lst_lossorders, jtp_min, episode, learn_weight, orders_learning_ref)
                                Show_learning_ref('', Learning_No, orders_learning_ref)
                                # hour_endday = 0
                                print('  End of Episode ' + str(episode), ', JTP Cycle:', jtp_id, ', JTP time', jtp_now, ', waste:', int(tot_waste) / 1000, ', Production hours:', ph_sum, ', losss orders:', lst_lossorders)
                                print('    ' + str(len(lst_lossorders)) + ' orders loss.')
                                break

                    tot_waste = int(tot_waste) / 1000
                    t_epi_end = dt.datetime.now()
                    fn = open(episode_file, 'a')
                    fn.writelines(str(Learning_No) + ',' + str(episode) + ',' + str(jtp_id) + ',' + str(t_epi_start)[:19] + ',' + str((t_epi_end - t_epi_start).seconds) + ',' + str(ph_sum) + ',' + str(tot_waste) + ',"' + str(lst_lossorders) + '"\n')
                    fn.close()
                    # print('  End of Episode ' + str(episode), ', ' + str(get_allloss_count()) + ' orders loss.')
                    if loss_count == 0:
                        Trial_loss = 0
                        # print('find', max_epi_noloss, episode, epi_max)
                        if max_epi_noloss < episode:
                            max_epi_noloss = episode
                            # max_epi_set = epi_max

                    if loss_count == 0 or episode >= epi_max:
                        # if min_losscount >= loss_count:
                        #  if min_losscount > loss_count or min_losscount == loss_count and min_waste > tot_waste:
                        if max_ph <= ph_sum:
                            if max_ph < ph_sum or max_ph == ph_sum and min_waste > tot_waste:
                                # to_output = True
                                min_waste = tot_waste
                                minWaste_at = Learning_No
                                min_losscount = loss_count
                                min_Ordercsv = ordercsv_file
                            max_ph = ph_sum
                    if episode == epi_max or loss_count == 0:
                        df_tmp = df_cycle[df_cycle['Learning Number'] == Learning_No]
                        m_epi = df_tmp['Episode Number'].max()
                        df_tmp[df_tmp['Episode Number'] == m_epi].to_csv(cycle_file, index=False)
                        df_RLCycle.to_csv(RLC_file, index=False)
                    episode += 1
                    if os.path.isfile(ret_file): break

                # print(loss_count)
                df_orders['O_Status'] = 'Done'
                if loss_count > 0: df_orders.loc[df_orders['order_code'].isin(lst_lossorders), 'O_Status'] = 'LOST'
                df_orders.to_csv(ordercsv_file, index=False)

                # if to_output:
                lst_bestresult = copy.deepcopy(lst_block)
                arr = np.array(lst_bestresult)
                np.save(data_file, arr)
                df_jtp.to_csv(jtpcsv_file, index=False)

                # df_tmp = df_cycle[df_cycle['Learning Number'] == Learning_No]
                # m_epi =  df_tmp['Episode Number'].max()
                # df_tmp[df_tmp['Episode Number'] == m_epi].to_csv(cycle_file, index=False)
                # output_file = 'PSR' + date_no + 'L' + str(Learning_No) + '.xlsx'
                # Output_Report(dirname + output_file)

                if loss_count == 0:
                    print('  End of Episode ' + str(episode - 1),
                          ' no loss, Production hours:' + str(ph_sum) + ', total waste:', tot_waste, ', min waste:',
                          min_waste)
                else:
                    print('  End of Episode ' + str(episode - 1),
                          ', ' + str(len(lst_lossorders)) + ' orders loss, Production hours:' + str(
                              ph_sum) + ', total waste:', tot_waste, ', min waste:', min_waste,
                          '(' + str(min_losscount) + ' orders loss)')
                    print('    Loss orders list: ' + ','.join(get_allloss(df_orders)))
                    # fn = open(loss_file, 'a')
                    # if episode >= epi_max:
                    #     fn.writelines(str(Learning_No) + ',' + ','.join(get_allloss(df_orders)) + '\n')
                    # else:
                    #     fn.writelines(str(Learning_No) + ',' + ','.join(get_lossorders(jtp_now, df_orders)) + '\n')
                    # fn.close()
                    df_orders.loc[df_orders['order_code'].isin(get_allloss(df_orders)), 'O_Status'] = 'LOST'

                print('--------\n')
                Learning_No += 1
                t_learn_end = dt.datetime.now()
                lst_PHs = Opt_func.GetEvery_PHs(lst_block, 250, df_lines)
                print(lst_PHs)
                extra_PHs = sum([(0 if x < maxline_count * 24 else x - maxline_count * 24) for x in lst_PHs])
                output_file = 'PSR' + date_no + 'L' + str(Learning_No - 1) + '.xlsx'
                Opt_RS_Report.Output_Report(mainpath, taskname, os.path.join(data_path, output_file), ordercsv_file, data_file, df_products, df_orders, df_lines, df_molds, df_orderdata, df_linedata,
                                         ths_index, Stage_name, maxline_count, Learning_No, Learning_max, episode, epi_max, demand_start, demand_end, begin_day, end_day, extra_PHs, tot_waste, lst_weekday, lst_lossorders, max_linearray)
                fn = open(learn_file, 'a')
                fn.writelines(str(Learning_No - 1) + ',' + str(t_learn_start)[:19] + ',' + str((t_learn_end - t_learn_start).seconds) + ',' + str(episode - 1) + ',' + str(ph_sum) + ',' + str(extra_PHs) + ',' + str(tot_waste) + ',"' + str(lst_lossorders) + '",' + output_file + '\n')
                fn.close()
                Opt_SL_Func.save_stagerecord(stage_file, max_epi)
                if os.path.isfile(ret_file): break

            if Stage_name == 'Trial Stage' and ths_index < n_stage - 1:
                # lst_OOD = []
                print('loss order file:', min_Ordercsv)
                if json_data['Trial Stage']['OOD GM']['Method'] == 'Not-After':
                    lst_OOD = GetExtraDay(min_Ordercsv, json_data['Trial Stage']['OOD GM']['transfer hours'][ths_index], begin_day, demand_start, json_data)
                    max_linearray = [maxline_count] * (Opt_func.getdays(begin_day, end_day) + 1)
                    for i in range(len(json_data[Stage_name]['OOD'])):
                        max_linearray[Opt_func.getdays(begin_day, json_data[Stage_name]['OOD'][i])] += 1
                else:
                    max_linearray, lst_OOD = Calc_OOD(min_Ordercsv, ths_index, maxline_count, begin_day, end_day, json_data, df_lines)
                json_data['Trial Stage']['OOD'] = lst_OOD

            fn = open(learn_file, 'a')
            if minWaste_at >= 0:
                fn.writelines('\nminimum waste ' + str(min_waste) + ' at Learing No. ' + str(minWaste_at) + '\n')
            else:
                fn.writelines('\nAll orders lost.\n')
            fn.close()
            Opt_SL_Func.save_stagerecord(stage_file, max_epi)

            print(dt.datetime.now())
            if os.path.isfile(ret_file): break
            if Stage_name == 'Trial Stage' and Trial_loss == 0:
                max_epi_set = min(k2 for k2 in [math.floor(k1 * max_epi) for k1 in json_data['Trial Stage']['ME ratio']] if k2 >= max_epi_noloss)
                print('find no loss, Target episode ', max_epi_set)
                json_data['Target Stage']['ME'] = max_epi_set
                break

        json_data[Stage_name]['Practice'] = 'Done'
        save_jsonfile(os.path.join(os.path.join(os.path.join(mainpath, taskname), stagename), 'StageParameter.json'), json_data)
        if os.path.isfile(ret_file):
            os.remove(ret_file)
            print('*** ', '結束排程', ' ***')
            break