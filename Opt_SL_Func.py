import json
import math
import os
import pandas as pd
import functools
import datetime as dt
import numpy as np

import Opt_func

def Run_Stage(mainpath, taskname, stagename):
    fn = open(os.path.join(mainpath, 'stagerun.txt'), 'a')
    fn.write(dt.datetime.strftime(dt.datetime.now(),  '%Y-%m-%d %H:%M:%S') + ',' + mainpath + ',' + taskname + ',' + stagename + '\n')
    fn.close()

def Stop_Stage(mainpath, taskname, stagename):
    spath = os.path.join(os.path.join(mainpath, taskname), stagename)
    fn = open(os.path.join(spath, 'ret.txt'), 'w')
    fn.write('Stop\n')
    fn.close()

def Get_Paradata(df, id, Y, moldcode, sPara, df_products):
    for v in range(1, 10):
        p = sPara.find('=%' + str(v))
        if p >= 0:
            if v == 1:
                df_tmp = df_products[df_products['product_code'] == df.loc[id, 'product_code']]
                ret = df_tmp.loc[df_tmp.index[0], 'roller_position']
            elif v == 2:
                df_tmp = df_products[df_products['product_code'] == df.loc[id, 'product_code']]
                ret = df_tmp.loc[df_tmp.index[0], 'lenti_pitch']
            elif v == 3:
                ret = moldcode
            elif v == 4:
                ret = (1000 if df.loc[id, 'width'] < 1000 else df.loc[id, 'width']) + (70 if df.loc[id, 'type'] == 'lenti' else 50)   #df.loc[id, 'width']
            elif v == 5:
                ret = df.loc[id, 'height']
            elif v == 6:
                ret = df.loc[id, 'composition']
            elif v == 7:
                ret = str(Y)
            elif v == 8:
                ret = df.loc[id, 'width']
            elif v == 9:
                ret = df.loc[id, 'height']
            else:
                ret = ''
            if ret != '': sPara = sPara[:p+1] + str(ret) + sPara[p+3:]
    return sPara

# ===== Stage Record =====
def save_stagerecord(sfile, max_epi):
    dirname = os.path.dirname(sfile)
    if os.path.isdir(dirname):
        writer = pd.ExcelWriter(sfile, engine='openpyxl')
        ncol = ['Learning No.', 'Episode Count', 'Start Time', 'Time Cost', 'Production hours', 'Extra PHs', 'Waste', 'Loss Orders', 'Report']
        for i in range(2):
            for k in range(10 if i == 0 else 1):
                TG_name = ('TR' if i == 0 else 'TG') + (str(k + 1) if i == 0 else '')
                for l_no in range(1, 100):
                    # stage_path = '\\PSR_' + TG_name + 'L' + str(l_no)
                    Stage_file = dirname + '\\history_learn_' + TG_name + '.txt'
                    Stange_name = 'Trial' + str(k + 1) if i == 0 else 'Target Stage'
                    if os.path.isfile(Stage_file):
                        f = open(dirname + '\\StagePara_' + TG_name + '.json', 'r')
                        jlog_data = json.load(f)
                        f.close()
                        if i == 0:
                            strOOD = str(jlog_data['Trial Stage']['OOD'])
                            strME = str([math.floor(x * max_epi) for x in jlog_data['Trial Stage']['ME ratio']])
                        else:
                            strOOD = str(jlog_data['Target Stage']['OOD'])
                            strME = str(jlog_data['Target Stage']['ME'])

                        df_learn = pd.read_csv(Stage_file)
                        df_ln_new = df_learn[ncol]
                        df_ln_new.to_excel(writer, sheet_name=Stange_name, startrow=3, startcol=0, index=False)
                        worksheet = writer.sheets[Stange_name]
                        worksheet.cell(row=1, column=1).value = 'OOD'
                        worksheet.cell(row=1, column=2).value = strOOD
                        worksheet.cell(row=2, column=1).value = 'ME'
                        worksheet.cell(row=2, column=2).value = strME
                    else:
                        break
        writer.save()

def frontblocktype(line_id, block_id, lst_block):
    if block_id > 0:
        ret = lst_block[line_id][block_id - 1][0]
        ret = 'S' if ret == 'No Bootup' or ret == 'Shutdown' else 'P'
    else:
        ret = ''
    return ret

def OLM_Orders(index, jtp_runtime, bShow, block_id, lst_block, line_id, lst_lines, lst_molds, lst_mold_canuse, df_lines, df_products, df_orders):
    df_tmp = df_lines[df_lines['line_name'] == lst_lines[index]]

    # get orders in JTP time range, Status=Waiting and do_lines; set: 集合
    mask1 = functools.reduce(np.logical_and, (df_orders['not_before'] <= jtp_runtime, df_orders['O_Status'] == 'Waiting', df_orders['Do_Lines'].str.contains(lst_lines[index]),
                            [df_orders.loc[x, 'not_after'] - dt.timedelta(hours=df_orders.loc[x, 'Production_Hours']) >= jtp_runtime for x in range(df_orders.shape[0])]))
    mask2 = df_orders['Do_Molds'].str.contains(lst_molds[index])
    mask3 = [True if len(set(df_orders.loc[x, 'Do_Molds'].split(';')) & set(lst_mold_canuse)) > 0 else False for x in range(df_orders.shape[0])]
    df_ret = df_orders[np.logical_and(mask1, np.logical_or(mask2, mask3))].sort_values(by=['not_after'])
    df_ret.reset_index(drop=True, inplace=True)

    l_type = df_tmp.loc[df_tmp.index[0], 'type']
    l_pitch = df_tmp.loc[df_tmp.index[0], 'lenti_pitch']
    l_roller = df_tmp.loc[df_tmp.index[0], 'roller_position']
    l_moldcode = df_tmp.loc[df_tmp.index[0], 'mold_code']
    l_mfgwidth = df_tmp.loc[df_tmp.index[0], 'mfg_width']
    l_width = df_tmp.loc[df_tmp.index[0], 'width']
    l_composition = df_tmp.loc[df_tmp.index[0], 'composition']
    l_thickness = df_tmp.loc[df_tmp.index[0], 'thickness']

    if bShow: print('(' + str(index + 1) + ') line name:', df_tmp.loc[df_tmp.index[0], 'line_name'], 'mold_code:', l_moldcode)
    df_ret['dispatch_type'] = ''
    df_ret['dispatch_step'] = ''
    df_ret['tune_step'] = ''
    df_ret['tune_x'] = 0
    df_ret['tune_w'] = 0
    df_ret['waste_hour'] = 0
    if df_ret.shape[0] > 0:
        if bShow: print('orders count:', df_ret.shape[0])

        n1 = df_ret.columns.get_loc('type')
        n2 = df_ret.columns.get_loc('product_code')
        n3 = df_ret.columns.get_loc('Do_Molds')
        n4 = df_ret.columns.get_loc('width')
        n5 = df_ret.columns.get_loc('length')
        n6 = df_ret.columns.get_loc('height')
        n7 = df_ret.columns.get_loc('density')
        n8 = df_ret.columns.get_loc('quantity')
        # n9 = df_ret.columns.get_loc('order_code')
        for i in range(df_ret.shape[0]):
            # print('order code', df_ret.iloc[i,n9])
            o_type = df_ret.iloc[i, n1]
            o_prod_code = df_ret.iloc[i, n2]
            o_domold = df_ret.iloc[i, n3]

            tx = df_products[df_products['product_code'] == o_prod_code]
            o_pitch = tx.loc[tx.index[0], 'lenti_pitch']
            o_roller = tx.loc[tx.index[0], 'roller_position']
            o_width = tx.loc[tx.index[0], 'width']
            o_thickness = tx.loc[tx.index[0], 'height']
            o_composition = tx.loc[tx.index[0], 'composition']

            lst_result = []
            s = ' (1) ' + l_type + ' to ' + o_type + ' => '
            if l_type != o_type:
                if l_type == 'plate':
                    s += ' 1.2 更換結構板滾輪'
                    lst_result.append('1.2')
                else:
                    s += ' 1.1 更換平板滾輪'
                    lst_result.append('1.1')
            else:
                s += ' 不更換滾輪'
                lst_result.append('')
            s += '\n'
            if o_type == 'lenti':
                s += ' (2) pitch=' + str(o_pitch) + ' ' + str(l_pitch) + ' roller=' + str(o_roller) + ' ' + str(
                    o_roller) + ' => '
                if o_pitch != l_pitch or o_roller != l_roller:
                    s += ' 1.2 更換結構板滾輪'
                    lst_result.append('1.2')
                else:
                    s += ' 不更換'
                    lst_result.append('')
            else:
                s += ' (2) 不是lenti'
                lst_result.append('')
            s += '\n'
            s += ' (3) Do_Molds: ' + df_ret.iloc[i, n3].replace(';', ' ')
            if Opt_func.list_index(df_ret.iloc[i, n3].split(';'), l_moldcode) < 0:
                s += ' Do_Molds沒有相同的模頭'
                if len(set(o_domold.split(';')) & set(lst_mold_canuse)) > 0:
                    s += ' Do_Molds有在尚未使用的模頭清單中 => 2.1 更換模頭'
                    lst_result.append('2.1')
                else:
                    s += ' Do_Molds沒有在尚未使用的清單中 => 訂單無法派工'
                    lst_result.append('')
            else:
                s += ' Do_Molds有相同的模頭 => 不更換模頭'
                lst_result.append('')
            s += '\n'
            if o_type == 'plate':
                s += ' (4) mfg_width=' + str(l_mfgwidth) + ' order_width+50=' + str(o_width + 50) + ' => '
                if l_mfgwidth < o_width + 50:
                    s += '2.2 平板模頭_提高製造寬度'
                    lst_result.append('2.2')
                else:
                    s += '不提高製造寬度'
                    lst_result.append('')
            else:
                s += ' (4) 不是plate'
                lst_result.append('')
            s += '\n'
            if o_type == 'lenti':
                if lst_result[1] == '':
                    s += ' (5) mfg_width=' + str(l_mfgwidth) + ' order_width+70=' + str(o_width + 70) + ' => '
                    if l_mfgwidth < o_width + 70:
                        s += ' 1.2 更換結構板滾輪 2.3 結構板模頭_提高製造寬度'
                        lst_result.append('1.2,2.3')
                    else:
                        s += ' 不提高製造寬度'
                        lst_result.append('')
                else:
                    s += ' (5) 步驟1,2不成立'
                    lst_result.append('')
            else:
                s += ' (5) 不是lenti'
            s += '\n'
            s += ' (6) ' + l_composition + ' to ' + o_composition + ' => '
            if o_composition[:2] == 'T6':
                if l_composition[:2] == 'T6' or l_composition == '0%':
                    s += ' 不清料桶'
                    lst_result.append('')
                else:
                    s += ' 3.1 清料桶_降低比例配方'
                    lst_result.append('3.1')
            else:
                p1 = int(o_composition.replace('%', ''))
                if l_composition[:2] == 'T6':
                    s += ' 不清料桶'
                    lst_result.append('')
                else:
                    p2 = int(l_composition.replace('%', ''))
                    if p1 < p2:
                        s += ' 3.1 清料桶_降低比例配方'
                        lst_result.append('3.1')
                    else:
                        s += ' 不清料桶'
                        lst_result.append('')

            if bShow:
                print(str(i + 1) + '. order code:', df_ret.iloc[i, 0], 'do_Molds:', df_ret.iloc[i, n3].split(';'))
                print(' 停機比對測試')
                print(s)

            ret = list(set(list(filter(None, ','.join(lst_result).split(',')))))  # 組成list, 移除空的, 用set移除重複, 最後排序
            ret.sort()
            fbtype = frontblocktype(line_id, block_id, lst_block)
            if len(ret) > 0 or len(ret) == 0 and fbtype == 'S':
                df_ret.loc[i, 'dispatch_type'] = 'NS'
                df_ret.loc[i, 'dispatch_step'] = ','.join(ret)
                if bShow: print(' test result:', ret, 'NS(Need Stop)')
            else:
                lst_result = []
                s = ''
                pre_orderwidth = l_width  # l_mfgwidth - (50 if o_type == 'plate' else 70)
                pre_orderthickness = l_thickness
                s += ' (1)' + str(pre_orderwidth) + ' ' + str(o_width)
                t_waste = 0
                if pre_orderwidth != o_width:
                    s += ' 0.1 生產中調整＿產品寬度切換'
                    if t_waste < 150: t_waste = 150
                    lst_result.append('0.1')
                else:
                    s += ' 不調整'
                    lst_result.append('')
                s += '\n'
                s += ' (2)' + str(pre_orderthickness) + ' ' + str(o_thickness)
                if pre_orderthickness != o_thickness:
                    s += ' 0.2 生產中調整＿產品厚度改變'
                    if t_waste < 300: t_waste = 300
                    lst_result.append('0.2')
                else:
                    s += ' 不調整'
                    lst_result.append('')
                s += '\n'
                s += ' (3)' + str(l_composition) + ' ' + str(o_composition)
                if l_composition != 'T6-0%' and o_composition != 'T6-0%':
                    p1 = int(o_composition.replace('%', ''))
                    p2 = int(l_composition.replace('%', ''))
                    if p1 > p2:
                        s += ' 0.3 直接切換料＿提高比例配方'
                        lst_result.append('0.3')
                    else:
                        s += ' 不切換料'
                        lst_result.append('')
                else:
                    s += ' 不切換料'
                    lst_result.append('')
                s += '\n'
                s += ' (4)' + str(l_composition) + ' ' + str(o_composition)
                if l_composition != o_composition and (l_composition == 'T6-0%' or o_composition == 'T6-0%'):
                    s += ' 0.4 生產中換料(T6-0%)'
                    if t_waste < 300: t_waste = 300
                    lst_result.append('0.4')
                else:
                    s += ' 不切換料'
                    lst_result.append('')
                if bShow:
                    print(' 不停機比對測試')
                    print(s)

                ret = list(set(list(filter(None, lst_result))))
                ret.sort()
                if len(ret) > 0:
                    ret2 = ret.copy()
                    df_ret.loc[i, 'dispatch_type'] = ''
                    df_ret.loc[i, 'dispatch_step'] = ','.join(ret)
                    if bShow: print(' test result:', ret, ' => 進行不停機廢料量比對測試')
                    lst_result = []
                    s = ''
                    base_waste = df_ret.iloc[i, n5] * df_ret.iloc[i, n6] * df_ret.iloc[i, n7] * df_ret.iloc[
                        i, n8] / 10 ** 6  # l*h*d*q
                    p_waste = (l_mfgwidth - df_ret.iloc[i, n4]) * base_waste
                    a_waste = t_waste + p_waste
                    if o_type == 'plate':
                        s += ' (1) A ' + str(a_waste) + ' = ' + str(t_waste) + ' + ' + str(p_waste)
                        w_waste = 50 * base_waste
                        b_waste = w_waste + 1650
                        s += ', B ' + str(b_waste) + ' = ' + str(w_waste) + ' + 1650'
                        if a_waste > b_waste:
                            s += '2.4 平板模頭_降低製造寬度'
                            lst_result.append('2.4')
                        else:
                            s += '不降低製造寬度'
                            lst_result.append('')
                    else:
                        s += ' (2) A ' + str(a_waste) + ' = ' + str(t_waste) + ' + ' + str(p_waste)
                        w_waste = 70 * base_waste
                        b_waste = w_waste + 2150
                        s += ', C ' + str(b_waste) + ' = ' + str(w_waste) + ' + 2150'
                        if a_waste > b_waste:
                            s += '2.5 結構板模頭_降低製造寬度'
                            lst_result.append('2.5')
                        else:
                            lst_result.append('')
                    if bShow:
                        print(' 不停機廢料量比對測試')
                        print(s)

                    ret = list(set(list(filter(None, lst_result))))
                    if len(ret) > 0:
                        df_ret.loc[i, 'dispatch_type'] = 'NS'
                        df_ret.loc[i, 'dispatch_step'] = ','.join(ret)
                        if bShow: print(' test result:', ret, 'NS(Need Stop)')
                    else:
                        df_ret.loc[i, 'waste_hour'] = a_waste
                        df_ret.loc[i, 'dispatch_type'] = 'IT'
                        df_ret.loc[i, 'dispatch_step'] = ','.join(ret2)
                        if bShow: print(' test result:', ret2, 'IT(In-situ Tunning)')
                else:
                    df_ret.loc[i, 'dispatch_type'] = 'DP'
                    df_ret.loc[i, 'dispatch_step'] = ''
                    if bShow: print(' test result:', ret, 'DP(Direct Production)')
            # tune_step, tune_x, tune_w
            if df_ret.loc[i, 'dispatch_type'] != 'DP':
                # df_ret.loc[i, 'tune_step'], df_ret.loc[i, 'tune_x'], df_ret.loc[i, 'tune_w'] = TIP_result(df_ret.loc[i, 'dispatch_type'], df_ret.loc[i, 'dispatch_step'], o_type, o_width, o_thickness, o_composition, l_width, l_thickness, l_composition)
                if df_ret.loc[i, 'dispatch_type'] == 'IT': df_ret.loc[i, 'waste_hour'] = df_ret.loc[i, 'waste_hour'] / (df_ret.loc[i, 'Production_Hours'] + df_ret.loc[i, 'tune_x'])
            if bShow: print('')
        if bShow: print(df_ret)
    return df_ret

def TIP_result(dispatch_type, dispatch_step, o_type, o_width, o_thickness, o_composition, l_width, l_thickness, l_composition, lst_step, lst_hour):
    lst = list(filter(None, dispatch_step.split(',')))
    tune_x = 0
    if dispatch_type == 'NS':
        # if len(set(lst) & set(lst_step[4:12])) > 0:  #1.1', '1.2', '2.1', '2.2', '2.3', '2.4', '2.5', '3.1'
        if len(set(lst) & set(lst_step[4:11])) == 0:
            if len(lst) > 0: tune_x = float(lst_hour[11])
        else:
            for v in range(4, 11):
                if Opt_func.list_index(lst, lst_step[v]) >= 0: tune_x += float(lst_hour[v])
        tune_x += 7
        if o_type == 'plate':
            tune_w = 1500
            tune_step = '4.1,5.1'
        else:
            tune_w = 2000
            if len(set(lst) & {'1.2', '2.3'}) > 0:
                tune_step = '4.1,5.0,5.2'
            else:
                tune_step = '4.1,5.2'

        if l_width != o_width: tune_step += ',4.2'
        if l_thickness != o_thickness: tune_step += ',4.3'
        if l_composition != 'T6-0%' and o_composition != 'T6-0%':
            p1 = int(o_composition.replace('%', ''))
            p2 = int(l_composition.replace('%', ''))
            if p1 > p2:  tune_step += ',4.4'
        if l_composition != o_composition and (l_composition == 'T6-0%' or o_composition == 'T6-0%'): tune_step += ',4.5'
        # sort tune step
        if tune_step != '':
            lst_tmp = tune_step.split(',')
            lst_tmp.sort()
            tune_step = ','.join(lst_tmp)
    else:  # IT
        for v in range(4):
            if Opt_func.list_index(lst, lst_step[v]) >= 0:
                if tune_x < float(lst_hour[v]): tune_x = float(lst_hour[v])
        tune_w = tune_x * 300
        tune_step = '0.0'
    return tune_step, tune_x, tune_w

def changemold(sline, new_moldcode, df_lines, df_molds):
    mold_id = df_molds[df_molds['mold_code'] == new_moldcode].index[0]
    new_moldno = df_molds.loc[mold_id, 'mold_no']
    new_moldmfgwidth = df_molds.loc[mold_id, 'width_max']

    line_id = df_lines[df_lines['line_name'] == sline].index[0]
    old_moldcode = df_lines.loc[line_id, 'mold_code']
    df_lines.loc[line_id, 'mold_code'] = new_moldcode
    df_lines.loc[line_id, 'mold_no'] = new_moldno
    df_lines.loc[line_id, 'mfg_width'] = new_moldmfgwidth

    df_molds.loc[df_molds[df_molds['mold_code'] == old_moldcode].index[0], 'Usage'] = ''
    df_molds.loc[mold_id, 'Usage'] = sline
    return df_lines, df_molds