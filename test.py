import Opt_SL_Run
import warnings

warnings.filterwarnings('ignore')
#Opt_SL_Run.Run_Learning('d:\\opt_web', 'TaskName1', 'StageLearning(4)')
Opt_SL_Run.Run_Learning('/aidata/DIPS', 'Task1', 'StageLearning(3)')


# ===  Test IMP Sector  ===

# import numpy as np
# import pandas as pd
# import Opt_RS_Report
#
# lst_block = np.load('d:\\opt_Web\\darray.npy', allow_pickle=True)
# lst_weekday = ['週一', '週二', '週三', '週四', '週五', '週六', '週日']
# max_linearray = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
#
# begin_day = '2021-06-01'
# end_day = '2021-12-31'
# maxline_count = 4
#
# df_lines = pd.read_csv('d:\\opt_Web\\ldata.csv')
# df_lines['Line Begin'] = df_lines['Line Begin'].astype('datetime64[ns]')
# df_lines.fillna('', inplace=True)
#
# df_sector, df_LRation, lst_PHData = Opt_RS_Report.ImpSector_Sheet(lst_block, lst_weekday, max_linearray, begin_day, end_day, maxline_count, df_lines)



# from threading import Thread
#
# def runA():
#     while True:
#         print('A')
#
# def runB():
#     while True:
#         print('B')
#
# if __name__ == "__main__":
#     t1 = Thread(target = runA)
#     t2 = Thread(target = runB)
#     t1.setDaemon(True)
#     t2.setDaemon(True)
#     t1.start()
#     t2.start()
#     while True:
#         pass

#
# import os
#
# print(os.name)
