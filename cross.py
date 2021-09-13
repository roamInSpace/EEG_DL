import xlrd
import xlwt
import random

G_path = 'D:/GYW_worker/.project/docon_diagnose/.result/'
G_k = 5
G_start = 0
G_num = 3

xls = xlrd.open_workbook(G_path + 'index_all.xls')
xls = xls.sheets()[0]

indexs = []
label_kinds = []

for i_case in range(xls.nrows):
    TF = True
    for i_lable_kind, lable_kind_i in enumerate(label_kinds):
        if(xls.cell(i_case, 1).value == lable_kind_i):
            indexs[i_lable_kind].append(xls.row_values(i_case, start_colx=G_start, end_colx=G_start+G_num))
            TF = False
            break
    if(TF):
        label_kinds.append(xls.cell(i_case, 1).value)
        indexs.append([xls.row_values(i_case, start_colx=G_start, end_colx=G_start+G_num)])

for i_index in range(len(indexs)):
    random.shuffle(indexs[i_index])


for i_k in range(G_k):
    cross_indexs = []

    for i_index in range(len(indexs)):
        num = int(len(indexs[i_index]) / (G_k-i_k))
        for i_case in range(num):
            cross_indexs.append(indexs[i_index][0])
            indexs[i_index].pop(0)

    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('0')

    for i_h, h_i in enumerate(cross_indexs):
        for i_w, w_i in enumerate(h_i):
            worksheet.write(i_h, i_w, label=w_i)
    
    workbook.save(G_path + 'index_cross' + str(i_k+1) + '.xls')