import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import xlrd

path = 'D:/GYW_worker/.project/docon_diagnose/.result/result/result/'
G_cross = 5
G_nn = 'L/'
G_epoch = 1
G_color = [(1.00, 0.00, 0.00), 
           (0.75, 0.25, 0.00), 
           (0.50, 0.50, 0.00), 
           (0.25, 0.75, 0.00), 
           (0.00, 1.00, 0.00), 
]

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 19,
}

plt.figure(figsize=(10,10))
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
ax=plt.gca()  #gca:get current axis得到当前轴
ax.spines['right'].set_color('none')#设置图片的右边框为不显示
ax.spines['top'].set_color('none')#设置图片的上边框为不显示
plt.xlim([-0.0, 1.05])
plt.ylim([-0.0, 1.05])
plt.xlabel('False Positive Rate', fontdict=font1)
plt.ylabel('True Positive Rate', fontdict=font1)
#plt.title('ROC of LSTM', fontdict=font1)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
sample_num = np.zeros((3, 3))

for i_cross in range(G_cross):
    xls = xlrd.open_workbook(path + G_nn + 'cross' + str(i_cross+1) + '/sample.xls')
    xls = xls.sheets()[G_epoch-1]
    for i1 in range(2):
        for i2 in range(2):
            sample_num[i1, i2] += int(xls.cell(i1+1, i2+1).value)

    xls = xlrd.open_workbook(path + G_nn + 'cross' + str(i_cross+1) + '/cases.xls')
    xls = xls.sheets()[G_epoch-1]
    y_test = []
    y_score = []
    for i_case in range(xls.nrows-1):
        if(xls.cell(i_case+1, 1).value == 'MCS'):
            y_test.append(0)
        else:
            y_test.append(1)

        y_score.append(float(xls.cell(i_case+1, 3).value))
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_test, y_score) ###计算真正率和假正率

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    roc_auc = auc(fpr,tpr) ###计算auc的值
    aucs.append(roc_auc)

    plt.plot(fpr, tpr, color=G_color[i_cross], lw=2, alpha=0.5, linestyle='--', 
             label='fold%d '%(i_cross+1) + '(AUC = %0.2f)'%roc_auc, 
    ) ###假正率为横坐标，真正率为纵坐标做曲线

sample_num[2, 2] = (sample_num[0, 0] + sample_num[1, 1])/(sample_num[0, 0] + sample_num[0, 1] + sample_num[1, 0] + sample_num[1, 1])
sample_num[0, 2] = sample_num[0, 0]/(sample_num[0, 0] + sample_num[0, 1])
sample_num[1, 2] = sample_num[1, 1]/(sample_num[1, 1] + sample_num[1, 0])
sample_num[2, 0] = sample_num[0, 0]/(sample_num[0, 0] + sample_num[1, 0])
sample_num[2, 1] = sample_num[1, 1]/(sample_num[1, 1] + sample_num[0, 1])
print(sample_num)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', lw=5, 
         label='mean (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='black', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.legend(prop=font2)
plt.show()