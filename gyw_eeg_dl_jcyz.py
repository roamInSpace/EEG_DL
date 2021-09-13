#深度学习之交叉验证多种网络综合评估
#步骤: 
#     1、读入配置，设置表头（多个cross、nn通用）
#     2、每个cross读入train和test（多个nn通用）
#     3、声明网络、训练、测试
import torch
import gyw
import xlwt
import os
import copy
import numpy as np

#region 初始化
G_cfg = gyw.Read_CFG('config.cfg')
sample_left = gyw.W2h(['按样本'] + G_cfg.labels)
sample_fornt = [G_cfg.labels]
cases_fornt = [G_cfg.labels]
#endregion

for i_cross in range(G_cfg.cross):
    (train_loader, train_weights, test_loaders, cases_left) = gyw.Load_all(G_cfg, i_cross)
    for i_nn, nn_i in enumerate(G_cfg.nns):
        print('')
        path_out = G_cfg.path_out + nn_i + '/cross' + str(i_cross+1) + '/'
        if not(os.path.exists(path_out)):
            os.makedirs(path_out)

        print('cross' + str(i_cross+1) + '_' + nn_i + 
              '------------------------------------------------------------'
        )

        nn_min = None

        #region 训练
        if(os.path.exists(path_out + 'net.pkl') and os.path.exists(path_out + G_cfg.loss)):
            print('已存在' + 'cross' + str(i_cross+1) + '_' + nn_i)
            nn_min = torch.load(path_out + 'net.pkl').to(G_cfg.device)
        else:        
            loss_workbook = xlwt.Workbook(encoding = 'utf-8')
            loss_xls = []
            
            eeg_nn = gyw.Eeg_nn(nn_i, G_cfg.channel, G_cfg.label_num)
            eeg_nn = eeg_nn.to(G_cfg.device)
            optimizer = torch.optim.Adam(eeg_nn.parameters(), lr=G_cfg.lr)
            loss_func = torch.nn.CrossEntropyLoss(weight=train_weights)
            loss_func.to(G_cfg.device)

            loss_min = float('inf')
            epoch_min = 0
            i_epoch = 0
            while(True):
                i_epoch += 1

                eeg_nn.train()
                for (x_i, y_i) in train_loader:
                    x_i = x_i.to(G_cfg.device)
                    y_i = y_i.to(G_cfg.device)
                    output = eeg_nn.forward(x_i)
                    loss = loss_func(output, y_i)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                eeg_nn.eval()

                loss_total = 0
                for (x_i, y_i) in train_loader:
                    x_i = x_i.to(G_cfg.device)
                    y_i = y_i.to(G_cfg.device)
                    output = eeg_nn.forward(x_i)
                    loss = loss_func(output, y_i)
                    loss_total += loss.item()
                loss = loss_total/len(train_loader)

                print('%.6f    '%loss + 
                    'cross' + str(i_cross+1) + '_' + nn_i + '_epoch' + str(i_epoch) 
                )
                loss_xls.append(['%.6f'%loss])

                if(loss < loss_min):
                    loss_min = loss
                    epoch_min = i_epoch
                    nn_min = copy.deepcopy(eeg_nn)
                else:
                    if(i_epoch-epoch_min >= G_cfg.stop):
                        break
            
            torch.save(nn_min, path_out + 'net.pkl')
            gyw.save_classify([], [], loss_xls, 
            loss_workbook, 1, path_out, G_cfg.loss, 
            )
        #endregion

        #region 测试
        sample_workbook = xlwt.Workbook(encoding = 'utf-8')
        cases_workbook = xlwt.Workbook(encoding = 'utf-8')
        sample_classify = np.zeros((G_cfg.label_num, G_cfg.label_num))
        cases_classify = np.zeros((len(test_loaders), G_cfg.label_num))

        nn_min.eval()
        for i_case, test_loader in enumerate(test_loaders):
            for (x_i, y_i) in test_loader:
                x_i = x_i.to(G_cfg.device)
                y_i = y_i.to(G_cfg.device)
                predict = nn_min.predict(x_i)
                for (y_i_i, predict_i) in zip(y_i, predict):

                    #对每一样本的实际与预测累加
                    sample_classify[y_i_i, predict_i] = sample_classify[y_i_i, predict_i] + 1
                    
                    #记录此病例的预测
                    cases_classify[i_case, predict_i] += 1
        cases_xls = gyw.Percent1(cases_classify)
        sample_xls = gyw.Percent2(sample_classify)

        gyw.save_classify(sample_left, sample_fornt, sample_xls, sample_workbook, 1, 
        path_out, G_cfg.sample, 
        )

        gyw.save_classify(cases_left, cases_fornt, cases_xls, cases_workbook, 1, 
        path_out, G_cfg.cases, 
        )
        print('测试完成')
        #endregion
