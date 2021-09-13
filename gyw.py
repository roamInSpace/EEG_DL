from configparser import ConfigParser
import numpy as np
import torch
import torch.nn as nn
import xlrd
import time

class Read_CFG:
    def __init__(self, _name):
        cp = ConfigParser()
        cp.read(_name)

        self.path_data = cp.get('mysql', 'path_data')
        self.path_out = cp.get('mysql', 'path_out')
        torch.cuda.set_device(cp.getint('mysql', 'device'))
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('使用GPU运算')
        else:
            print('使用CPU运算')
        self.interval = cp.getint('mysql', 'interval')

        self.nns = cp.get('mysql', 'nns').split('|')
        self.banchSize = cp.getint('mysql', 'banchSize')
        self.lr = cp.getfloat('mysql', 'lr')
        self.cross = cp.getint('mysql', 'cross')
        self.stop = cp.getint('mysql', 'stop')

        self.labels = cp.get('mysql', 'labels').split('|')
        self.label_num = len(self.labels)
        self.channel = cp.getint('mysql', 'channel')
        self.Hz = cp.getint('mysql', 'Hz')
        self.timeUnit = cp.getint('mysql', 'timeUnit')
        self.timeStep = cp.getint('mysql', 'timeStep')
        self.denoise = cp.getint('mysql', 'denoise')

        self.index = cp.get('mysql', 'index')
        self.sample = cp.get('mysql', 'sample')
        self.cases = cp.get('mysql', 'cases')
        self.loss = cp.get('mysql', 'loss')

def Load_all(G_cfg, cross_test):
    last_time = time.time()
    train_datas = torch.Tensor().type(torch.FloatTensor)
    train_labels = torch.Tensor().type(torch.LongTensor)
    train_weights = [0 for _ in range(G_cfg.label_num)]
    test_loaders = []
    cases_left = [[]]
    for i_cross in range(G_cfg.cross):
        xls = xlrd.open_workbook(G_cfg.index + 'cross' + str(i_cross+1) + '.xls')
        xls = xls.sheets()[0]
        for i_case in range(xls.nrows):
            now_time = time.time()
            if(now_time-last_time > G_cfg.interval):
                print(time.asctime(time.localtime(now_time)))
                last_time = now_time

            TF = True
            for i_label, lable_i in enumerate(G_cfg.labels):
                if(xls.cell(i_case, 1).value == lable_i):
                    datas = Eeg_cut(G_cfg.path_data + xls.cell(i_case, 0).value + '.NED.txt', 
                    G_cfg.Hz, G_cfg.timeUnit, G_cfg.timeStep, G_cfg.denoise, 
                    )
                    
                    weight = len(datas)
                    datas = torch.Tensor(datas).type(torch.FloatTensor)
                    labels = torch.Tensor([i_label for _ in range(weight)]).type(torch.LongTensor)

                    if(i_cross == cross_test):
                        cases_left.append(xls.row_values(i_case))
                        test_loader = torch.utils.data.DataLoader(
                            dataset=torch.utils.data.TensorDataset(datas, labels), 
                            batch_size=G_cfg.banchSize, 
                        )
                        test_loaders.append(test_loader)
                    else:
                        train_weights[i_label] += weight
                        train_datas = torch.cat([train_datas, datas], dim=0)
                        train_labels = torch.cat([train_labels, labels], dim=0)
                    
                    TF = False
                    break

            if(TF):
                exit('标签错误：cross' + str(i_cross) + '_' + xls.cell(i_case, 1).value)
    
    print('')
    print('cross' + str(cross_test+1) + '：' + str(train_weights))
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(train_datas, train_labels),
        batch_size=G_cfg.banchSize,
        shuffle=True,
    )
    train_weights = Weight(train_weights)
    
    return(train_loader, train_weights, test_loaders, cases_left)

def Eeg_cut(path, Hz, timeUnit, timeStep, denoise):
    eeg = np.loadtxt(path)
    time_s = int(eeg.shape[0] / Hz)
    epoches = int((time_s - timeUnit) / timeStep + 1)
    eeg = eeg[0 : time_s * Hz, :]

    datas = []

    for i_epoch in range(epoches):
        start = i_epoch * timeStep * Hz
        end = start + timeUnit * Hz
        data = eeg[start : end, :]
        if(data.max() < denoise and data.min() > -denoise):
            datas.append(data.tolist())
    
    if(len(datas) < 10):
        exit(path)

    return datas

def Percent1(n):
    d0 = n.shape[0] + 1
    d1 = n.shape[1] + 1
    l = [['' for i1 in range(d1)] for i0 in range(d0)]
    for i1 in range(len(n)):
        for i2 in range(len(n[i1])):
            t = n[i1, i2] / sum(n[i1, :])
            l[i1][i2] = '%.4f'%t

    return(l)

def Percent2(n):
    d0 = n.shape[0] + 1
    d1 = n.shape[1] + 1
    l = [['' for i1 in range(d1)] for i0 in range(d0)]
    correct = 0
    for i1 in range(len(n)):
        for i2 in range(len(n[i1])):
            l[i1][i2] = str(int(n[i1, i2]))
    for i1 in range(min(n.shape)):
        correct += n[i1, i1]
        t = n[i1, i1] / sum(n[i1, :])
        l[i1][d1-1] = '%.4f'%t

        if(sum(n[:, i1]) == 0):
            l[d0-1][i1] = '0.0000'
        else:
            t = n[i1, i1] / sum(n[:, i1])
            l[d0-1][i1] = '%.4f'%t
    t = correct/n.sum()
    l[d0-1][d1-1] = '%.4f'%t

    return(l)
        
def W2h(w):
    h = []
    for w_i in w:
        h.append([w_i])
    return(h)

def Weight(w):
    w = np.array(w)
    w = sum(w) / w
    w = torch.from_numpy(w).float()
    return w

def Eeg_nn(name, channel, label_num):
    result = None

    if(name == 'C'):
        result = EEG_CNN(channel, label_num)
    elif(name == 'L'):
        result = EEG_LSTM(channel, label_num)
    elif(name == 'C1'):
        result = EEG_CNN1(channel, label_num)
    else:
        exit('无' + name + '网络')
    
    return result

def save_classify(left, front, content, workbook, sheet_number, path, name):

    worksheet = workbook.add_sheet(str(sheet_number))
    max1 = 0
    for i_h, h_i in enumerate(left):
        if(len(h_i) > max1):
            max1 = len(h_i)
        for i_w, w_i in enumerate(h_i):
            worksheet.write(i_h, i_w, label=w_i)
    
    max2 = len(front)
    for i_h, h_i in enumerate(front):
        for i_w, w_i in enumerate(h_i):
            worksheet.write(i_h, i_w+max1, label=w_i)


    for i_h, h_i in enumerate(content):
        for i_w, w_i in enumerate(h_i):
            worksheet.write(i_h+max2, i_w+max1, label=w_i)
    
    for _ in range(60):
        try:
            workbook.save(path + name)
            break
        except:
            time.sleep(1)        

class EEG_LSTM(nn.Module):
    def __init__(self, channel, label_num):
        self.channel = channel
        super(EEG_LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=channel,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            dropout=0.5
        )

        self.out = nn.Linear(64, label_num)

    def forward(self, x):
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(x)
        output = self.out(r_out[:, -1, :])
        return output

    def predict(self, x):
        output = self.forward(x)
        output = torch.max(output, 1)[1]
        return output

class EEG_CNN(nn.Module):
    def __init__(self, channel, label_num):
        self.channel = channel
        super(EEG_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (15, 1), 1, (7, 0)), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (9, 1), 1, (4, 0)), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, (1, channel), 1, (0, 0)), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (9, 1), 1, (4, 0)), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.out = nn.Sequential(
            nn.Linear(256 * 20, 1000), 
            nn.ReLU(), 
            nn.Linear(1000, 50), 
            nn.ReLU(), 
            nn.Linear(50, label_num), 
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
    def predict(self, x):
        output = self.forward(x)
        output = torch.max(output, 1)[1]
        return output

class EEG_CNN1(nn.Module):
    def __init__(self, channel, label_num):
        self.channel = channel
        super(EEG_CNN1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (15, 1), 1, (7, 0)), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (9, 1), 1, (4, 0)), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (9, 1), 1, (4, 0)), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 1), 1, (2, 0)), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 1)), 
        )

        self.out = nn.Sequential(
            nn.Linear(16 * 64 * 20, 1000), 
            nn.ReLU(), 
            nn.Linear(1000, 50), 
            nn.ReLU(), 
            nn.Linear(50, label_num), 
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
    def predict(self, x):
        output = self.forward(x)
        output = torch.max(output, 1)[1]
        return output
