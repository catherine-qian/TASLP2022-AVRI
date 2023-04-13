import torch
import numpy as np
import time
from torch.autograd import Variable
import hdf5storage
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import importlib
import fun

torch.manual_seed(7)  # For reproducibility across different computers
torch.cuda.manual_seed(7)

starttime= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hri localization experiments")
    parser.add_argument('-gpuidx', type=int, default=0)
    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-model', type=str, default='gccmlp')
    parser.add_argument('-frame', type=int, default=64, help="N of frames (32 fps)")  # tracking frame length
    parser.add_argument('-ep', type=int, default=20)  # epoch number
    parser.add_argument('-snr', type=int, default=None)  # epoch number
    parser.add_argument('-lr', type=float, default=0.001)  # laearning rate
    parser.add_argument('-train', type=str, default=None)  # use all noise to train?
    parser.add_argument('-gt', type=bool, default=True)  #
    parser.add_argument('-vadon', type=bool, default=True)  #  evaluate only at VAD frames
    parser.add_argument('-savename', type=bool, default=False)  #  save results
    parser.add_argument('-datapath', type=str, default='data', help='data directory')


    args = parser.parse_args()

    # define parameters
    TRAIN_RATIO = 0.7
    BATCH_SIZE = args.bs
    gpuidx = args.gpuidx
    frame_len = args.frame
    epoch_num = args.ep  # epoch
    datapath = args.datapath
    device = torch.device("cuda:{}".format(gpuidx) if torch.cuda.is_available() else 'cpu')

    print("Experiments - TMM2023 single-speaker tracking")
    print("batch size:{}  gpuidx:{}  device:{}".format(BATCH_SIZE, gpuidx, device))

def training_process(epoch):
    print("------------------------ start training "+args.model+"  "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"-----------------------")

    training(epoch, train_loader)
    # train_acc, train_mae, ACC_face, MAE_face, ACC_without_face, MAE_without_face = testing(Xtr, Ytr, Vtr, True, facetr)
    # print("train ep{}".format(epoch)+": acc=", train_acc, "mae=", train_mae)


def testing_processing(epoch):
    # print("------------start test ---------")
    savename = args.model+str(args.snr) if args.savename else None
    train_acc, train_mae, ACC_face, MAE_face, ACC_without_face, MAE_without_face = testing(Xtr, Ytr, Vtr, facetr)
    test_acc, test_mae, ACC_face, MAE_face, ACC_without_face, MAE_without_face = testing(Xte, Yte, Vte, facete,savename=savename)

    print("train ep{:3d}".format(epoch)+":         acc={:.4f} mae={:.2f}".format(train_acc, train_mae))
    print("test  ep{:3d}".format(epoch)+":         acc={:.4f} mae={:.2f}".format(test_acc, test_mae))
    print("test(with face) :    acc={:.4f} mae={:.2f}".format(ACC_face, MAE_face))
    print("test(without face) : acc={:.4f} mae={:.2f}".format(ACC_without_face, MAE_without_face))

    train_acc_matric[epoch] = train_acc
    train_mae_matric[epoch] = train_mae

    test_acc_matric[epoch] = test_acc
    test_mae_matric[epoch] = test_mae

    test_acc_matric_with_face[epoch] = ACC_face
    test_acc_matric_without_face[epoch] = ACC_without_face

    test_mae_matric_with_face[epoch] = MAE_face
    test_mae_matric_without_face[epoch] = MAE_without_face


def training(epoch, train_loader):
    print('start training, epoch = ', epoch)
    model.train()
    Y_pred_t = []
    tar = []
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        inputs, target = Variable(data).type(torch.FloatTensor).to(device), Variable(target).type(torch.FloatTensor).to(
            device)
        y_pred = model.forward(inputs)  # return the predicted angle
        loss = criterion(y_pred.double(), target.double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Y_pred_t.extend(y_pred.cpu().detach().numpy())  # in CPU
        tar.extend(target.cpu().detach().numpy())
        if batch_idx % 1000 == 0:
            print("iteration", batch_idx, "loss=", loss.item())
    torch.cuda.empty_cache()


def testing(Xte, Yte, Vte, face_te, savename=None):
    model.eval()
    # print('start testing')
    Y_pred_t = []
    for ist in range(0, len(Xte), BATCH_SIZE):
        ied = np.min([ist + BATCH_SIZE, len(Xte)])  # select ending index
        inputs = Variable(torch.from_numpy(Xte[ist:ied])).type(torch.FloatTensor).to(device)
        output = model.forward(inputs)
        Y_pred_t.extend(output.cpu().detach().numpy())  # in CPU

    # ------------ error evaluate   ---------
    ACC1, MAE1, ACC_face, MAE_face, ACC_without_face, MAE_without_face = fun.errorcompute(Y_pred_t, Yte, Vte, face_te, savename)
    torch.cuda.empty_cache()
    return ACC1, MAE1, ACC_face, MAE_face, ACC_without_face, MAE_without_face



criterion = torch.nn.MSELoss(reduction='sum')  # loss
model = importlib.import_module("finalmodel.{}".format(args.model)).Model(time_frame=frame_len).to(device)
print("===========model===============")
print(model)
fun.get_parameter_number(model)

optimizer = torch.optim.Adam(model.parameters(), args.lr)
best_acc = 0


snr_name_list =['gcc_mel_face'] if args.snr is None else ['gcc-mel-face({}db)'.format(args.snr)]
print("load data: {}".format(snr_name_list[0]))

data = hdf5storage.loadmat(os.path.join(datapath, snr_name_list[0]))

def processdata(data, args):
    X = data['Xtr']  # features
    vad = data['vad'] > 0  # VAD label
    facedata = data['face']
    face = np.sum(facedata[:, 0, :], axis=1) > 0

    if args.gt: # use ground truth
        data = hdf5storage.loadmat(os.path.join(datapath, 'ground_truth'))  # vad label and trajectory
        Y = data['Ytr']  # one-hot DoA
        Z = data['Ztr']  # posterior

    # model --- stft, gccmlp, avmlp, acrnn, avcrnn, avcmat
    if args.model == 'gccmlp':
        X = X[:, 0:6, :]
        X = X.reshape(-1, X.shape[1] * X.shape[2])
    elif args.model == 'avmlp':
        X = X[:, 0:6, :]
        X = np.concatenate((X, facedata), axis=1)
        X = X.reshape(-1, X.shape[1] * X.shape[2])
    elif args.model == 'acrnn':
        X = X
    elif 'av'in args.model and 'mlp' not in args.model:
        X = np.concatenate((X, facedata), axis=1)
    else:
        print("error! with model name")

    if 'mlp' not in args.model and 'stft' not in args.model: # tracking
        frame_all = X.shape[0]
        print("number of frames {}".format(frame_all))
        frame_nb = (int)(frame_all // frame_len)
        X = X[:frame_nb * frame_len, :]
        X = X.reshape(frame_nb, frame_len, X.shape[1], X.shape[2]) # (15648, 64, 10, 21)
        Y = Y[:frame_nb * frame_len, :]
        Y = Y.reshape(frame_nb, frame_len, -1)  # (15648, 64, 360)
        Z = Z[:frame_nb * frame_len, :]
        Z = Z.reshape(frame_nb, frame_len, -1)  # (15648, 64, 360)
        vad = vad[:frame_nb * frame_len, :]
        vad = vad.reshape(frame_nb, frame_len, -1) # (15648, 64, 1)
        face = face[:frame_nb * frame_len]
        face = face.reshape(frame_nb, frame_len, -1) # (15648, 64, 1)


    # split dataset into train/test set
    N = Y.shape[0]
    train_len = int(args.bs*(N * TRAIN_RATIO//args.bs)) # training length

    Xtr, Ytr, Ztr, Vtr, facetr = X[:train_len], Y[:train_len], Z[:train_len], vad[:train_len], face[:train_len]
    Xte, Yte, Zte, Vte, facete = X[train_len:], Y[train_len:], Z[train_len:], vad[train_len:], face[train_len:]
    return Xtr, Ytr, Ztr, Vtr, facetr, Xte, Yte, Zte, Vte, facete

Xtr, Ytr, Ztr, Vtr, facetr, Xte, Yte, Zte, Vte, facete = processdata(data, args) # split train/test set


print(args)
if args.train is not None:  # use all data to train
    snr_name_list = ['gcc-mel-face(40db)', 'gcc-mel-face(20db)', 'gcc-mel-face(0db)', 'gcc-mel-face(-20db)']
    # snr_name_list = ['gcc-mel-face(-20db)']
    for i in range(len(snr_name_list)):
        data = hdf5storage.loadmat(os.path.join(datapath, snr_name_list[i]))
        _, Xtr0, Ytr0, Ztr0, Vtr0, facetr0, *_ = processdata(data, args)  # split train/test set
        Xtr, Ytr, Ztr = np.concatenate((Xtr, Xtr0), axis=0), np.concatenate((Ytr, Ytr0), axis=0), np.concatenate((Ztr, Ztr0), axis=0)
        Vtr, facetr = np.concatenate((Vtr, Vtr0), axis=0), np.concatenate((facetr, facetr0), axis=0)

print(Xtr.shape)

# give train data to dataloader
train_loader_obj = fun.MyDataloaderClass(Xtr, Ztr)
train_loader = DataLoader(dataset=train_loader_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



#------------------------ Training & Testing --------------
train_acc_matric = np.zeros(epoch_num)
train_mae_matric = np.zeros(epoch_num)
test_acc_matric = np.zeros(epoch_num)
test_mae_matric = np.zeros(epoch_num)

test_acc_matric_with_face = np.zeros(epoch_num)
test_acc_matric_without_face = np.zeros(epoch_num)
test_mae_matric_with_face = np.zeros(epoch_num)
test_mae_matric_without_face = np.zeros(epoch_num)

for epoch in range(epoch_num):


    training_process(epoch)
    testing_processing(epoch)

    if test_acc_matric[epoch] > best_acc:
        best_acc = test_acc_matric[epoch]
        # torch.save(model, 'best'+args.model+str(args.snr)+'.pkl')
        # print("best model save successfully")

    print(starttime)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # print("################  end  ###########################")