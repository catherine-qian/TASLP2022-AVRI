import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
import random

def dataswap(Xdata):
    print('SWAP the GCC-PHAT and STFT data at 10% of the total data')
    # data = hdf5storage.loadmat('/Volumes/MyPassport/TASLP/gcc-mel-face(0db).mat')
    # data = hdf5storage.loadmat('/home/zhengdong/code/matlabcode/our-dataset/test/data/gcc-mel-face(0db).mat')
    # Xdata = data['Xdata']
    L = len(Xdata)  # length

    slots = 3, 15, 25, 30  # number of frames
    ssum = sum(slots)
    T = round(L * 0.1 / ssum / 2)

    total = 0
    for t in range(T):
        for slen in slots:
            st1 = random.randint(0, L - slen)  # start time 1
            st2 = random.randint(0, L - slen)  # start time 2

            mat1 = Xdata[st1:st1 + slen, :]
            Xdata[st1:st1 + slen, :] = Xdata[st2:st2 + slen, :]
            Xdata[st2:st2 + slen, :] = mat1  # swap
            total = total + slen * 2
    print('randomly swap {}% of the original data'.format(total*100/L))
    return Xdata


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    return {'Total': total_num, 'Trainable': trainable_num}

def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


def ACC(MAE, th):
    return sum([error <= th for error in MAE]) / len(MAE)

class MyDataloaderClass(Dataset):

    def __init__(self, X_data, Y_data):
        self.x_data = X_data
        self.y_data = Y_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



def errorcompute(Y_pred_t, Yte, vad, face, savename):

    # ------------ error evaluate   ----------
    Y_pred = np.array(Y_pred_t)
    Yte, Y_pred = Yte.reshape(-1, 360), Y_pred.reshape(-1, 360)
    vad, face = vad.reshape(-1, 1).squeeze(), face.reshape(-1, 1).squeeze()

    gt = np.argmax(Yte, axis=1) # ground truth DoA
    pred = np.argmax(Y_pred, axis=1) # DoA prediction
    er = angular_distance_compute(gt, pred) # error computation


    MAE1 = sum(er[vad]) / len(er[vad])
    ACC1 = ACC(er[vad], 10)

    MAEstd=np.std(er[vad])
    print('MAE std={:.2f}'.format(MAEstd))
    if savename is not None:

        ervad=er[vad]

        MAEacc, MAEaccstd = np.mean(ervad[ervad<=10]), np.std(ervad[ervad<=10])
        sio.savemat("MAE-"+savename+".mat",{"er":er, "vad":vad,'gt':gt,'pred':pred})
        print("test set: MAE={:.2f}+{:.2f}| MAEacc={:.2f}+{:.2f}".format(MAE1, MAEstd, MAEacc, MAEaccstd))


    erfacevad=er[vad*face]
    MAE_face_vad = sum(erfacevad) / len(erfacevad)
    ACC_face_vad = ACC(erfacevad, 10)

    er0facevad=er[vad*~face]
    MAE_without_face_vad = sum(er0facevad) / len(er0facevad)
    ACC_without_face_vad = ACC(er0facevad, 10)


    return ACC1, MAE1, ACC_face_vad, MAE_face_vad, ACC_without_face_vad, MAE_without_face_vad



