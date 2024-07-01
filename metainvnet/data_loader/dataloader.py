from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, im1, im2, im3):
        super(DatasetLoader, self).__init__()
        self.im1 = im1
        self.im2 = im2
        self.im3 = im3

    def __getitem__(self, index):
        return self.im1[index,], self.im2[index,], self.im3[index,]

    def __len__(self):
        return self.im1.size()[0]

