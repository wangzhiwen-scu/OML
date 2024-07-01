import sys

sys.path.append('.')
from metainvnet.config import get_config
from metainvnet.model import MetaInvNet
from metainvnet.data_loader import Phantom2dDataset
from metainvnet.trainer import Trainer


if __name__ == '__main__':

    args = get_config()

    # model
    model = MetaInvNet(args)

    # main
    tr_dataset = Phantom2dDataset(args, phase='tr', datadir=args.tr_dir)
    train = Trainer(args, model, tr_dset=tr_dataset)
    train.tr()

    print('[*] Finish!')