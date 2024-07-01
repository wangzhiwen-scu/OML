import argparse
class get_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='CT img Recon')
        self.parser.add_argument('--log', default=False, help='write output to file')
        self.parser.add_argument('--phase', type=str, default='tr', help='tr')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--data_type', default='dcm', help='dcm, png')
        # Training Parameters
        self.parser.add_argument('--epoch', type=int, default=10, help='#epoch ')
        self.parser.add_argument('--tr_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--layers', type=int, default=3, help='net layers')
        self.parser.add_argument('--deep', type=int, default=17, help='depth')
        self.parser.add_argument('--img_size', default=[512,512], help='image size')
        self.parser.add_argument('--sino_size', nargs='*', default=[360,600], help='sino size')
        self.parser.add_argument('--poiss_level',default=1e5, help='Poisson noise level')
        self.parser.add_argument('--gauss_level',default=[0.05], help='Gaussian noise level')
        self.parser.parse_args(namespace=self)
        self.tr_dir = 'train_data_path'  # training data directory