import sys
import argparse
import os
import torch
import pickle

from utils import mkdirs

class BaseOptions(object):
    r"""

    Examples::
        >>> opt = BaseOptions().parse()

    """
    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='label2coco',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--dataset_name', type=str, default='coco', help='dataset type decides methods of loading')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training hyper_parameters
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--shuffle_data', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')

        # file paths
        parser.add_argument('--dataroot', type=str, default='./datasets/', help='file path of data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # hardware
        parser.add_argument('--nThreads', default=0, type=int,
                            help='threads for loading data, argument: num_workers in DataLoader')
        parser.add_argument('--gpu_ids', type=str, default='-1',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # option file
        parser.add_argument('--save_opt_file', action='store_true')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')

        # input/output sizes
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop',
                            help='scaling and cropping of images at load time.',
                            choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=1024,
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=512,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=182,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true',
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # it will print default value if use '--help'
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self):

        opt = self.gather_options()
        # train or test
        opt.isTrain = True if opt.phase == 'train' else False

        self.print_options(opt)

        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batch_size, len(opt.gpu_ids))

        if opt.save_opt_file:
            self.save_options(opt)

        self.opt = opt
        return self.opt
