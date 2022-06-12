import argparse
import onmt.opts as opts
import torch
import glob
import sys

def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def main():
    # 命令行参数加载
    opt = parse_args()

    if (opt.max_shard_size > 0):
        raise AssertionError("-max_shard_size is deprecated, please use \
                             -shard_size (number of examples) instead.")
    if (opt.shuffle > 0):
        raise AssertionError("-shuffle is not implemented, please make sure \
                             you shuffle your data before pre-processing.")
    

if __name__ == "__main__":
    main()