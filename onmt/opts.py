import argparse
from ast import parse


def add_md_help_argument(parser):
    """md help parse"""
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')

def preprocess_opts(parser):
    """ Pre-processing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="""Type of the source input.
                       Options are [text|img].""")

    group.add_argument('-train_src', required=True,
                       help="Path to the training source data")
    group.add_argument('-train_tgt', required=True,
                       help="Path to the training target data")
    group.add_argument('-valid_src', required=True,
                       help="Path to the validation source data")
    group.add_argument('-valid_tgt', required=True,
                       help="Path to the validation target data")

    group.add_argument('-src_dir', default="",
                       help="Source directory for image or audio files.")

    group.add_argument('-save_data', required=True,
                       help="Output file for the prepared data")

    group.add_argument('-max_shard_size', type=int, default=0,
                       help="""Deprecated use shard_size instead""")

    group.add_argument('-shard_size', type=int, default=1000000,
                       help="""Divide src_corpus and tgt_corpus into
                       smaller multiple src_copus and tgt corpus files, then
                       build shards, each shard will have
                       opt.shard_size samples except last shard.
                       shard_size=0 means no segmentation
                       shard_size>0 means segment dataset into multiple shards,
                       each shard has shard_size samples""")

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab', default="",
                       help="""Path to an existing source vocabulary. Format:
                       one word per line.""")
    group.add_argument('-tgt_vocab', default="",
                       help="""Path to an existing target vocabulary. Format:
                       one word per line.""")
    group.add_argument('-features_vocabs_prefix', type=str, default='',
                       help="Path prefix to existing features vocabularies")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=100000,
                       help="Maximum source sequence length")
    group.add_argument('-src_seq_length_trunc', type=int, default=200,
                       help="Truncate source sequence length.")
    group.add_argument('-tgt_seq_length', type=int, default=100,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                       help="Truncate target sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase data')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=0,
                       help="Shuffle data")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")
    group.add_argument('-log_file', type=str, default="",
                       help="Output logs to a file under this path.")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")
    group.add_argument('-window_stride', type=float, default=.01,
                       help="Window stride for spectrogram in seconds.")
    group.add_argument('-window', default='hamming',
                       help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add_argument('-image_channel_size', type=int, default=3,
                       choices=[3, 1],
                       help="""Using grayscale image can training
                       model faster and smaller""")


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    """ MD help action """

    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()

class MarkdownHelpAction(argparse.Action):
    """MD help action"""
    def __init__(self, option_strings,
                dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs
        )
    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()