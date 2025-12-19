import os
from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    args = parser.parse_args()
    args.method = 'DUaPH'
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))
    os.makedirs(args.save_dir, exist_ok=True)

    return args
