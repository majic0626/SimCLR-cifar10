import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--batch", type=int, default=128, help="batch size")
parser.add_argument("--epoch", type=int, default=100, help="epoch")
parser.add_argument("--classNum", type=int, default=10, help="class number")
parser.add_argument("--data_root", type=str, default="/usr/share/dataset/cifar10", help="root dir of dataset")
parser.add_argument("--dir_ckpt", type=str, default="./ckpts/t05_100", help="dir for saving checkpoints")
parser.add_argument("--dir_log", type=str, default="./log/t05_100", help="dir for saving log file")
parser.add_argument("--workers", type=int, default=8, help="number of subprocess for loading data")
parser.add_argument("--temperature", type=float, default=0.5, help="Temperature scale for loss function")
parser.add_argument("--accumulate", type=int, default=4, help="calucalte gradient until accumulating N batch")
parser.add_argument("--strength", type=float, default=0.5, help="strength of augmentation")
parser.add_argument("--useLARS", type=bool, default=True, help="using LARS optimizer")
args = parser.parse_args()
