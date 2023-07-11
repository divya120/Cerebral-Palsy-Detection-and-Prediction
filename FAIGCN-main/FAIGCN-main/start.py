#!/usr/bin/env python
import argparse
import sys
import os
# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':
    print('LOO-CV Start')
    # VAL set data:01
    print('=========Val set:01,  Result=========')
    os.system('python main.py recognition -c data/01/train.yaml')
    # VAL set data:02
    print('=========Val set:02,  Result=========')
    os.system('python main.py recognition -c data/02/train.yaml')
    # VAL set data:03
    print('=========Val set:03,  Result=========')
    os.system('python main.py recognition -c data/03/train.yaml')
    # VAL set data:04
    print('=========Val set:04,  Result=========')
    os.system('python main.py recognition -c data/04/train.yaml')
    # VAL set data:05
    print('=========Val set:05,  Result=========')
    os.system('python main.py recognition -c data/05/train.yaml')
    # VAL set data:06
    print('=========Val set:06,  Result=========')
    os.system('python main.py recognition -c data/06/train.yaml')
    # VAL set data:07
    print('=========Val set:07,  Result=========')
    os.system('python main.py recognition -c data/07/train.yaml') 
    # VAL set data:08
    print('=========Val set:08,  Result=========')
    os.system('python main.py recognition -c data/08/train.yaml')
    # VAL set data:09
    print('=========Val set:09,  Result=========')
    os.system('python main.py recognition -c data/09/train.yaml')
    # VAL set data:10
    print('=========Val set:10,  Result=========')
    os.system('python main.py recognition -c data/10/train.yaml')
    # VAL set data:11
    print('=========Val set:11,  Result=========')
    os.system('python main.py recognition -c data/11/train.yaml')
    # VAL set data:12       
    print('=========Val set:12,  Result=========')
    os.system('python main.py recognition -c data/12/train.yaml')


