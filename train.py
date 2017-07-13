"""Command-line utility for training"""

import argparse
from model import train
import numpy as np
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the data')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs of the algorithm')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_train', type=int, default=512, help='Batch size')
    parser.add_argument('--batch_test', type=int, default=512, help='Batch size for testing')
    parser.add_argument('--hidden_units', type=int, default=64, help='Number of hiddden units of the LSTM cells')
    parser.add_argument('--beta', type=float, default=5e-4, help='Coefficient of the L2 regularization')
    parser.add_argument('--dropout', type=float, default=1., help='Probability to keep a given neuron (dropout)')
    parser.add_argument('--inception', action='store_true', help='If specified, trains the inception-like net')
    parser.add_argument('--concat', action='store_true', help='If specified, the outputs of the LSTM cells are concatenated instead of averaged')
    parser.add_argument('--bidirectional', action='store_true', help='If specified, bidirectional LSTM is used (inception-like net uses bidirectional LSTM regardless of this parameter)')
    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data = pickle.load(f)

    precision = []
    recall = []
    run_n = args.runs

    args = vars(args) 
    del args['path']
    del args['runs']
    
    for i in range(run_n):
        print('*' * 40)
        print('[RUN %i/%i]' % (i + 1, run_n))
        print('*' * 40)
        results = train(data, **args)
        precision.append(results['precision'])
        recall.append(results['recall'])
    print('*' * 40)
    print('*' * 40)
    print('Final results averaged over %i runs:' % run_n)
    print('Recall %.4f (%.4f), precision %.4f (%.4f)' % (np.mean(recall), np.std(recall), np.mean(precision), np.std(precision)))
    print('***** Parameters *****')
    print(results)
    print('*' * 40)
    print('*' * 40)

if __name__ == "__main__":
    main()
