#!/usr/bin/env python
"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import os
import numpy as np
import time
import pickle
import argparse
import torch
import json

from torch.utils.data import DataLoader
from esim.dataset import NLIDataset
from esim.utils import correct_predictions
from esim.model_fusion import ESIM_f


def test(model, dataloader):
    """
    Test the accuracy of a model on some dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    res = []
    res_num = []
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch['premise'].to(device)
            premises_lengths = batch['premise_length'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypotheses_lengths = batch['hypothesis_length'].to(device)
            labels = batch['label'].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            res_num += [int(w) for w in torch.argmax(probs, 1)]
            res += [[float(w) for w in t] for t in probs.data]

            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start

    return batch_time, total_time, res, res_num


def main(test_file,
         pretrained_file,
         vocab_size,
         embedding_dim=300,
         hidden_size=300,
         num_classes=3,
         batch_size=32,
         fold=10):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")
    result = []
    result_num = []
    for fd in range(fold):
        print("\t* Loading test data... Fold :{}".format(fd))
        with open(test_file, 'rb') as pkl:
            test_data = NLIDataset(pickle.load(pkl))

        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

        print("\t* Building model...")
        model = ESIM_f(vocab_size,
                    embedding_dim,
                    hidden_size,
                    num_classes=num_classes,
                    device=device).to(device)

        checkpoint = torch.load(pretrained_file.format(fd))
        model.load_state_dict(checkpoint['model'])

        print(20 * "=",
            " Testing ESIM model on device: {} ".format(device),
            20 * "=")
        batch_time, total_time, res, res_num  = test(model, test_loader)
        result.append(res)
        result_num.append(res_num)
    answer = []
    result = np.array(result).sum(0)
    result = np.argmax(result, 1)

    result_num = np.sum(result_num, 0)
    total_res = [] 
    for w, z in zip(result, result_num):
        if z < fold/2:
            answer.append(0)
        elif z > fold/2:
            answer.append(1)
        else:
            answer.append(w)
        total_res.append(str(w)+'\t'+str(z)+'\t'+str(answer[-1]))
    #open("../data/pingan/all_res.csv", 'w').write('\n'.join(total_res))


    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s,".format(batch_time, total_time))
    source = open('../data/pingan/test.csv', 'r').read().splitlines()
    source = source[:1] + [w + str(z) for w, z in zip(source[1:], answer)]
    open('../data/pingan/scofied7419_test_result.csv', 'w').write('\n'.join(source))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 some dataset')
    parser.add_argument('--checkpoint', default='../data/checkpoints/best_{}.pth.tar',
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument('--config', default='../config/test.json',
                        help='Path to a configuration file')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(config['test_data']),
         args.checkpoint,
         config['vocab_size'],
         config['embedding_dim'],
         config['hidden_size'],
         config['num_classes'],
         config['batch_size'],
         config['kfold'])
