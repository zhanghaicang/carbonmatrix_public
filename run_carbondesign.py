import os
import argparse
import logging
from logging.handlers import QueueHandler, QueueListener
import resource
import json
from random import shuffle

import ml_collections
import numpy as np
import torch
import torch.multiprocessing as mp
from einops import rearrange
from collections import OrderedDict
from carbondesign.data.pdbio import save_pdb

from carbondesign.model.carbondesign import CarbonDesign
from carbondesign.testloader import dataset_test
from carbondesign.common.utils import index_to_str_seq

def worker_device(rank, args):
    if args.device == 'gpu':
        return torch.device(f'cuda:{args.gpu_idx[rank]}')
    else:
        return torch.device('cpu')

def worker_load(rank, args):
    def _feats_gen(feats, device):
        for fn, opts in feats:
            if 'device' in opts:
                opts['device'] = device
            yield fn, opts
    
    device = worker_device(rank, args)
    
    # model
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
        config = ml_collections.ConfigDict(config)

    checkpoint = torch.load(args.model, map_location=device)
    model = CarbonDesign(config=config.model)
    model_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        model_state_dict[k] = v
    model.load_state_dict(model_state_dict,strict = False)
    
    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.loads(f.read())

        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device

    model = model.to(device=device)
    model.eval()
    return list(_feats_gen(feats, device)), model


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)

def one_hot(a, num_classes=21):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def mrf_score(label, site_repr, pair_repr, site_mask, pair_mask):
    N, C = site_repr.shape
    pair_repr = np.reshape(pair_repr, [N, N, C, C])
    label = one_hot(label)
    
    score = np.sum(site_repr * label * site_mask[:,None]) + np.sum(label[None,:,None,:] * label[:,None,:,None] * pair_repr * pair_mask[:,:,None,None]) / 2.0
    return score

def temp_softmax(z, T):
    exp_z = np.exp(z/T)
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z

def infer_mrf(init_label, site_repr, pair_repr, site_mask, pair_mask, T):
    N, C = site_repr.shape
    pair_repr = np.reshape(pair_repr, [N, N, C, C])
    deg = np.sum(pair_mask, axis=-1)

    prev_label = np.array(init_label)
    pos = np.argsort(deg)
    
    for cycle in range(5):
        #shuffle(pos)
        updated_count = 0
        for i in pos:
            if not site_mask[i]:
                continue
            obj = -10000.0
            obj_c = -1
            t_lis = np.zeros(C-1)
            for c in range(C - 1):
                t = site_repr[i, c]
                for k in range(N):
                    if pair_mask[i, k]:
                        t += pair_repr[i, k, c, prev_label[k]]
                        t_lis[c] = t
            if T >= 0.1:
                probs = temp_softmax(t_lis, T)
                obj_c = np.argmax(np.random.multinomial(1, probs))
            else:
                obj_c = np.argmax(t_lis)
            if prev_label[i] != obj_c:
                prev_label[i] = obj_c
                updated_count += 1
        if updated_count == 0:
            break
    return prev_label

def replace_with_x(seq_str, mask):
    ret_str = []
    for a, m in zip(seq_str, mask):
        if m:
            ret_str.append(a)
        else:
            ret_str.append('X')
    return ''.join(ret_str)

def evaluate_mrf_one(name, gt_str_seq, site_repr, pair_repr, site_mask, pair_mask, args, sidechain_all):
    site_prob = softmax(site_repr)
    label = np.argmax(site_prob, axis=-1)
    
    pred_str_seq1 = index_to_str_seq(label)
    pred_str_seq1 = replace_with_x(pred_str_seq1, site_mask)    
    total_len = len(gt_str_seq)
    valid_len = np.sum(site_mask)
    T = args.temperature

    label = infer_mrf(label, site_repr, pair_repr, site_mask, pair_mask, T)
    pred_str_seq2 = index_to_str_seq(label)
    
    label[~site_mask] = 20
    L = len(sidechain_all[0])
    sidechain_single = []
    for i in range(L):
        sidechain_single.append(sidechain_all[label[0],i,:])
    pred_str_seq2 = replace_with_x(pred_str_seq2, site_mask)
    sidechain_single = np.array(sidechain_single)
    sidechain_all = np.array(sidechain_all)
    print(f'>{name}\tCarbonDesign\n{pred_str_seq2}\n')
    with open(os.path.join(args.output_dir, name + '.fasta'), 'w') as fw:
        fw.write(f'>{name}\tCarbonDesign\n{pred_str_seq2}\n')

    if args.save_mrf:
        np.savez(os.path.join(args.output_dir, name + '.mrf.npz'), 
                site_repr=site_repr,
                pair_repr=pair_repr,
                site_mask=site_mask,
                pair_mask=pair_mask) 
    if args.save_sidechain:
        pdb_file = os.path.join(args.output_dir, f'{name}_sidechain.pdb')
        sequence = []
        sequence.append(pred_str_seq2)
        plddt = None
        chains = 'A'
        save_pdb(sequence, sidechain_single, pdb_file, chains)

def evaluate_mrf_batch(batch, ret, args):
    
    gt_str_seq, site_repr, pair_repr, sidechain_all = batch['str_seq'], ret['heads']['seqhead']['logits'].to('cpu').numpy(), ret['heads']['pairhead']['logits'].to('cpu').numpy(), ret['heads']['folding']['sidechains']['atom_pos'].cpu().numpy()
    site_mask, pair_mask = batch['mask'].to('cpu').numpy(), batch['pair_mask'].to('cpu').numpy()
    names = batch['name']
    L = sidechain_all.shape[1]
    sidechain_all=sidechain_all.reshape(-1,21,L,14,3)
    for name, site_mask_, pair_mask_, gt_str_seq_, pair_repr_, site_repr_, sidechain_all_ in zip(names, site_mask, pair_mask, gt_str_seq, pair_repr, site_repr, sidechain_all):
        evaluate_mrf_one(name, gt_str_seq_, site_repr_, pair_repr_, site_mask_, pair_mask_, args, sidechain_all_)
def evaluate(rank, log_queue, args):
    #worker_setup(rank, log_queue, args)

    feats, model = worker_load(rank, args)
    # logging.info('feats: %s', feats)

    
    device = worker_device(rank, args)
    name_idx = []
    with open(args.name_idx) as f:
        name_idx = [x.strip() for x in f]

    test_loader = dataset_test.load(
        data_dir=args.data_dir,
        name_idx=name_idx,
        feats=feats,
        is_cluster_idx=False,
        rank=None,
        world_size=1,
        batch_size=args.batch_size)
    
    for i, batch in enumerate(test_loader):
        if batch is None:
            continue
        try:
            logging.debug('name: %s', ','.join(batch['name']))
            logging.debug('len : %s', batch['seq'].shape[1])
            logging.debug('seq : %s', batch['str_seq'][0])
            if batch['seq'].shape[1] > 600:
                continue
            with torch.no_grad():
                ret = model(batch=batch, compute_loss=True)

            #print(ret['heads']['folding']['final_atom14_positions'].shape)
            sidechain_21 = ret['heads']['folding']['sidechains']['atom_pos'].shape

            evaluate_mrf_batch(batch, ret, args)
        except:
            logging.error('fails in predicting', batch['name'])


def main(args):
    mp.set_start_method('spawn', force=True)
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                args.output_dir,
                f'{os.path.splitext(os.path.basename(__file__))[0]}.log'))]

    def handler_apply(h, f, *arg):
        f(*arg)
        return h
    level = logging.DEBUG if args.verbose else logging.INFO
    handlers = list(map(lambda x: handler_apply(
        x, x.setLevel, level), handlers))
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
    handlers = list(map(lambda x: handler_apply(
        x, x.setFormatter, logging.Formatter(fmt)), handlers))

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    log_queue = mp.Queue(-1)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    # logging.info('-----------------')
    # logging.info('Arguments: %s', args)
    # logging.info('-----------------')

    evaluate(0, log_queue, args)

    '''
    logging.info('-----------------')
    logging.info('Resources(myself): %s',
                 resource.getrusage(resource.RUSAGE_SELF))
    logging.info('Resources(children): %s',
                 resource.getrusage(resource.RUSAGE_CHILDREN))
    logging.info('-----------------')
    '''

    listener.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_mrf', action='store_true')
    parser.add_argument('--save_sidechain',action='store_true')
    parser.add_argument('--model', type=str,
            default='./params/carbondesign_default.ckpt')
    parser.add_argument('--model_features', type=str,
            default='./config/config_data_mrf2.json')
    parser.add_argument('--model_config', type=str,
            default='./config/config_model_mrf_pair_enable_esm_sc.json')
    
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu_idx', type=int, nargs='+', default=[0])
    parser.add_argument('--map_location', type=str, default=None)
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--ipc_file', type=str, default='./test.ipc')

    args = parser.parse_args()

    main(args)

