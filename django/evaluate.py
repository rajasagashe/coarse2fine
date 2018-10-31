from __future__ import division
import os
import argparse
import torch
import codecs
import glob
import random

import table
import table.IO
import opts

parser = argparse.ArgumentParser(description='evaluate.py')
opts.translate_opts(parser)
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
opt.anno = os.path.join(opt.root_dir, opt.dataset, '{}.json'.format(opt.split))
opt.pre_word_vecs = os.path.join(opt.root_dir, opt.dataset, 'embedding')

if opt.beam_size > 0:
    opt.batch_size = 1

def write_to_file(file, pred, gold):
    file.write('==================' + '\n')
    file.write('NL:' + ' '.join(gold['src']) + '\n')
    file.write('Tgt:  ' + ' '.join(gold['lay']) + '\n')
    file.write('Pred: ' + ' '.join(pred.lay) + '\n')
    file.write('\n')
    file.write('Tgt:  ' + ' '.join(gold['tgt']) + '\n')
    file.write('Tgt:  ' + ' '.join(gold['token']) + '\n')
    file.write('Pred: ' + ' '.join(pred.tgt) + '\n')


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = table.IO.read_anno_json(opt.anno, opt)

    metric_name_list = ['tgt', 'lay']
    prev_best = (None, None)
    for fn_model in glob.glob(opt.model_path):
        opt.model = fn_model
        print(fn_model)
        print(opt.anno)

        translator = table.Translator(opt, dummy_opt.__dict__)
        data = table.IO.TableDataset(
            js_list, translator.fields, 0, None, False)
        test_data = table.IO.OrderedIterator(
            dataset=data, device=opt.gpu, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)

        # inference
        r_list = []
        for batch in test_data:
            r = translator.translate(batch)
            r_list += r
        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
            len(r_list), len(js_list))

        # open correct, corrlayincorrtgt, incorrlay files
        correct = open(os.path.join(opt.root_dir, 'out/correct.txt'), 'w')
        lay_corr_name_wrong = open(os.path.join(opt.root_dir, 'out/lay-right-name-wrong.txt'), 'w')
        lay_wrong = open(os.path.join(opt.root_dir, 'out/lay-wrong.txt'), 'w')
        pred_gold = []
        # evaluation
        for pred, gold in zip(r_list, js_list):
            pred.eval(gold)
            pred_gold.append((pred, gold))

        random.shuffle(pred_gold)
        for pred, gold in pred_gold:
            if pred.correct['tgt']:
                write_to_file(correct, pred, gold)
            elif pred.correct['lay']:
                print(gold)
                write_to_file(lay_corr_name_wrong, pred, gold)
            else:
                write_to_file(lay_wrong, pred, gold)

        correct.close()
        lay_corr_name_wrong.close()
        lay_wrong.close()

        print('Results:')
        for metric_name in metric_name_list:
            c_correct = sum((x.correct[metric_name] for x in r_list))
            acc = c_correct / len(r_list)
            print('{}: {} / {} = {:.2%}'.format(metric_name,
                                                c_correct, len(r_list), acc))
            if metric_name == 'tgt' and (prev_best[0] is None or acc > prev_best[1]):
                prev_best = (fn_model, acc)

    if (opt.split == 'dev') and (prev_best[0] is not None):
        with codecs.open(os.path.join(opt.root_dir, opt.dataset, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
            f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()
