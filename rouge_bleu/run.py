from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import normalize
from utils import compute_bleu_rouge
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data(pred_file, ref_file):
    """
    Prepare data.
    """
    with open(pred_file, "r") as f_pred:
        pred_lines = f_pred.readlines()
    with open(ref_file, "r") as f_ref:
        ref_lines = f_ref.readlines()

    pred_answers, ref_answers = [], []
    for line in pred_lines:
        sample = json.loads(line)  # type is <class 'dict'>
        pred_answers.append(sample)

    for line in ref_lines:
        sample = json.loads(line)  # type is <class 'dict'>
        ref_answers.append(sample)

    return pred_answers, ref_answers


def run_eval(pred_answers_list, ref_answers_list):
    """
    Run eval.
    """
    pred_answers = pred_answers_list
    ref_answers = ref_answers_list

    # compute the bleu and rouge scores if reference answers is provided
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None

    return bleu_rouge


if __name__ == '__main__':
    pred_file, ref_file = "samples/predictions.json", "samples/reference.json"
    # acquire eval data
    pred_answers, ref_answers = prepare_data(pred_file, ref_file)

    # run overall
    bleu_rouge = run_eval(pred_answers, ref_answers)

    """On dureader2.0 `dev`
    {
        'Bleu-1': 0.5837681460393821, 
        'Bleu-2': 0.5226142125975742, 
        'Bleu-3': 0.48347404434526725, 
        'Bleu-4': 0.45561738017705655, 
        'Rouge-L': 0.5022681316295654
    }
    """
    logger.info('Eval results: \n {}'.format(bleu_rouge))
