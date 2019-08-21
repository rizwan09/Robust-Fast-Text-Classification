"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
    archive_file          path to an archived trained model
    input_file            path to the file containing the evaluation data

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file to save metrics
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any, Iterable
import argparse
import logging
<<<<<<< HEAD
import numpy as np
import pickle, sys, math
=======
import json

import torch
>>>>>>> 9b2f0b45abdad09c78036fdaebe7dd2c9973128a

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import prepare_environment
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class Evaluate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Evaluate the specified model + dataset')

        subparser.add_argument('archive_file', type=str, help='path to an archived trained model')

        subparser.add_argument('input_file', type=str, help='path to the file containing the evaluation data')

        subparser.add_argument('--output-file', type=str, help='path to output file')

        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--batch-weight-key',
                               type=str,
                               default="",
                               help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

        cuda_device.add_argument('--theta',
                                 type=float,
                                 default=None,
                                 help='thereshold for skimming')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
<<<<<<< HEAD
             theta: float = None) -> Dict[str, Any]:
    model.eval()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    iterator = data_iterator(instances, num_epochs=1, cuda_device=cuda_device, for_training=False)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
    all_rep = []
    nrow = 0
    ncolumn = 0
    skimmed_words = 0
    total_words = 1
    time_encoder = 0
            
    for batch in generator_tqdm:
        nrow+=1
        # if nrow<633:
        #     continue
        # import pdb
        # pdb.set_trace()
        if theta: batch["theta"] = theta
        output_dict = model(**batch)
        skimmed_words += output_dict["skimmed_words"] 
        total_words+=output_dict['total_words'] 
        time_encoder += output_dict['time_needed']

        metrics = model.get_metrics()
        description = ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||" 
        description = description + "skimmed:"+ str(skimmed_words)+"/"+ str(total_words) + \
        " frac: "+str( 1.0*skimmed_words/total_words) + 'encoder time: ' + convert_to_minutes(time_encoder)
        generator_tqdm.set_description(description, refresh=False)
        # print("skimmed:"+ str(skimmed_words)+"/"+ str(total_words)+ " frac: "+str( 1.0*skimmed_words/total_words), file=sys.stderr) 
    

        # all_rep.append(output_dict['rep'])
        # nrow += output_dict['rep'].size(0)
        # ncolum = output_dict['rep'].size(1)

    # numpy_all_rep = np.zeros((nrow, ncolum))
    # idx = 0
    # for var in all_rep:
    #     sz = var.size(0)
    #     numpy_all_rep[idx:idx+sz, :] = var.data.numpy()
    #     idx += sz
    
    # task = 'SST'
    # with open('WAG_LSTM_'+task+'_FULL_REP.pkl', 'wb') as f:
    #     pickle.dump(numpy_all_rep,  f)

    # with open('WAG_AGNEWS_FULL_TEXT_HIDDEN_REP.pkl', 'rb') as f:
    #     nloaded_np_rep = pickle.load(f)
    # import pdb
    # pdb.set_trace()

    logger.info("skimmed:"+ str(skimmed_words)+"/"+ str(total_words) + \
        " frac: "+str( 1.0*skimmed_words/total_words) + 'encoder time: ' + convert_to_minutes(time_encoder)) 
    return model.get_metrics(reset=True)
=======
             batch_weight_key: str) -> Dict[str, Any]:
    _warned_tqdm_ignores_underscores = False
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not _warned_tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                _warned_tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                   "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics
>>>>>>> 9b2f0b45abdad09c78036fdaebe7dd2c9973128a


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements

    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

<<<<<<< HEAD
    metrics = evaluate(model, instances, iterator, args.cuda_device, args.theta)
=======
    metrics = evaluate(model, instances, iterator, args.cuda_device, args.batch_weight_key)
>>>>>>> 9b2f0b45abdad09c78036fdaebe7dd2c9973128a

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_file = args.output_file
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics
