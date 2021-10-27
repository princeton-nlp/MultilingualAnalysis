"""
Script for running pre-training and finetuning experiments on GPUs (locally) and TPUs

Usage: python run_experiments.py --tpu --task  mlm --run_type monolingual --language en --other_args train_file=something.txt bool_arg_1 bool_arg_2
"""
import argparse
import collections
import os

SCRIPTS = {
    'mlm': 'transformers/examples/language-modeling/run_mlm_synthetic.py',
    'xnli': 'transformers/examples/text-classification/run_glue_synthetic.py',
    'ner': 'transformers/examples/token-classification/run_ner_synthetic.py',
    'pos': 'transformers/examples/token-classification/run_ner_synthetic.py',
    'tatoeba': 'transformers/examples/sentence_retrieval/run_sentence_retrieval_synthetic.py',
    'xquad': 'transformers/examples/question-answering/run_qa_synthetic.py'
}


def construct_command(args, arg_dict, default_hyperparams, task):
    num_cores = 8 if task == 'mlm' else '1'
    if args.tpu:
        prefix = 'python transformers/examples/xla_spawn.py --num_cores {} '.format(num_cores)
    else:
        prefix = 'python '.format(num_cores)

    # Append the task name to run_name and output directory
    arg_dict['output_dir'] = os.path.join(arg_dict['output_dir'], task)
    arg_dict['run_name'] = arg_dict['run_name'] + '_{}'.format(task)

    # Append script_name
    prefix = prefix + SCRIPTS[task]

    # Add all passed arguments (in arg_dict) to default_hyperparams
    # Hyperparameters in arg_dict are given precedence
    default_hyperparams.update(arg_dict)

    # Append all the passed arguments to the command
    for key in default_hyperparams:
        if default_hyperparams[key]:
            prefix = prefix + ' --{} {}'.format(key, default_hyperparams[key])
        else:
            prefix = prefix + ' --{}'.format(key)

    return prefix


def add_hyperparams_get_command(default_hyperparams, tpu_hyperparams, tpu_files, debug_hyperparams, debug_files, args, arg_dict):
    # We need to construct two commands for supervised datasets
    # One is the standard training on default settings, and the other is zero-shot evaluation

    # Combine all the default hyperparameters
    if args.tpu:
        default_hyperparams.update(tpu_hyperparams)
        default_hyperparams.update(tpu_files)
    else:
        default_hyperparams.update(debug_hyperparams)
        default_hyperparams.update(debug_files)

    # Construct the train command first. Don't pass any arg_dict.
    temp_dict_for_train = {'output_dir': arg_dict['output_dir'], 'run_name': arg_dict['run_name'], 'model_name_or_path': arg_dict['model_name_or_path']}
    train_command = construct_command(args, temp_dict_for_train, default_hyperparams, args.task)

    # Zero-shot evaluation command
    # The model name or path should be the output_dir of the previous run
    arg_dict['model_name_or_path'] = str(os.path.join(arg_dict['output_dir'], args.task))
    # Remove do_train
    default_hyperparams.pop('do_train', None)
    arg_dict.pop('do_train', None)
    zero_shot_command = construct_command(args, arg_dict, default_hyperparams, args.task)

    # Construct and return the argument
    return [train_command, zero_shot_command]


def get_mlm_command(args, arg_dict):
    # Define the default hyperparameters in a dictionary
    # Try avoiding arguments which are already defaults in the code

    # Default hyperparameters
    default_hyperparams = {
        'warmup_steps': '10000', 'learning_rate': '1e-4', 'save_steps': '-1', 'max_seq_length': 512,
        'logging_steps': '50', 'overwrite_output_dir': None, 'model_type': 'roberta',
        'config_name': 'config/{}/roberta_8/config.json'.format(args.language), 'tokenizer_name': 'config/{}/roberta_8/'.format(args.language),
        'do_train': None, 'do_eval': None
    }
    
    # Debug hyperparameters
    debug_hyperparams = {
        'max_steps': '1000', 'per_device_train_batch_size': '3', 'per_device_eval_batch_size': '3'
    }
    # TPU hyperparameters
    tpu_hyperparams = {
        'max_steps': '500000', 'per_device_train_batch_size': '16', 'per_device_eval_batch_size': '16'
    }

    # Get the train and validation file
    # Debug files
    prefix = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/dependency_parse_data'
    debug_files = {
        'train_file': os.path.join(prefix, args.language, 'train.txt'), 'validation_file': os.path.join(prefix, args.language, 'valid.txt')
    }
    # TPU files
    prefix = '../../bucket/pretrain_data'
    tpu_files = {
        'train_file': os.path.join(prefix, args.language, 'train.txt'), 'validation_file': os.path.join(prefix, args.language, 'valid.txt')
    }

    # Combine all the default hyperparameters
    if args.tpu:
        default_hyperparams.update(tpu_hyperparams)
        default_hyperparams.update(tpu_files)
    else:
        default_hyperparams.update(debug_hyperparams)
        default_hyperparams.update(debug_files)

    # Construct and return the argument
    return construct_command(args, arg_dict, default_hyperparams, 'mlm')


def get_xnli_command(args, arg_dict):
    # Define the default hyperparameters in a dictionary
    # Try avoiding arguments which are already defaults in the code

    # Default hyperparameters
    default_hyperparams = {
        'learning_rate': '2e-5', 'save_steps': '-1', 'max_seq_length': 128,
        'logging_steps': '50', 'overwrite_output_dir': None,
        'do_train': None, 'do_eval': None
    }
    
    # Debug hyperparameters
    debug_hyperparams = {
        'num_train_epochs': '1', 'per_device_train_batch_size': '3', 'per_device_eval_batch_size': '3'
    }
    # TPU hyperparameters
    tpu_hyperparams = {
        'num_train_epochs': '5', 'per_device_train_batch_size': '32', 'per_device_eval_batch_size': '32'
    }

    # Get the train and validation file
    # Debug files
    prefix = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli'
    debug_files = {
        'train_file': os.path.join(prefix, args.language, 'train_{}.json'.format(args.language)), 'validation_file': os.path.join(prefix, args.language, 'dev_{}.json'.format(args.language))
    }
    # TPU files
    prefix = '../../bucket/supervised_data/xnli'
    tpu_files = {
        'train_file': os.path.join(prefix, args.language, 'train_{}.json'.format(args.language)), 'validation_file': os.path.join(prefix, args.language, 'dev_{}.json'.format(args.language))
    }

    # Pass all the parameters and get the command
    return add_hyperparams_get_command(default_hyperparams, tpu_hyperparams, tpu_files, debug_hyperparams, debug_files, args, arg_dict)


def get_ner_command(args, arg_dict):
    # Define the default hyperparameters in a dictionary
    # Try avoiding arguments which are already defaults in the code

    # Default hyperparameters
    # max_seq_length argument is not accepted for some reason
    default_hyperparams = {
        'learning_rate': '2e-5', 'save_steps': '-1', 'task_name': args.task,
        'logging_steps': '500', 'overwrite_output_dir': None,
        'do_train': None, 'do_eval': None
    }
    
    # Debug hyperparameters
    debug_hyperparams = {
        'num_train_epochs': '1', 'per_device_train_batch_size': '3', 'per_device_eval_batch_size': '3'
    }
    # TPU hyperparameters
    tpu_hyperparams = {
        'num_train_epochs': '10', 'per_device_train_batch_size': '32', 'per_device_eval_batch_size': '32'
    }

    # Get the train and validation file
    # Debug files
    prefix = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/ner'
    debug_files = {
        'train_file': os.path.join(prefix, args.language, 'train.json'), 'validation_file': os.path.join(prefix, args.language, 'dev.json')
    }
    # TPU files
    prefix = '../../bucket/supervised_data/ner'
    tpu_files = {
        'train_file': os.path.join(prefix, args.language, 'train.json'), 'validation_file': os.path.join(prefix, args.language, 'dev.json')
    }

    # Pass all the parameters and get the command
    return add_hyperparams_get_command(default_hyperparams, tpu_hyperparams, tpu_files, debug_hyperparams, debug_files, args, arg_dict)


def get_pos_command(args, arg_dict):
    # Define the default hyperparameters in a dictionary
    # Try avoiding arguments which are already defaults in the code

    # Default hyperparameters
    # max_seq_length argument is not accepted for some reason
    default_hyperparams = {
        'learning_rate': '2e-5', 'save_steps': '-1', 'task_name': args.task,
        'logging_steps': '500', 'overwrite_output_dir': None,
        'do_train': None, 'do_eval': None
    }
    
    # Debug hyperparameters
    debug_hyperparams = {
        'num_train_epochs': '1', 'per_device_train_batch_size': '3', 'per_device_eval_batch_size': '3'
    }
    # TPU hyperparameters
    tpu_hyperparams = {
        'num_train_epochs': '10', 'per_device_train_batch_size': '32', 'per_device_eval_batch_size': '32'
    }

    # Get the train and validation file
    # Debug files
    prefix = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/pos'
    debug_files = {
        'train_file': os.path.join(prefix, args.language, 'train-{}.json'.format(args.language)), 'validation_file': os.path.join(prefix, args.language, 'dev-{}.json'.format(args.language))
    }
    # TPU files
    prefix = '../../bucket/supervised_data/pos'
    tpu_files = {
        'train_file': os.path.join(prefix, args.language, 'train-{}.json'.format(args.language)), 'validation_file': os.path.join(prefix, args.language, 'dev-{}.json'.format(args.language))
    }

    # Pass all the parameters and get the command
    return add_hyperparams_get_command(default_hyperparams, tpu_hyperparams, tpu_files, debug_hyperparams, debug_files, args, arg_dict)


def get_tatoeba_command(args, arg_dict):
    # Define the default hyperparameters in a dictionary
    # Try avoiding arguments which are already defaults in the code

    # Default hyperparameters
    default_hyperparams = {
        'max_seq_length': 128, 'pool_type': 'middle',
        'logging_steps': '50', 'overwrite_output_dir': None,
        'do_train': None
    }
    
    # Debug hyperparameters
    debug_hyperparams = {
        'per_device_train_batch_size': '3', 'per_device_eval_batch_size': '3'
    }
    # TPU hyperparameters
    tpu_hyperparams = {
        'per_device_train_batch_size': '32', 'per_device_eval_batch_size': '32'
    }

    # Get the train and validation file
    # Debug files
    prefix = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/tatoeba'
    debug_files = {
        'train_file': os.path.join(prefix, args.language, '{}.json'.format(args.language))
    }
    # TPU files
    prefix = '../../bucket/supervised_data/tatoeba'
    tpu_files = {
        'train_file': os.path.join(prefix, args.language, '{}.json'.format(args.language))
    }

    # Combine all the default hyperparameters
    if args.tpu:
        default_hyperparams.update(tpu_hyperparams)
        default_hyperparams.update(tpu_files)
    else:
        default_hyperparams.update(debug_hyperparams)
        default_hyperparams.update(debug_files)

    # Zero-shot evaluation command
    # The model name or path should be the output_dir of the previous run
    zero_shot_command = construct_command(args, arg_dict, default_hyperparams, args.task)

    # Construct and return the argument
    return [zero_shot_command]    


def get_xquad_command(args, arg_dict):
    # Define the default hyperparameters in a dictionary
    # Try avoiding arguments which are already defaults in the code

    # Default hyperparameters
    default_hyperparams = {
        'learning_rate': '3e-5', 'save_steps': '-1', 'max_seq_length': 384,
        'doc_stride': 128, 'warmup_steps': 500, 'weight_decay': 0.0001,
        'logging_steps': '50', 'overwrite_output_dir': None,
        'do_train': None, 'do_eval': None
    }
    
    # Debug hyperparameters
    debug_hyperparams = {
        'num_train_epochs': '1', 'per_device_train_batch_size': '3', 'per_device_eval_batch_size': '3'
    }
    # TPU hyperparameters
    tpu_hyperparams = {
        'num_train_epochs': '2', 'per_device_train_batch_size': '16', 'per_device_eval_batch_size': '16'
    }

    # Get the train and validation file
    # Debug files
    prefix = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xquad'
    debug_files = {
        'train_file': os.path.join(prefix, args.language, 'train_{}.json'.format(args.language)), 'validation_file': os.path.join(prefix, args.language, 'dev_{}.json'.format(args.language))
    }
    # TPU files
    prefix = '../../bucket/supervised_data/xquad'
    tpu_files = {
        'train_file': os.path.join(prefix, args.language, 'train_{}.json'.format(args.language)), 'validation_file': os.path.join(prefix, args.language, 'dev_{}.json'.format(args.language))
    }

    # Pass all the parameters and get the command
    return add_hyperparams_get_command(default_hyperparams, tpu_hyperparams, tpu_files, debug_hyperparams, debug_files, args, arg_dict)  


def construct_and_run(args):
    # Create a dictionary of all the arguments that need to be passed to the script
    arg_dict = {}
    if args.other_args:
        for arg in args.other_args:
            if '=' in arg:
                # Arguments with values
                split_args = arg.split('=')
                arg_dict[split_args[0]] = split_args[1]
            else:
                # Boolean arguments
                arg_dict[arg] = None

    # Make sure that the output_dir and run_name arguments are passed
    assert 'output_dir' in arg_dict and 'run_name' in arg_dict, "Please provide the {} and {} flags".format('output_dir', 'run_name')

    # Model name or path should be passed if task is not MLM
    if args.task != 'mlm':
        assert 'model_name_or_path' in arg_dict, "Please provide {} argument.".format('model_name_or_path')

    # Construct a command for each task
    commands = collections.OrderedDict([('mlm', None), ('xnli', None), ('ner', None), ('pos', None), ('tatoeba', None), ('xquad', None)])
    
    # Construct the MLM command (if required)
    if args.task in ['all', 'mlm']:
        commands['mlm'] = get_mlm_command(args, arg_dict)

    # Construct the XNLI command (if required)
    if args.task in ['all', 'xnli']:
        commands['xnli'] = get_xnli_command(args, arg_dict)

    # Construct the NER command (if required)
    if args.task in ['all', 'ner']:
        commands['ner'] = get_ner_command(args, arg_dict)

    # Construct the POS command (if required)
    if args.task in ['all', 'pos']:
        commands['pos'] = get_pos_command(args, arg_dict)

    # Construct the TATOEBA command (if required)
    if args.task in ['all', 'tatoeba']:
        commands['tatoeba'] = get_tatoeba_command(args, arg_dict)

    # Construct the XQuAD command (if required)
    if args.task in ['all', 'xquad']:
        commands['xquad'] = get_xquad_command(args, arg_dict)

    # Run all the commands one after another
    # NOTE: This waits for the execution to complete
    for key in commands:
        if commands[key]:
            if key == 'mlm':
                if args.run_command:
                    os.system(commands[key])
                else:
                    print('\n'+commands[key]+'\n')
            else:
                for command in commands[key]:
                    if args.run_command:
                        os.system(command)
                    else:
                        print('\n'+command+'\n')


def main():
    parser = argparse.ArgumentParser()

    # Script Arguments
    parser.add_argument("--tpu", action="store_true", help="If this flag is selected, the code will be run on the TPUs.")
    parser.add_argument("--run_command", action="store_true", help="If this flag is not used, the commands to use are only printed.")
    parser.add_argument("--task", type=str, required=True, help="~all~/mlm/xnli/ner/pos/tatoeba/xquad")
    parser.add_argument("--run_type", type=str, required=True, help="word_modification/syntax_modification/monolingual/bilingual")
    parser.add_argument("--language", type=str, required=True, help="en/fr/ar/hi")

    # All other arguments passed are stored separately
    # Pass all arguments which need a value with an `=` sign. Like --model_name_or_path=./dir
    # This makes an assumption that there is no `=` sign in any of the values
    # Don't use the -- sign when passing arguments
    # Example command: python main_script.py --tpu --task all --other_args do_train do_eval model_name_or_path=bert-base-uncased
    parser.add_argument('--other_args', nargs="*")

    # args.other_args is now a list of all the arguments that need to be passed
    args = parser.parse_args()

    # Only word modification is supported for now
    assert args.run_type in ['word_modification', 'monolingual'], "Given run type is not supported."

    # Helpful reminder messages
    print("***** Make sure wandb is enabled *****")
    print("***** The task name is appended to the output_dir *****")
    print("***** Keep the output_dir the same for all the tasks. The script automatically appends the task name. *****")

    # Construct and run command
    construct_and_run(args)

if __name__ == '__main__':
    main()