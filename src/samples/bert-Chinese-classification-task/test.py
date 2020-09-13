# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2020/6/5
# -------------------------------------------------------------------------------

import sys, os

from tokenization_word import *

from run_classifier_word import *

dir_ = '/home/zjj/Pycharm/PytorchLearning/src/samples/bert-Chinese-classification-task/GLUE/glue_data/'

dir_ = './GLUE/glue_data/'

CUDA_LAUNCH_BLOCKING = 1


def testMain(args):
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "news": NewsProcessor,
        'qnli': QnliProcessor
    }

    # device = torch.device("cpu")
    # n_gpu = 1

    bert_config = BertConfig(args.vocab_size)

    task_name = args.task_name.lower()
    assert task_name in processors
    assert args.max_seq_length < bert_config.max_position_embeddings

    processor = processors[task_name]()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    label_list = processor.get_labels()

    # Prepare model
    model = BertForSequenceClassification(bert_config, len(label_list))
    device = torch.device("cuda", )
    n_gpu = 1
    model.to(device)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()  ### 状态设置
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                optimizer.zero_grad()
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)

                loss.backward()
                optimizer.step()

                # if (step + 1) % args.gradient_accumulation_steps == 0:
                #     if args.fp16 or args.optimize_on_cpu:
                #         if args.fp16 and args.loss_scale != 1.0:
                #             # scale down gradients for fp16 training
                #             for param in model.parameters():
                #                 param.grad.data = param.grad.data / args.loss_scale
                #         is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                #         if is_nan:
                #             logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                #             args.loss_scale = args.loss_scale / 2
                #             model.zero_grad()
                #             continue
                #         optimizer.step()
                #         copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                #     else:
                #         optimizer.step()
                #     model.zero_grad()
                #     global_step += 1

    os.makedirs(args.output_dir, exist_ok=True)
    state_dic_path = 'args.output_dir' + '/bert_' + args.task_name + '.pt'
    os.remove(state_dic_path)
    torch.save(model.state_dict(), state_dic_path)


def evaMolde(model_dic_path, eva_data_dir):
    processor = QnliProcessor()
    label_list = processor.get_labels()

    vocab_file = '/home/zjj/Pycharm/PytorchLearning/src/samples/bert-Chinese-classification-task/GLUE/glue_data/vocab.vob'
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    model = BertForSequenceClassification(BertConfig(45838), len(label_list))
    model.load_state_dict(torch.load(model_dic_path))
    device = torch.device("cuda")
    model.to(device)

    eval_examples = processor.get_dev_examples(eva_data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, 128, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    eval_dataloader = DataLoader(eval_data, batch_size=32)

    eval_accuracy = 0
    nb_eval_examples = 0

    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracyBCE(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += input_ids.size(0)

    eval_accuracy = eval_accuracy / nb_eval_examples

    print("eval_accuracy %f  %d" % (eval_accuracy, nb_eval_examples))


if __name__ == '__main__':
    args = BertParser()
    testMain(args)

    # evaMolde('/home/zjj/Pycharm/PytorchLearning/src/samples/bert-Chinese-classification-task/Bert/qnli_output/bert_QNLI.pt',
    #          '/home/zjj/Pycharm/PytorchLearning/src/samples/bert-Chinese-classification-task/GLUE/glue_data/QNLI/')
