import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from flask_deep.utils import init_logger

import glob
import json 

logger = logging.getLogger(__name__)

def read_input_file(pred_config):

    lines = []

    input_txt = pred_config.input_txt
    input_txt = input_txt.split("\n")

    for line in input_txt:

        line = line.strip()
        words = line.split()

        lines.append(words)

    return lines

def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    all_input_tokens = []
    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            
            # use the real label id for all tokens of the word
            slot_label_mask.extend([0] * (len(word_tokens)))

        all_input_tokens.append(tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)

        input_ids = input_ids + ([pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset, all_input_tokens


def predict_server(pred_config):

    # load model and args
    training_params = pred_config.training_params
    
    label_lst = training_params['label_lst']
    args = training_params['training_args']

    model = pred_config.model
    device = pred_config.device
    
    logger.info(args)
    logger.info(label_lst)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = pred_config.tokenizer

    lines = read_input_file(pred_config)
    dataset, all_input_tokens = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    preds = None
    # sms = None
    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": None} # label이 None이므로 
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)

            logits = outputs[0] 

            # sm = torch.nn.functional.softmax(logits, dim=-1)
            # sm = sm.detach().cpu().numpy()

            if preds is None: # the very first one 
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
                # sms = sm
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
                # sms = np.append(sms, axis=0)

    preds = np.argmax(preds, axis=2)


    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    # sms_list = [[] for _ in range(sms.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])
                # sms_list[i].append(sms[i][j])
                
    
    result_dict = {}
    result_dict['result'] = []
    
    for words, preds in zip(all_input_tokens, preds_list):
        
            # words가 문장 단위
            
            line = ""
            
            ahead_tag = ""
            pii_word = ""
            
            final_idx = (len(preds) if len(words) > len(preds) else len(words))-1
            
            for i, (word, pred) in enumerate(zip(words, preds)):
                
                pred = pred.split("-")[0] # BIO tag 제거
                
                if i == 0:
                    ahead_tag = pred
                    
            
                if '#' in word:

                    pii_ended = words.index(word) #list 

                    word = word.strip('#')
                    
                    if pred == ahead_tag:
                        pii_word += word
                        
                    else: 
                        if ahead_tag == 'O':
                            line += pii_word
                        else:
                            
                            line = line + "[{}:{}] ".format(pii_word, ahead_tag)

                            result_dict['result'].append({
                                "token":pii_word.strip(),
                                "tag":ahead_tag,
                                "start":0,
                                "end":0,
                            })
                            
                        pii_word = word
                        
                else:
                    if pred == ahead_tag:
                        # pii_started =
                        pii_word = pii_word + " " + word

                    else:
                        if ahead_tag == 'O':
                            line += pii_word
                        else:
                            line = line + "[{}:{}] ".format(pii_word, ahead_tag)
                            
                            result_dict['result'].append({
                                "token":pii_word.strip(),
                                "tag":ahead_tag,
                                "start":0,
                                "end":0,
                            })
                            
                        pii_word = word
                        
                ahead_tag = pred
                
                if i == final_idx:
                    if pred == 'O':
                        line += pii_word
                    else:
                        line = line + "[{}:{}] ".format(pii_word, ahead_tag)

    logger.info("Prediction Done!")
    
    # korean: ensure_ascii = False
    result_json = json.dumps(result_dict, indent="\t", ensure_ascii=False)
    
    return result_json


class model_conf:
    
    input_txt = None
    batch_size = 32
    no_cuda = False
    model = None
    model_dir = None
    device = None
    training_params = None
    tokenizer = None
    
    def __init__(self, input_txt, model_dict):
        
        self.input_txt = input_txt
        self.model = model_dict['model']
        self.model_dir = model_dict['model_dir']
        self.device = model_dict['device']
        self.training_params = model_dict['training_params']
        self.tokenizer = model_dict['tokenizer']

# run via server
def main(input_txt, model_dict):
    
    init_logger()

    pred_config = model_conf(input_txt, model_dict)
    result_dict = predict_server(pred_config)

    return result_dict