"""
 * Copyright (c) 2025.
 * All rights reserved.
 * Code for SayNext project
 * Part from: InterVL
"""

import numpy as np
import torch
import json

IGNORE_INDEX = -100

# Edit  target_len

def concat_pad_data_collator(features, pad_id=0, target_len =  20, question = False):  

    first = features[0]
    batch = {}

    if 'prim_target' in first and first['prim_target'] is not None:
        patch_counts = [f['pixel_values'].shape[0] for f in features]
        batch['patch_counts'] = torch.tensor(patch_counts, dtype=torch.long)
      
        prim_tensors = []
        for f in features:
            vec = f['prim_target']            
            if isinstance(vec, str):
                vec = json.loads(vec)
            if len(vec) < target_len:
                vec = vec + [0.0] * (target_len - len(vec))
            elif len(vec) > target_len:
                vec = vec[:target_len]            

            # vec: list of length 20
            t = torch.tensor(vec, dtype=torch.float).view(-1, 1)  # [20,1]
            prim_tensors.append(t)
        batch['prim_target'] = torch.stack(prim_tensors, dim=0)  # [batch,20,1]



    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        if question:
            temp_question_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_question_ids[:feat['question_ids'].shape[0]] = feat['question_ids']
            feat['question_ids'] = temp_question_ids

        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)


    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch

def pad_data_collator(features, pad_id=0, target_len =  20):

    first = features[0]
    batch = {}

    if 'prim_target' in first and first['prim_target'] is not None:
        prim_tensors = []
        for f in features:
            vec = f['prim_target']
            if isinstance(vec, str):
                vec = json.loads(vec)
            t = torch.tensor(vec, dtype=torch.float)    # shape [20]
            t = t.view(-1, 1)                           # shape [20,1]
            prim_tensors.append(t)
        batch['prim_target'] = torch.stack(prim_tensors, dim=0)

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


