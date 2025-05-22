import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque

from .experience_maker import Experience
from .data_processor import BaseDataProcessor


common_keys = (
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "validity"
    )

@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    visual_inputs: Optional[dict]
    validity: Optional[float] = None


def split_experience_batch(experience: Experience, data_processor: Optional[BaseDataProcessor]) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = common_keys
    flags = [False for _ in range(batch_size)]
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    if data_processor is not None:
        visual_inputs_batch = experience.visual_inputs
        visual_inputs_batch['input_ids'] = experience.sequences
        visual_inputs_chunks = data_processor.split_input_batch(visual_inputs_batch)
        # used some techniques to identify troublesome examples 
        last_valid_placeholder = None
        for idx, entry in enumerate(visual_inputs_chunks):
            if entry is None:
                flags[idx] = True
            elif entry['input_ids'] is None:
                flags[idx] = True
            else:
                last_valid_placeholder = entry 
        for idx, flag in enumerate(flags):
            if flag:
                visual_inputs_chunks[idx] = last_valid_placeholder
        # Note: if all entries are invalid, there will be bug
        for i, visual_inputs in enumerate(visual_inputs_chunks):
            if visual_inputs is None or 'input_ids' not in visual_inputs: continue 
            visual_inputs.pop('input_ids')
            batch_kwargs[i]["visual_inputs"] = visual_inputs
    else:
        for i in range(batch_size):
            batch_kwargs[i]["visual_inputs"] = None

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    
    for k, v in experience.info.items():
        if v is None: continue
        elif isinstance(v, list): vals = v 
        else:
            # print('!!!! debug', k, type(v))
            vals = torch.unbind(v)
        assert batch_size == len(vals),f"info key {k} shape {len(vals)} vs target {batch_size} incorrect"
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    # find the first valid item, and replace all invalid entries with it
    num_invalid = sum(flags)
    invalid_placeholders = []
    while len(invalid_placeholders) < num_invalid:
        for i, entry in enumerate(batch_kwargs):
            if not flags[i]: # a valid item 
                invalid_placeholders.append(entry)
    invalid_index = 0
    for i, entry in enumerate(batch_kwargs):
        if flags[i]: 
            batch_kwargs[i] = invalid_placeholders[invalid_index]
            invalid_index += 1
    
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    if not isinstance(sequences[0], torch.Tensor): return sequences
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], data_processor: Optional[BaseDataProcessor], packing_samples=False) -> Experience:
    kwargs = {}
    keys = common_keys
    # print('!!!! make batch', [x.advantages for x in items])
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    # converting to tensors
    # print('!!!! debug', items[0].info.keys())
    to_convert = {'reward','validity','return'}
    to_skip = {'question'}
    print(f"===> [verbose] make_experience_batch() making exp.info, converting {to_convert}, skipping {to_skip}")
    
    for key in items[0].info.keys():
        if key in to_skip: continue
        tmplist = [item.info[key] for item in items]
        # dict_keys(['reward', 'response_length', 'total_length', 'validity', 'norepeat', 'usefmt', 'match', 'use_codes', 'round0_nwait', 'round1_nwait', 'round0_correctness', 'round1_correctness', 'qids', 'kl', 'wait_bonus', 'difficulty', 'solve_all', 'solve_none', 'easy', 'hard', 'medium', 'return'])
        # if key in {'difficulty','response_length','total_length','kl','solve_all',"solve_none",'easy','hard','medium','num_switch','use_codes','usefmt','norepeat'}:
        if key in to_convert:
            vals = torch.tensor(tmplist)
        else: vals = tmplist 
        
        kwargs["info"][key] = vals
    if data_processor is not None:
        kwargs["visual_inputs"] = data_processor.make_input_batch([item.visual_inputs for item in items])
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask, # contains what follows the query, so action_mask is not the same shape with seq
        )
        # right_pad = (1 - act_mask.long()).sum()
        # right_pad = torch.flip(act_mask, (0,)).float().argmax()
        right_pad = 0 # now don't find the last valid in act_mask, because we need padded vision tokens to mix data
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax() # the first position that is 1 
        (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            None if act_log_prob is None else act_log_prob[:right_pad],
            value[:right_pad] if item.values is not None else None,
            None if ret is None else ret[:right_pad],
            None if adv is None else adv[:right_pad],
            None if att_mask is None else att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
        # if item.sequences[-1]==151655:
        #     breakpoint()
    return items

def shuffle_questions(data):
    """
    Given a list of question strings, 
    first obtain question to index list,
    then shuffle the questions,
    and return the index list based on the shuffled questions

    Args:
        data: A list of question strings.

    Returns:
        A list of indices corresponding to the shuffled questions.
    """
    # Create a dictionary to map questions to their indices
    q2idx = defaultdict(list)
    for index, question in enumerate(data):
        q2idx[question].append(index)
    # question_to_index = {question: index for index, question in enumerate(data)}

    # Shuffle the questions
    shuffled_questions = list(data)
    np.random.shuffle(shuffled_questions)

    # Create a list of indices based on the shuffled questions
    shuffled_indices = []
    for question in shuffled_questions:
        shuffled_indices.extend(q2idx[question])
    
    return shuffled_indices

def separate_and_shuffle_questions(questions, diffs, seed=None):
    """
    Separates a list of questions into two sets based on corresponding scores,
    sorts the non-zero set by scores (high to low), and then shuffles it
    using the provided seed.

    Args:
        questions: A list of questions.
        diffs: A list of scores corresponding to the questions.
        seed: An optional integer seed for shuffling.

    Returns:
        A tuple containing two lists:
            - non_zero_questions: A list of questions with non-zero scores,
                                  sorted by score (high to low) and then shuffled.
            - zero_questions: A list of questions with zero scores.
    """
    if len(questions) != len(diffs):
        raise ValueError("The lengths of questions and diffs lists must be equal.")

    non_zero_items = []
    zero_questions = []
    non_zero_questions = []

    for question, diff in zip(questions, diffs):
        # if 0.05<diff <0.95 or (diff>0.95 and np.random.uniform()<0.3): # allneg or allcorrect
        #     non_zero_items.append((question, diff))
        # else:
        #     zero_questions.append(question)
        if diff==0 and np.random.uniform()>0.8:
            zero_questions.append(question)
        else:
            non_zero_questions.append(question)
            
        

    # Sort non-zero questions based on diffs in descending order
    # non_zero_items.sort(key=lambda item: item[1], reverse=True)
    # non_zero_questions = [item[0] for item in non_zero_items]

    # Shuffle the sorted non-zero questions using the seed if provided
    if seed is not None:
        random.seed(seed)
        random.shuffle(non_zero_questions)

    return non_zero_questions, zero_questions

class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, 
        sample_batch_size: int, 
        data_processor: Optional[BaseDataProcessor] = None, 
        limit: int = 0, 
        cpu_offload: bool = True, 
        packing_samples: bool = False,
        drop_maxlen: bool = False,
        maxlen: int = 10**8,
        train_batch_size: int = 64, 
        use_pos: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.data_processor = data_processor
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.keep_items = None 
        self.maxlen = maxlen
        self.drop_maxlen = drop_maxlen
        self.eval_items: List[BufferItem] = []
        self.shuffled_indexes = None
        self.sample_num = 0
        self.train_batch_size = train_batch_size 
        self.use_pos = use_pos
        

    @torch.no_grad()
    def append_split(self, experience: Experience, is_sft=False, is_eval=False) -> None:
        # print('!!!! append', experience.validity)
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))

        items = split_experience_batch(experience, self.data_processor)
        # print('!!!! debug, peek append item info keys', items[0].info.keys())
        # NOTE: No tested
        if False: # self.drop_maxlen
            original_len = len(items)
            items = list(filter(lambda x: x.sequences.shape[-1] <= self.maxlen, items))
            if original_len - len(items) > 0:
                print(f"drop {original_len - len(items)} samples")
        # the packed samples comes with no padding
        if not self.packing_samples and not is_sft and not is_eval:
            items = remove_padding_in_sequences(items)
        if is_sft:
            # print('!!!! add sft samples', len(items))
            self.sft_items.extend(items)
        if is_eval:
            # print('!!!! add eval samples', len(items))
            self.eval_items.extend(items)
        else: self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()
        # self.sft_items.clear()
        self.eval_items.clear()
        self.shuffled_indexes = None
        self.sample_num = 0

    @torch.no_grad()
    def sample(self) -> Experience:
        # if self.shuffled_indexes:
        #     num_total = len(self.shuffled_indexes)//self.sample_batch_size
        #     ptr = self.sample_num % num_total
        #     idxes = self.shuffled_indexes[self.sample_batch_size*ptr:self.sample_batch_size*(ptr+1)]
        #     items = [self.items[idx] for idx in idxes]
        # else: 
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.data_processor, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        # if self.shuffled_indexes:
        #     idx = self.shuffled_indexes[idx]
        # print('!!!! getitem', idx, self.items[idx].info['question'])
        
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.data_processor, self.packing_samples)
        return experience

    def prepare_shuffle(self):
        questions = [item.info['question'] for item in self.items]
        shuffled_indexes = shuffle_questions(questions)
        self.shuffled_indexes = shuffled_indexes
    
    def active_sampling(self, do_filter=False, do_ssr=False):
        print(f'!!!! [debug] shuffling for filter mode, num items = {len(self.items)}')
        numkeep = len(self.items)
        if self.keep_items is None:
            self.keep_items = deque(maxlen=numkeep)
        
        questions = np.arange(len(self.items))
        diffs = [item.advantages[-1].item() for item in self.items]
        rewards = [item.info['reward'] for item in self.items] # already float
        mean_rewards = np.mean([item.info['round0_correctness'] for item in self.items])
        non_zero_items = []
        zero_questions = []
        pos_items = []
        non_zero_questions = []
        stats = {}
        print([ii.info['uniformity'] for ii in self.items])
        for question, item in enumerate(self.items):
            is_uniform = item.info['uniformity']>0.5 
            if is_uniform: 
                zero_questions.append(question)
            else:
                non_zero_questions.append(question)
        # for question, diff, rew in zip(questions, diffs, rewards):
        #     if diff==0:
        #         if rew>0.95:
        #             pos_items.append(question)
        #         zero_questions.append(question)
        #     else:
        #         non_zero_questions.append(question)
        self.keep_items.extend([self.items[ii] for ii in non_zero_questions])
        zero_percentage = len(zero_questions)/len(questions)
        ret_info = {"mean_reward": mean_rewards, 
                    "initial_saturation": zero_percentage}
        for k in ['ALLTrue','ALLFalse','Easy', 'Medium','Hard']:
            ret_info[f'initial_{k}'] = 0.
            stats[f'initial_{k}'] = np.mean([item.info[f'round0_{k}'] for item in self.items])
         
        ret_info.update(stats)
        seed = 42
        random.seed(seed)
        random.shuffle(non_zero_questions)
        idxlist = non_zero_questions+zero_questions
        numtotal = len(idxlist)//2
        if not do_filter : return ret_info
        # if do_ssr and len(self.keep_items)<numtotal: 
        #     print(f'!!!! [debug] {len(self.keep_items)} not enough for SSR')
        #     return ret_info
        
        print(f'!!!! [debug] {len(zero_questions)}/{len(idxlist)} qas have zero advantages, useless')
        ratio = len(zero_questions)/len(idxlist)
        if not do_ssr: 
            if len(non_zero_questions)>0:
                num_repeat = len(self.items)//len(non_zero_questions)
                new_items = []
                for _ in range(num_repeat):
                    random.shuffle(non_zero_questions)
                    new_items.extend(non_zero_questions)
                
                num_remain = len(self.items) % len(non_zero_questions)
                if num_remain>0:
                    random.shuffle(non_zero_questions)
                    new_items.extend(non_zero_questions[:num_remain])   
                self.items = [self.items[idx] for idx in new_items]
            return ret_info
            
        else:
            #######################
            # numtotal = len(idxlist)//2
            # ratio = 0.1 # 0.1  if self.use_pos else 0.0
            # num_pos = int(ratio*len(non_zero_questions))
            # if len(non_zero_questions)==0: # there are no non-zero questions 
            #     num_pos = min(numtotal, len(pos_items))
            #     print(f'!!!! [debug] warning:  because non-zero questions {len(non_zero_questions)} qas is scarce, we include more positive items')
            # elif len(non_zero_questions) < numtotal//2:
            #     print(f'!!!! [debug] warning:  because non-zero questions {len(non_zero_questions)} qas is scarce, we include more positive items')
            #     num_pos = min(num_pos, numtotal//2-len(non_zero_questions))
            
            # sel = non_zero_questions + pos_items[:num_pos]
            # if len(non_zero_questions)==0:
            #     sel = idxlist 
            #     print(f"!!!! [warning] queries all-zero")
            # sel_alist = np.array([abs(self.items[idx].advantages[0].item())+1e-4 for idx in sel])
            # sel_p = sel_alist/np.sum(sel_alist)
            # newlist = []
            # numiter = 0
            # while numtotal>0:
            #     if numtotal>len(sel):
            #         newlist.extend(sel)
            #     else:
            #         newlist.extend(np.random.choice(sel, size=numtotal, p=sel_p if do_ssr else None))
            #     numiter += 1
            #     numtotal -= len(sel)
            # print(f"!!!! [debug] SSR={do_ssr}, replay buffer repeat for {numiter} times to fill up nonzero questions")
            # self.items = [self.items[idx] for idx in newlist]
            ################
            
            print(f'!!!! [debug] warning: in filter mode, the remaining non-zero is too scarce, {len(non_zero_questions)} qas will repeat to {numtotal}')
            ratio = 0.0
            current_effective = []
            
            ########### we don't use sampling from current effective now
            num_pos = int(ratio*len(non_zero_questions))
            # sel = non_zero_questions + pos_items[:num_pos]
            sel = non_zero_questions[:numtotal]
            current_effective = [self.items[ii] for ii in sel]
            # if len(current_effective)>0:
            #     ######### added sampling from current_effective
            #     sel_alist = np.array([abs(iitem.advantages[0].item()) for iitem in current_effective])
            #     sel_p = sel_alist/np.sum(sel_alist)
            #     if sel_p.sum()<1.0:
            #         sel_p[-1] = 1.0 - np.sum(sel_p[:-1]) # make sure the sum == 1
            #     # print(f"{sel_p};{sel_alist}")
            #     # assert sel_p.sum()==1.0, f"debug: {sel_p};{np.sum(sel_alist)};{len(current_effective)}"
            #     current_selected = np.random.choice(np.arange(len(current_effective)), size=min(len(idxlist)//2,len(current_effective)), p=sel_p)
            #     current_effective = [current_effective[ii] for ii in current_selected]
            print(f'!!!! [debug] current effective {len(current_effective)} qas')
            ################
            sel = np.arange(len(self.keep_items))
            numtotal -= len(current_effective)
            sel_alist = np.array([abs(iitem.advantages[0].item())+1e-4 for iitem in list(self.keep_items)])*1.0
            sel_p = sel_alist/np.sum(sel_alist)
            newlist = []
            numiter = 0
            while numtotal>0:
                if numtotal>len(sel):
                    newlist.extend(sel)
                else:
                    newlist.extend(np.random.choice(sel, size=numtotal, p=sel_p))
                numiter += 1
                numtotal -= len(sel)
            print(f"!!!! [debug] SSR={do_ssr}, replay buffer repeat for {numiter} times to fill up nonzero questions")
            augmented = [self.keep_items[idx] for idx in newlist]
            self.items = current_effective + augmented
            return ret_info
        
    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            after = (items[i] - mean) * rstd
            before = items[i]
            print('!!!! normalize: na', len(getattr(item, 'action_log_probs')), item.info, f'rloo adv={before[-1]}->{after[-1]}')
            match = getattr(item, 'match', 0.0)
            if match>0.5 and after[-1]<0: continue 
            setattr(item, attribute, after)
            # setattr(item, attribute, (items[i] - mean) * rstd + 1e-8)
