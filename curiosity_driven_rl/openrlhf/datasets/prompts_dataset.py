from torch.utils.data import Dataset
from tqdm import tqdm
import json

default_img = "/home/ma-user/work/haozhe/workspace/lmm-r1/data/DynaMath/assets/dyna-logo.png"
def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    
    return prompt


templates = dict(longcot="""
You are a thoughtful and diligent student tasked with solving a problem. As you work through the problem, document your thought process in a reflective, first-person narrative. Think of yourself as talking to yourself through each step. Consider each step carefully, question your reasoning, and adjust as needed to arrive at a sound solution. Here's how you should proceed:

1. **Step-by-Step Analysis**: Start by thoroughly understanding the problem. Identify what is provided and what is being asked. Consider high-level strategies or approaches first, and then break them down into smaller, manageable steps. Ensure you address each component one at a time and do not skip over any details.
   
2. **Self-Questioning**: As you work through each step, ask yourself reflective questions like, "Is this correct?", "Does it make sense?", or "What might I be overlooking?" Be critical of your own reasoning, and adjust your approach as needed. Use <confidence></confidence> notation to express your confidence and evaluate the progress about solving the problem.
   
3. **Reassessment**: If you notice a mistake or feel uncertain about your approach, reassess your work. Go back and revise your assumptions, logic, or calculations to correct any missteps, ensuring you're on the right track.

4. **Alternative Approaches**: If you find yourself stuck or unsure about the current method, consider alternative approaches. Look at the problem from different angles, and if one method feels insufficient, explore others.

5. **Clear Detailing**: For each step, explain your reasoning clearly and in simple language. Make sure anyone who follows your work can easily understand the logic behind your decisions and the steps you've taken.

6. **Final Solution**: Once you're confident in your solution, enclose it in \\boxed{} to highlight your final answer.

**Your goal is to approach the problem in a reflective, iterative manner, ensuring that no steps are skipped and no assumptions go unchecked.**
""",
vcot="\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For a video, we can look closer by `select_frames`. For an image, we can look closer by `crop_image`. Reason with the visual information step by step, and put your final answer within \\boxed{}.",
notool="Please think step by step, and put your final answer within \\boxed{}.",
default="Please reason step by step, and put your final answer within \\boxed{}.",
elaborate="First understand the problem: understand what information is given in the text and understand what the images describes. Then think about what the problem is asking for and what knowledge the problem aims to examine. Finally, think about how to solve the problem step by step. Explain your solution in simple words that are easy to follow, assuming the readers are junior students who DOT NOT master well the relevant knowledge. Remember to put your final answer within \\boxed{}.",
elaborate_rethink="""Guidelines:
- First understand the problem: understand what information is given in the text and understand what the images describes. Then think about what the problem is asking for and what knowledge the problem aims to examine. Finally, think about how to solve the problem step by step. Explain your solution in simple words that are easy to follow, assuming the readers are junior students who DOT NOT master well the relevant knowledge. 
- **Regularly perform self-questioning, self-verification, self-correction to check your ongoing reasoning**, using connectives such as "Wait a moment", "Wait, does it seem right?", etc.
- Remember to put your final answer within \\boxed{}.""",
explain="""Guidelines:
Understand what the problem is asking for, and what knowledge the problem aims to examine. 
Explain the problem and your solution in simple words to a reader, assuming he has rare knowledge and poor mastery about the related concepts. 
Remember to put your final answer within \\boxed{}.""",
rethink="""Guidelines: 
Please think step by step, and **regularly perform self-questioning, self-verification, self-correction to check your ongoing reasoning**, using connectives such as "Wait a moment", "Wait, does it seem right?", etc. Remember to put your final answer within \\boxed{}.""",
qwen="Please select the correct answer from the options above. If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely\u2014provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
)
templates['none'] = ""
templates['autocode'] = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"



class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def preprocess_data(self, data, input_template=None, input_key="input", apply_chat_template=None, system_prompt="longcot") -> str:
        has_vlm_processor = self.processor is not None
        # print('!!!! apply chat', apply_chat_template)
        # print('!!!! sys', system_prompt, input_key)
        # import pdb; pdb.set_trace()
        # if system_prompt=='dpsk':
        #     # import json
        #     if input_key=='response' and not self.is_eval:
        #         chat = [{"role": "user", "content": data['question']},
        #                 # {"role": "assistant", "content": data['response']}
        #                 ]
                
        #         prompt = data['question'] # self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        #     else:    
        #         input_key = 'messages'
        #         chat = data[input_key]
        #         prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # elif system_prompt=='dsmath':
        #     chat = data['messages']
        #     for entry in chat:
        #         if entry['role']=='user': break 
        #     template = "User:{instruction}\n\nAssistant:"
        #     # entry['content'] += f'\n{templates["default"]}'
            
        #     prompt = template.format(instruction=entry['content'])
        # elif system_prompt=='autocode':
        #     chat = data['messages']
        #     for entry in chat:
        #         if entry['role']=='user': break 
        #     template = templates[system_prompt]
        #     # template = "User:{instruction}\n\nAssistant:"
        #     # entry['content'] += f'\n{templates["default"]}'
            
        #     prompt = template.format(entry['content'])
        # elif input_key=='question':
        #     prompt = data[input_key]
        #     if system_prompt=='default':
        #         trigger = templates[system_prompt]
        #         chat = [{"role": "system", "content": trigger},
        #                 {"role": "user", "content": prompt}]
        #         prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        #     else:
        #         input_template = templates[system_prompt]
        #         prompt = input_template.format(prompt)
        if has_vlm_processor: 
            if False:
                chat = data[input_key]
                if system_prompt in templates:
                    chat.insert(0, dict(role='system', content=templates[system_prompt]))
                else: print(f'!!!! warning: {system_prompt} not in templates')
                if isinstance(chat[-1]['content'], str): 
                    text = chat[-1]['content']
                    content = [
                        # dict(type='image', image=None),
                        dict(type='text', text=text)
                    ]
                    chat[-1]['content'] = content
                
            else:
                # sysp = None
                # if system_prompt in templates:
                #     sysp = templates[system_prompt]
                # else: print(f'!!!! warning: {system_prompt} not in templates')
                # now we don't use system prompt 
                if system_prompt == 'notrigger':
                    trigger = ""
                elif system_prompt == 'elaborate':
                    trigger = f"\n\n{templates['elaborate']}"
                elif system_prompt == 'elaborate_rethink':
                    trigger = f"\n\n{templates['elaborate_rethink']}"
                elif system_prompt == 'rethink':
                    trigger = f"\n\n{templates['rethink']}"
                else: 
                    trigger = f"\n\n{templates[system_prompt]}"
                q = data['question']
                img = data.get('image', None)
                imglist = []
                if img is None or img=="" : 
                    pass # keep it empty
                elif isinstance(img, list): 
                    if data.get('is_video', False): 
                        imglist = [dict(type='video', video=img)] # +[dict(type='image', image='/home/ma-user/work/haozhe/workspace/lmm-r1/muzedata/2000-4000/2000/1.jpg')]
                    else:
                        imglist = [dict(type='image', image=imm) for imm in img if imm]
                else: imglist = [dict(type='image', image=img)]
                # if len(imglist)>10:
                #     print('!!! [debug]', img)
                chat = [dict(role='user', 
                             content=imglist+[dict(type='text', text=q+trigger)] # if img else q
                        )]
                
                if 'qid' in data:
                    chat.append(dict(qid=data['qid']))
            prompt = json.dumps(chat)
        elif input_key=='question':
            chat = [{"role": "system", "content": templates["default"]},
                {"role": "user", "content": data['question']}]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        elif input_key=='messages':
            chat = data[input_key]
            if len(chat)>1: 
                chat[0] = dict(role='system', content=templates[system_prompt]) # replace 
            else: 
                if system_prompt in templates:
                    chat.insert(0, dict(role='system', content=templates[system_prompt]))
                else: print(f'!!!! warning: {system_prompt} not in templates')
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        elif apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            else: # messages 
                # if system_prompt!="none":
                if len(chat)>1: 
                    chat[0] = dict(role='system', content=templates[system_prompt]) # replace 
                else: chat.insert(0, dict(role='system', content=templates[system_prompt]))
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            input_template = templates[system_prompt]
            if system_prompt in ['none']:
                print(f"template cannot be {system_prompt} when not using chat template")
                chat = [dict(role='system', content=templates[system_prompt]),
                        dict(role='user', content=prompt)
                        ]
                prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            else:
                prompt = input_template.format(prompt)
        if prompt=="": print('!!!! warning, prompts incorrect')
        return prompt

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        is_eval=False,
        processor=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_eval = is_eval
        
        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        controlled_shuffle = getattr(self.strategy.args, "controlled_shuffle", 0)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        
        system_prompt = getattr(self.strategy.args, "system_prompt", "none")
        # print("sysprompt", system_prompt)
        do_vlm = getattr(self.strategy.args, "train_vlm", False)
        # import pdb; pdb.set_trace()
        if apply_chat_template:
            apply_chat_template = self.processor.apply_chat_template if do_vlm else self.tokenizer.apply_chat_template
        

        self.prompts = []
        repeat = 1 if controlled_shuffle==0 or is_eval else controlled_shuffle
        for _ in range(repeat):
            for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
                prompt = self.preprocess_data(data, input_template, input_key, apply_chat_template, system_prompt)
                self.prompts.append(prompt)
        # print("!!!! peek", self.prompts[0])
        

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
