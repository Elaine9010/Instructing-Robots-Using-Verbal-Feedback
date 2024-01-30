# This code is used to evaluate the code-as-policy model performance with wizardcoder-backbone.
# The evaluation is based on the following metrics:
# 1. Success rate on seen/unseen instructions
# 2. Success rate on seen/unseen attributs
# The evaluation is based on the following parameters:
# 1. model_size: the size of the model, e.g. 7B
# 2. temperature: the temperature used in the model
# 3. prompt_version: the version of the prompt used in the model

import sys
import os
import json
import numpy as np
from code_as_policy import *
from vllm import LLM, SamplingParams
from moviepy.editor import ImageSequenceClip
from pprint import pprint

if len(sys.argv) != 5:
    print("Usage: python script.py <model_size> <temperature> <instruction_type> <attribute_type>")
    sys.exit(1)
# parameters
model_size = sys.argv[1] # 7B or 13B
temperature = float(sys.argv[2]) # 0, 0.5 or 1
dist_threshold = 0.11
prompt_version = "v3"
instruction_type = sys.argv[3] # seen or unseen
attribute_type = sys.argv[4] # seen or unseen


# instructions
seen_instructions = [
    "put the <block> on the <bowl>",
    "move the <block> to the <side>",
    "place the <nth> block from the <direction> on the top left corner",
    "put the <block> in the bowl with matching color",
    "put all the blocks on the <side>",
]
unseen_instructions =[
    "pick up the <block> and place it on the <bowl>",
    "place the <block> on the <side>",
    "find the <nth> block from the <direction> and move it to the top left corner",
    "put the <block> in the bowl with the same color",
    "move all the blocks to the <side> of the table",
]
seen_attributes = {
    "block": ['blue block', 'red block', 'green block', 'orange block', 'yellow block' ],
    "bowl": ['blue bowl', 'red bowl', 'green bowl', 'orange bowl', 'yellow bowl'],
    "side": ["left side", "top side"],
    "nth": ["first", "third"],
    "direction": ["left", "top"]
}
unseen_attributes = {
    "block": ['pink block', 'cyan block', 'brown block', 'gray block', 'purple block' ],
    "bowl": ['pink bowl', 'cyan bowl', 'brown bowl', 'gray bowl', 'purple bowl' ],
    "side": ["right side", "bottom side"],
    "nth": ["second"],
    "direction": ["right", "bottom"]
}

def set_up_scene(instruction_template, attribute_type):
    #create a list of objects including 3 blocks and 3 bowls according to the attribute type.
    #replace the attributes in the instruction template with the corresponding attribute.
    #return the instruction and the obj_list
    np.random.seed()
    #print("//////////////// check random seed: ", np.random.get_state())
    instruction = instruction_template
    obj_list = []   
    if attribute_type == "seen":
        attributes = seen_attributes
    elif attribute_type == "unseen":
        attributes = unseen_attributes
    else:
        print("ERROR: attribute_type should be seen or unseen!")
        return
    block_list = np.random.choice(attributes["block"], size=3, replace=False).tolist()
    bowl_list = block_list.copy()
    bowl_list = [bowl.replace("block", "bowl") for bowl in bowl_list]
    obj_list = block_list + bowl_list
    for key, value in attributes.items():
        if key == "block" or key == "bowl":
            continue
        else:
            instruction = instruction.replace(f"<{key}>", np.random.choice(value))
    instruction = instruction.replace("<block>", np.random.choice(block_list))
    instruction = instruction.replace("<bowl>", np.random.choice(bowl_list))
    return instruction, obj_list

def find_nth_block(direction_name, nth_idx, obj_list):
    #find the nth block from the direction 
    block_list = [obj for obj in obj_list if "block" in obj]
    if direction_name == "left":
        block_name = sorted(block_list, key=lambda x: env.get_obj_pos(x)[0])[nth_idx]
    elif direction_name == "bottom":
        block_name = sorted(block_list, key=lambda x: env.get_obj_pos(x)[1])[nth_idx]
    elif direction_name == "right":
        block_name = sorted(block_list, key=lambda x: env.get_obj_pos(x)[0], reverse=True)[nth_idx]
    elif direction_name == "top":
        block_name = sorted(block_list, key=lambda x: env.get_obj_pos(x)[1], reverse=True)[nth_idx]
    return block_name

def set_up_answer(env, idx, instruction_type, instruction, obj_list):
    #set up the desired objects positions for the given instruction
    #return the desired objects positions
    desired_objs_pos = {}
    for obj in obj_list:
        desired_objs_pos[obj] = env.get_obj_pos(obj)
    if instruction_type == "seen":
        if idx == 0:
            instruction_template = "put the <block> on the <bowl>"
            block_name = instruction.split("put the ")[1].split(" on the ")[0]
            bowl_name = instruction.split(" on the ")[1]
            desired_objs_pos[block_name] = env.get_obj_pos(bowl_name)
        elif idx == 1:
            instruction_template = "move the <block> to the <side>"
            block_name = instruction.split("move the ")[1].split(" to the ")[0]
            corner_name = instruction.split(" to the ")[1]
            desired_objs_pos[block_name] = env.get_obj_pos(corner_name)
        elif idx == 2:
            instruction_template = "place the <nth> block from the <direction> on the top left corner"
            nth_name = instruction.split("place the ")[1].split(" block from the ")[0]
            nth_idx_map = {"first": 0, "second": 1, "third": 2}
            nth_idx = nth_idx_map.get(nth_name, 0)
            direction_name = instruction.split(" block from the ")[1].split(" on the ")[0]
            block_name = find_nth_block(direction_name, nth_idx, obj_list)
            desired_objs_pos[block_name] = env.get_obj_pos("top left corner")
        elif idx == 3:
            instruction_template = "put the <block> in the bowl with matching color"
            block_name = instruction.split("put the ")[1].split(" in the ")[0]
            bowl_name = block_name.replace("block", "bowl")
            desired_objs_pos[block_name] = env.get_obj_pos(bowl_name)
        elif idx == 4:
            instruction_template = "put all the blocks on the <side>"
            corner_name = instruction.split("put all the blocks on the ")[1]
            block_list = [obj for obj in obj_list if "block" in obj]
            for block_name in block_list:
                desired_objs_pos[block_name] = env.get_obj_pos(corner_name)

    elif instruction_type == "unseen":
        if idx == 0:
            instruction_template = "pick up the <block> and place it on the <bowl>"
            block_name = instruction.split("pick up the ")[1].split(" and place it on the ")[0]
            bowl_name = instruction.split(" and place it on the ")[1]
            desired_objs_pos[block_name] = env.get_obj_pos(bowl_name)
        elif idx == 1:
            instruction_template = "place the <block> on the <side>"
            block_name = instruction.split("place the ")[1].split(" on the ")[0]
            corner_name = instruction.split(" on the ")[1]
            desired_objs_pos[block_name] = env.get_obj_pos(corner_name)
        elif idx == 2:
            instruction_template = "find the <nth> block from the <direction> and move it to the top left corner"
            nth_name = instruction.split("find the ")[1].split(" block from the ")[0]
            nth_idx_map = {"first": 0, "second": 1, "third": 2}
            nth_idx = nth_idx_map.get(nth_name, 0)
            direction_name = instruction.split(" block from the ")[1].split(" and move it")[0]
            block_name = find_nth_block(direction_name, nth_idx, obj_list)
            desired_objs_pos[block_name] = env.get_obj_pos("top left corner")
        elif idx == 3:
            instruction_template = "put the <block> in the bowl with the same color"
            block_name = instruction.split("put the ")[1].split(" in the ")[0]
            bowl_name = block_name.replace("block", "bowl")
            desired_objs_pos[block_name] = env.get_obj_pos(bowl_name)
        elif idx == 4:
            instruction_template = "move all the blocks to the <side> of the table"
            corner_name = instruction.split("all the blocks to the ")[1].split(" of the table")[0]
            block_list = [obj for obj in obj_list if "block" in obj]
            for block_name in block_list:
                desired_objs_pos[block_name] = env.get_obj_pos(corner_name)
    return desired_objs_pos


# load all prompts
with open(f"configs/prompts_{prompt_version}.json", "r") as f:
    promptsDict = json.load(f)

# Creating variables dynamically
for key, value in promptsDict.items():
    globals()[key] = value

# print word count in each prompt
prompt_names = [
    "prompt_tabletop_ui",
    "prompt_parse_position",
    "prompt_parse_obj_name",
    "prompt_parse_question",
    "prompt_fgen",
    "prompt_transform_shape_pts",
]
prompt_word_count = {}
for name in prompt_names:
    prompt_word_count[name] = len(eval(name).split(" "))
    print(name, prompt_word_count[name])

cfg_tabletop = {
    "lmps": {
        "tabletop_ui": {
            "prompt_text": prompt_tabletop_ui,
            "engine": "wizardcoder",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["<EOS>","'''"],
            "maintain_session": True,
            "debug_mode": False,
            "include_context": True,
            "has_return": False,
            "return_val_name": "ret_val",
        },
        "parse_obj_name": {
            "prompt_text": prompt_parse_obj_name,
            "engine": "wizardcoder",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#","<EOS>"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "parse_position": {
            "prompt_text": prompt_parse_position,
            "engine": "wizardcoder",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#","<EOS>"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "parse_question": {
            "prompt_text": prompt_parse_question,
            "engine": "wizardcoder",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#", '<EOS>'],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "transform_shape_pts": {
            "prompt_text": prompt_transform_shape_pts,
            "engine": "wizardcoder",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "new_shape_pts",
        },
        "fgen": {
            "prompt_text": prompt_fgen,
            "engine": "wizardcoder",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# define function: ",
            "query_suffix": ".",
            "stop": ["#", "<EOS>"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
        },
    }
}

# setup LLM
base_model = "WizardLM/WizardCoder-Python-{f}-V1.0".format(f=model_size)
llm = LLM(model=base_model, tensor_parallel_size=1)

# setup env
high_resolution = False
high_frame_rate = False

# success rate

instruction_templates = seen_instructions if instruction_type == "seen" else unseen_instructions
with open(f"results/B_eval_result_{model_size}_{temperature}.txt","a") as f:
    f.write("--------------------------------------------------------------------------------------------------\n")
    f.write(f"model_size: {model_size}, temperature: {temperature}, prompt_version: {prompt_version}")
    f.write("\n")
    f.write(f"Instruction type: {instruction_type}, Attribute type: {attribute_type}")
    f.write("\n")
    f.write("--------------------------------------------------------------------------------------------------\n")

syntatic_errors = []
non_syntatic_errors = []

for idx, instruction_template in enumerate(instruction_templates):
    successes = []
    for i in range(20):
        print(f"INFO: Processing {idx+1}th instruction for the {i+1}th time.")
        # setup env and LMP
        env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
        instruction, obj_list = set_up_scene(instruction_template, attribute_type)
        print("Available objects: ", obj_list)
        _ = env.reset(obj_list)  
        lmp_tabletop_ui = setup_LMP(env, llm, cfg_tabletop)

        #setup desired objects position for given instruction
        desired_objs_pos = set_up_answer(env, idx, instruction_type,instruction, obj_list)

        #prompt the LMP to execute the instruction
        try:
            lmp_tabletop_ui(instruction, f"objects = {env.object_list}")
        except Exception as e:
            print("WARNING: LMP failed to execute. ", e)
            successes.append(False)
            with open(f"results/B_eval_result_{model_size}_{temperature}.txt","a") as f:
                f.write(f"Instruction: {instruction}\n")
                f.write(f"Failed with error {e}\n")
            syntatic_errors.append(str(idx+1)+"th instruction  error: "+str(e))
            continue
        
        # calculate euclidean distance between current object pos and desired object pos
        max_dist = 0
        for obj in obj_list:
            current_obj_pos = env.get_obj_pos(obj)
            desired_obj_pos = desired_objs_pos[obj]
            dist = np.linalg.norm(np.array(current_obj_pos) - np.array(desired_obj_pos))
            if dist > max_dist:
                max_dist = dist
                max_dist_obj = obj
        if max_dist < dist_threshold:
            successes.append(True)
            print(f"INFO: {idx+1}th instruction success!")
            with open(f"results/B_eval_result_{model_size}_{temperature}.txt","a") as f:
                f.write(f"Instruction: {instruction}\n")
                f.write(f"Success!\n")
        else:
            successes.append(False)
            print(f"INFO: {idx+1}th instruction failed! max_dist: {max_dist}")
            with open(f"results/B_eval_result_{model_size}_{temperature}.txt","a") as f:
                f.write(f"Instruction: {instruction}\n")
                f.write(f"Failed with error {max_dist}\n")
                f.write(f"max_dist_obj: {max_dist_obj}\n")
            non_syntatic_errors.append(str(idx+1)+"th instruction:"+str(instruction)+f" Failure case{idx}_{i+1}")
                
            # save failure video
            if not os.path.exists("failure_videos"):
                os.makedirs("failure_videos")
            output_file = f"failure_videos/failure_case_{idx+1}_{i+1}.mp4"
            if env.cache_video:
                rendered_clip = ImageSequenceClip(
                    env.cache_video, fps=35 if high_frame_rate else 25
                )
                rendered_clip.write_videofile(output_file, codec="libx264")
                print(f"Video saved to {output_file}")
        print("-----------------------------------------------------------------------------------")
    print("###################################################################################")
    print("INFO: Instruction: ", instruction_template)
    print("INFO: Successes: ", successes)
    success_rate = sum(successes)/len(successes)
    print(f"INFO: Success rate for {idx+1}th instruction: {success_rate}")
    print("###################################################################################")
    with open(f"results/B_eval_result_{model_size}_{temperature}.txt","a") as f:
        f.write(f"Instruction: {instruction_template}\n")
        f.write(f"Successes: {successes}\n")
        f.write(f"Success rate for {idx+1}th instruction: {success_rate}\n")
        f.write("----------------------------------------------------------------------------------\n")
with open(f"results/B_eval_errors_{model_size}_{temperature}.txt","a") as f:
    f.write(f"{len(syntatic_errors)} Syntatic errors:\n")
    f.write("\n".join(syntatic_errors))
    f.write("\n")
    f.write("----------------------------------------------------------------------------------\n")
    f.write(f"{len(non_syntatic_errors)} Non-syntatic errors:\n")
    f.write("\n".join(non_syntatic_errors))
    f.write("\n")
    f.write("----------------------------------------------------------------------------------\n")