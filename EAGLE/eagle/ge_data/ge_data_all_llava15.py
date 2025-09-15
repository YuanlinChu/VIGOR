import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--image_data_path', type=str, default=None)
parser.add_argument('--json_data_path', type=str, default=None)

args = parser.parse_args()

bigname=args.model
image_data_path=args.image_data_path
json_data_path=args.json_data_path

# 添加调试信息，打印参数值
print(f"参数信息: model={bigname}, image_data_path={image_data_path}, json_data_path={json_data_path}")
print(f"image_data_path类型: {type(image_data_path)}")
print(f"image_data_path是否为None: {image_data_path is None}")
print(f"image_data_path的值: '{image_data_path}'")
# 如果image_data_path是空字符串或字符串"None"，将其设置为None
if image_data_path == "" or image_data_path == "None":
    print(f"image_data_path是'{image_data_path}'，将其设置为None")
    image_data_path = None

import os
# args.gpu_index.append(args.gpu_index[0] + 4) 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init

from PIL import Image
import requests
from io import BytesIO

def my_process_images(images, image_processor, model_cfg):
    image_aspect_ratio = model_cfg.get("image_aspect_ratio", "original")
    new_images = []
    if image_aspect_ratio == "original":
        # Process images while keeping their original size
        for image in images:
            # Explicitly disable resizing, center cropping, and enforce keeping original dimensions
            image_tensor = image_processor.preprocess(
                image, 
                do_resize=False, 
                do_center_crop=False, 
                return_tensors='pt'
            )['pixel_values'][0]
            new_images.append(image_tensor)
    elif image_aspect_ratio == "448":

        target_size = {"height": 448, "width": 448}
        for image in images:
            image_tensor = image_processor.preprocess(
                image, 
                do_resize=True,  
                size=target_size,  
                do_center_crop=False,
                return_tensors='pt'
            )['pixel_values'][0]
            new_images.append(image_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def load_image(image_file):
    try:
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    except FileNotFoundError:
        print(f"警告：图片文件 {image_file} 不存在，将跳过此条数据")
        return None
    except Exception as e:
        print(f"警告：加载图片 {image_file} 时出错: {e}，将跳过此条数据")
        return None

def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files=json_data_path)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 1

    def preprocess_function(examples):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "loss_mask": [],
            "image": [],
            "image_size": []
        }
        for i in range(len(examples['id'])):
            try:
                conv_mode = "llava_v1"
                conv = conv_templates[conv_mode].copy()
                conv.system = ""
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
                
                # 添加调试信息
                print(f"处理对话 {i}")
                
                # 检查对话是否为空
                if 'conversations' not in examples or i >= len(examples['conversations']) or not examples['conversations'][i]:
                    print(f"警告：对话 {i} 为空或不存在，跳过")
                    continue
                    
                source = examples['conversations'][i]
                print(f"对话 {i} 长度: {len(source)}")
                
                # 检查第一条消息是否存在
                if not source or len(source) == 0:
                    print(f"警告：对话 {i} 没有消息，跳过")
                    continue
                    
                # 检查第一条消息的格式
                if "from" not in source[0]:
                    print(f"警告：对话 {i} 的第一条消息格式错误，跳过")
                    continue
                    
                if roles[source[0]["from"]] != conv.roles[0]:
                    print(f"对话 {i} 不是以人类开始，调整顺序")
                    source = source[1:]
                    
                # 检查调整后的对话是否为空
                if not source or len(source) == 0:
                    print(f"警告：调整后对话 {i} 没有消息，跳过")
                    continue
                    
                conv.messages = []
                for j, sentence in enumerate(source):
                    # 检查消息格式
                    if "from" not in sentence or "value" not in sentence:
                        print(f"警告：对话 {i} 的第 {j} 条消息格式错误，跳过此消息")
                        continue
                        
                    role = roles[sentence["from"]]
                    # 放宽角色顺序限制，只打印警告而不中断
                    expected_role = conv.roles[j % 2]
                    if role != expected_role:
                        print(f"警告：对话 {i} 的第 {j} 条消息角色顺序不符合预期 (预期 {expected_role}, 实际 {role})")
                    
                    if sentence["from"] == "gpt":
                        sentence["value"] = " " + sentence["value"]
                    conv.append_message(role, sentence["value"])
            except Exception as e:
                print(f"处理对话 {i} 时出错: {str(e)}")
                continue
            conversation = conv.get_prompt()
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            image_tensor = None
            image_size = None
            # 检查是否需要处理图像
            try:
                # 使用全局变量，避免局部变量引用错误
                global image_data_path
                # 再次检查image_data_path是否为空字符串或"None"，如果是则设为None
                if image_data_path == "" or image_data_path == "None":
                    image_data_path = None
                    
                if image_data_path is not None and image_data_path.strip() != "":
                    # 如果提供了有效的image_data_path，则需要处理图像
                    print(f"对话 {i}: 检查图像数据，image_data_path={image_data_path}")
                    
                    # 检查图片字段是否存在
                    if "image" not in examples:
                        print(f"警告：对话 {i} 没有image字段，跳过")
                        continue
                        
                    # 检查索引是否有效
                    if i >= len(examples["image"]):
                        print(f"警告：对话 {i} 的image索引超出范围，跳过")
                        continue
                        
                    # 检查图片是否为None
                    if examples["image"][i] is None:
                        print(f"警告：对话 {i} 的图片为None，跳过")
                        continue
                    
                    # 构建图片路径并尝试加载
                    image_path = os.path.join(image_data_path, examples["image"][i])
                    print(f"对话 {i}: 尝试加载图片 {image_path}")
                    
                    image = load_image(image_path)
                    if image is None:
                        print(f"警告：对话 {i} 的图片 {image_path} 加载失败，跳过")
                        continue
                    
                    print(f"对话 {i}: 图片加载成功，大小为 {image.size}")
                    image_size = image.size
                    
                    # 处理图片
                    try:
                        image_tensor = process_images([image], image_processor, {"image_aspect_ratio": "pad"})
                        print(f"对话 {i}: 图片处理成功")
                    except Exception as e:
                        print(f"警告：对话 {i} 的图片处理失败: {str(e)}，跳过")
                        continue
                else:
                    # 如果没有提供image_data_path，则只处理文本部分
                    print(f"对话 {i}: 无image_data_path，仅处理文本")
                    image_tensor = None
            except Exception as e:
                print(f"处理对话 {i} 的图像时出错: {str(e)}")
                image_tensor = None
                image_size = None
                continue

            input_ids = tokenizer_image_token(
                    conversation, 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    return_tensors='pt'
            )


            if -200 in input_ids:
                loss_mask = torch.ones(input_ids.shape[0] + 575, dtype=input_ids.dtype)
                cur_len = 1 + 575
            else:
                loss_mask = torch.ones_like(input_ids)
                cur_len = 1

            sep = conv.sep + conv.roles[1] + ": "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(conv.sep2)

            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                # cur_len += 2
                if i != 0 and not tokenizer.legacy:
                    cur_len -= 1
                # tokenizer.decode(input_ids[loss_mask[-input_ids.shape[0]:]==1])
            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["image"].append(image_tensor)
            new_examples["image_size"].append(image_size)

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1



from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto'}
big_model_model_name = get_model_name_from_path(bigname)
bigtokenizer, bigmodel, image_processor, _ = load_pretrained_model(bigname, None, big_model_model_name, **kwargs)
ds = build_dataset_rank(bigtokenizer)
print(ds)
bigmodel.eval()










@torch.no_grad()
def ge(data):
    try:
        print(f"处理数据: {data.keys()}")
        
        # 检查必要的字段是否存在
        required_fields = ["input_ids", "loss_mask"]
        for field in required_fields:
            if field not in data:
                print(f"错误: 数据中缺少必要字段 {field}")
                return None
        
        # 检查input_ids的形状
        print(f"input_ids形状: {data['input_ids'].shape}")
        if len(data['input_ids'].shape) == 0 or data['input_ids'].numel() == 0:
            print("错误: input_ids为空")
            return None
            
        input_ids = data["input_ids"].cuda()
        image = None
        image_size = None
        
        # 处理图像数据
        if "image" in data and data["image"] is not None:
            print("处理图像数据")
            try:
                image = data["image"].to(dtype=torch.float16).cuda()
                if "image_size" in data and data["image_size"] is not None:
                    image_size = data["image_size"].cuda()
                    print(f"图像大小: {image_size}")
                else:
                    print("警告: 有图像但缺少image_size")
            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
                # 如果图像处理失败，继续处理文本部分
                image = None
                image_size = None
        
        # 获取输入嵌入
        try:
            print("获取输入嵌入")
            inputs_embeds, _ = bigmodel.get_inputs_embeds(input_ids, image, image_size)
        except Exception as e:
            print(f"获取输入嵌入时出错: {str(e)}")
            return None
        
        # 运行模型
        try:
            print("运行模型")
            outs_big = bigmodel(inputs_embeds=inputs_embeds, output_hidden_states=True)
        except Exception as e:
            print(f"运行模型时出错: {str(e)}")
            return None
        
        # 处理模型输出
        try:
            print("处理模型输出")
            hidden_state_big = outs_big.hidden_states[-1]
            max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
            probs = torch.softmax(outs_big.logits, dim=-1)
            maxp = probs[0].max(dim=1).values
            
            # 检查loss_mask的形状
            print(f"loss_mask形状: {data['loss_mask'].shape}")
            
            # 创建输出字典
            td = {
                "input_ids": input_ids.cpu()[0],
                "inputs_embeds": inputs_embeds.cpu()[0],
                "hidden_state": hidden_state_big.cpu()[0],
                "loss_mask": data["loss_mask"].cpu()[0]
            }
            print("数据处理成功")
            return td
        except Exception as e:
            print(f"处理模型输出时出错: {str(e)}")
            return None
    except Exception as e:
        print(f"ge函数中出现未处理的错误: {str(e)}")
        return None

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

for id,data in tqdm(enumerate(ds), total=len(ds), unit="samples"):
    print(f"\n处理样本 {id}")
    try:
        outdata = ge(data)
        if outdata is None:
            print(f"警告：样本 {id} 处理失败，跳过")
            continue
        writedata(outdir,outdata)
        print(f"样本 {id} 处理成功并保存")
    except Exception as e:
        print(f"处理样本 {id} 时出错: {str(e)}，跳过")
        continue