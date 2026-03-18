"""
方案1: LXMERT + DoRA (rank=16) on cross-modal encoder 5层
核心思路: 保留LXMERT的VL预训练权重, 在cross-modal encoder的5层x_layers上添加DoRA适配器
DoRA将权重分解为幅度(magnitude)和方向(direction), 比LoRA更精细地控制更新
Pass 1 (LxmertForPreTraining): 冻结, 做MLM预测谓词
Pass 2 (LxmertModel): 基础权重冻结, 仅训练DoRA参数
对比方案2(LoRA)量化DoRA vs LoRA在SGG跨模态任务上的差异
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LxmertModel, LxmertForPreTraining, LxmertTokenizer
from .utils_relation import layer_init
import json
import h5py


class DoRALinear(nn.Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation
    将权重分解为 magnitude * direction, 其中direction通过LoRA更新"""
    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.scaling = alpha / rank

        # 冻结原始权重
        self.register_buffer('frozen_weight', original_linear.weight.data.clone())
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA分解矩阵
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

        # DoRA: 可学习的幅度向量 (初始化为原始权重的列范数)
        with torch.no_grad():
            self.magnitude = nn.Parameter(self.frozen_weight.norm(dim=1))

    def forward(self, x):
        # 组合权重 = 原始权重 + LoRA增量
        delta_w = (self.lora_A @ self.lora_B).T * self.scaling
        combined_w = self.frozen_weight + delta_w
        # 分解为方向 + 幅度
        direction = F.normalize(combined_w, dim=1)
        dora_weight = self.magnitude.unsqueeze(1) * direction
        return F.linear(x, dora_weight, self.bias)


def apply_dora_to_xlayers(model, rank=16, alpha=32):
    """仅对LxmertModel的x_layers (cross-modal encoder)中的Q/K/V添加DoRA"""
    target_keywords = ["visual_attention.att", "lang_self_att.self", "visn_self_att.self"]
    proj_names = ["query", "key", "value"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # 检查是否在x_layers中且是attention的QKV
            if "x_layers" in name and any(kw in name for kw in target_keywords) and any(p in name for p in proj_names):
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, DoRALinear(module, rank=rank, alpha=alpha))
    return model


class GlobalPromptLxmertDoRA(nn.Module):
    def __init__(self, pretrained_model_name='/home/yj/zgw/lxmert-base-uncased', prompt_length=3):
        super().__init__()
        # Pass 1: 冻结的LxmertForPreTraining用于MLM预测
        self.lxmert = LxmertForPreTraining.from_pretrained(pretrained_model_name)
        # Pass 2: LxmertModel + DoRA
        self.lxmert2 = LxmertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = LxmertTokenizer.from_pretrained(pretrained_model_name)

        self.prompt_length = prompt_length
        embed_dim = self.lxmert.config.hidden_size  # 768

        # 对lxmert2的x_layers应用DoRA
        self.lxmert2 = apply_dora_to_xlayers(self.lxmert2, rank=16, alpha=32)

        # 冻结lxmert2中非DoRA的参数
        for name, param in self.lxmert2.named_parameters():
            if 'lora_' not in name and 'magnitude' not in name:
                param.requires_grad = False

        # 全局可训练提示
        self.global_prompts = nn.Parameter(torch.randn(prompt_length, embed_dim))

        # 加载谓词词表
        with open('/home/yj/zgw/het2/Datasets/VG/idx_to_predicate.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.entity_word_list = list(data['idx_to_label'].values())
        self.predicate_word_list = [str(p) for p in data['idx_to_predicate'].values()]
        self.lxmert_predicate_idList = self.tokenizer.convert_tokens_to_ids(self.predicate_word_list)
        self.predicate_idx = {word: idx for idx, word in enumerate(self.lxmert_predicate_idList)}

        # Graph prompt
        struct_proto = '/home/yj/zgw/het2/proto/struct_p.h5'
        f2 = h5py.File(struct_proto, 'r')
        self.struct_proto = torch.from_numpy(f2['struct_proto'][:])
        self.graph_prompt = nn.Parameter(self.struct_proto)
        self.dev = nn.Linear(2048, 768)
        layer_init(self.dev, xavier=True)

        # 冻结Pass 1
        for param in self.lxmert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, visual_feats, visual_pos, token_type_ids=None, attention_mask=None):
        text_embeds = self.lxmert.lxmert.embeddings.word_embeddings(input_ids)
        batch_size = text_embeds.size(0)

        # === Pass 1: MLM预测谓词 (冻结) ===
        outputs = self.lxmert(
            inputs_embeds=text_embeds,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        batch_logits = outputs.prediction_logits[torch.arange(batch_size), mask_positions]
        batch_logits[:, ~torch.isin(torch.arange(batch_logits.shape[-1]), torch.tensor(self.lxmert_predicate_idList))] = -float('inf')
        predicted_ids = torch.argmax(batch_logits, dim=-1)
        indice = torch.tensor([self.predicate_idx.get(value.item(), -1) for value in predicted_ids])

        # 检索graph prompt
        graph_prompt = self.dev(self.graph_prompt[indice, :]).unsqueeze(1)

        # === Pass 2: DoRA增强的LXMERT提取跨模态嵌入 ===
        prompts = self.global_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        start = text_embeds[:, 0, :].unsqueeze(1)
        text_embeds_tmp = torch.cat([start, prompts], dim=1)
        text_embeds_tmp = torch.cat([text_embeds_tmp, text_embeds[:, 1, :].unsqueeze(1)], dim=1)
        text_embeds_tmp = torch.cat([text_embeds_tmp, graph_prompt], dim=1)
        text_embeds = torch.cat([text_embeds_tmp, text_embeds[:, 1:, :]], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones((batch_size, self.prompt_length + 2), device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.lxmert2(
            inputs_embeds=text_embeds,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        language_last = outputs.language_hidden_states[-1]
        sub_embeddings = language_last[:, 4, :]
        obj_embeddings = language_last[:, 7, :]
        rel_embeddings = language_last[:, 6, :]
        sub_vis_embedding = outputs.vision_hidden_states[-1][:, 0, :]
        obj_vis_embedding = outputs.vision_hidden_states[-1][:, 1, :]
        cls_embedding = outputs.pooled_output

        return sub_embeddings, obj_embeddings, rel_embeddings, indice, sub_vis_embedding, obj_vis_embedding, cls_embedding
