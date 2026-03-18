"""
方案1: UNITER-style 单流BERT-base替换LXMERT
核心思想: 将text tokens和projected visual tokens拼接为单一序列, 通过self-attention实现隐式跨模态融合
优势: 原生ROI兼容, 条件MLM, 最接近drop-in替换
"""
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from .utils_relation import layer_init
import json
import h5py


class GlobalPromptUNITER(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', prompt_length=3):
        super().__init__()
        # 单流模型: Pass1用ForMaskedLM获取MLM logits, Pass2用BertModel获取hidden states
        self.model_mlm = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.model_enc = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        self.prompt_length = prompt_length
        self.embed_dim = self.model_mlm.config.hidden_size  # 768

        # ===== 视觉特征投影 (UNITER核心: img_linear + pos_linear) =====
        self.vis_proj = nn.Linear(2048, self.embed_dim)
        self.pos_proj = nn.Linear(4, self.embed_dim)
        self.vis_ln = nn.LayerNorm(self.embed_dim)
        layer_init(self.vis_proj, xavier=True)
        layer_init(self.pos_proj, xavier=True)

        # ===== 全局可训练提示 =====
        self.global_prompts = nn.Parameter(torch.randn(prompt_length, self.embed_dim))

        # ===== 加载谓词词表 =====
        with open('/home/yj/zgw/het2/Datasets/VG/idx_to_predicate.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entity_word_list = list(data['idx_to_label'].values())
        self.predicate_word_list = [str(p) for p in data['idx_to_predicate'].values()]

        self.predicate_idList = self.tokenizer.convert_tokens_to_ids(self.predicate_word_list)
        self.predicate_idx = {word: idx for idx, word in enumerate(self.predicate_idList)}

        # ===== Graph prompt =====
        struct_proto = '/home/yj/zgw/het2/proto/struct_p.h5'
        f2 = h5py.File(struct_proto, 'r')
        self.struct_proto = torch.from_numpy(f2['struct_proto'][:])
        self.graph_prompt = nn.Parameter(self.struct_proto)

        self.dev = nn.Linear(2048, self.embed_dim)
        layer_init(self.dev, xavier=True)

        # ===== 冻结预训练backbone =====
        for param in self.model_mlm.parameters():
            param.requires_grad = False

    def _project_visual(self, visual_feats, visual_pos):
        """将ROI features投影到embedding空间, 类似UNITER的img_linear+pos_linear"""
        vis_embeds = self.vis_proj(visual_feats) + self.pos_proj(visual_pos)
        vis_embeds = self.vis_ln(vis_embeds)
        return vis_embeds  # [B, 2, 768]

    def _build_single_stream_input(self, text_embeds, vis_embeds, attention_mask):
        """拼接text和visual tokens为单一序列 (UNITER核心设计)"""
        batch_size = text_embeds.size(0)
        device = text_embeds.device

        # 拼接: [text_tokens, vis_sub, vis_obj]
        combined_embeds = torch.cat([text_embeds, vis_embeds], dim=1)

        # Attention mask
        vis_mask = torch.ones(batch_size, vis_embeds.size(1), device=device)
        if attention_mask is not None:
            combined_mask = torch.cat([attention_mask, vis_mask], dim=1)
        else:
            text_mask = torch.ones(batch_size, text_embeds.size(1), device=device)
            combined_mask = torch.cat([text_mask, vis_mask], dim=1)

        # Token type IDs: 0=text, 1=visual
        text_type = torch.zeros(batch_size, text_embeds.size(1), dtype=torch.long, device=device)
        vis_type = torch.ones(batch_size, vis_embeds.size(1), dtype=torch.long, device=device)
        combined_type = torch.cat([text_type, vis_type], dim=1)

        return combined_embeds, combined_mask, combined_type

    def forward(self, input_ids, visual_feats, visual_pos, token_type_ids=None, attention_mask=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # ===== 投影视觉特征 =====
        vis_embeds = self._project_visual(visual_feats, visual_pos)  # [B, 2, 768]

        # ===== 获取文本嵌入 =====
        text_embeds = self.model_mlm.bert.embeddings.word_embeddings(input_ids)

        # ===== Pass 1: MLM预测谓词标签 =====
        combined_embeds, combined_mask, combined_type = self._build_single_stream_input(
            text_embeds, vis_embeds, attention_mask
        )

        with torch.no_grad():
            outputs1 = self.model_mlm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                token_type_ids=combined_type,
            )

        # 在[MASK]位置提取预测logits
        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        batch_logits = outputs1.logits[torch.arange(batch_size, device=device), mask_positions]

        # 过滤到谓词词表
        pred_id_tensor = torch.tensor(self.predicate_idList, device=device)
        mask_filter = ~torch.isin(torch.arange(batch_logits.shape[-1], device=device), pred_id_tensor)
        batch_logits[:, mask_filter] = -float('inf')

        predicted_ids = torch.argmax(batch_logits, dim=-1)
        indice = torch.tensor([self.predicate_idx.get(v.item(), 0) for v in predicted_ids])

        # ===== 检索Graph Prompt =====
        graph_prompt = self.dev(self.graph_prompt[indice, :]).unsqueeze(1)  # [B, 1, 768]

        # ===== Pass 2: 带增强提示的跨模态嵌入提取 =====
        prompts = self.global_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        start = text_embeds[:, 0, :].unsqueeze(1)  # [CLS]

        # 构造增强序列: [CLS, p1, p2, p3, sub_word, graph_prompt, sub_word, mask, obj_word, SEP]
        aug_text = torch.cat([
            start,                                    # position 0: [CLS]
            prompts,                                  # position 1-3: p1,p2,p3
            text_embeds[:, 1, :].unsqueeze(1),       # position 4: sub_word
            graph_prompt,                             # position 5: graph_prompt
            text_embeds[:, 1:, :],                   # position 6+: sub_word, [MASK], obj_word, [SEP]
        ], dim=1)

        # 调整attention mask
        prompt_mask = torch.ones(batch_size, self.prompt_length + 2, device=device)
        if attention_mask is not None:
            aug_attn_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            aug_attn_mask = torch.ones(batch_size, aug_text.size(1), device=device)

        # 构造单流输入
        combined_embeds2, combined_mask2, combined_type2 = self._build_single_stream_input(
            aug_text, vis_embeds, aug_attn_mask
        )

        outputs2 = self.model_enc(
            inputs_embeds=combined_embeds2,
            attention_mask=combined_mask2,
            token_type_ids=combined_type2,
            output_hidden_states=True,
        )

        last_hidden = outputs2.last_hidden_state
        n_text = aug_text.size(1)

        # ===== 提取嵌入 (与原始LXMERT相同的位置语义) =====
        sub_embeddings = last_hidden[:, 4, :]      # sub_word位置
        obj_embeddings = last_hidden[:, 7, :]      # 对应原始的obj位置
        rel_embeddings = last_hidden[:, 6, :]      # 对应原始的rel位置

        # 视觉嵌入在序列末尾 (单流特有)
        sub_vis_embedding = last_hidden[:, n_text, :]      # vis_sub
        obj_vis_embedding = last_hidden[:, n_text + 1, :]  # vis_obj

        cls_embedding = outputs2.pooler_output

        return sub_embeddings, obj_embeddings, rel_embeddings, indice, sub_vis_embedding, obj_vis_embedding, cls_embedding
