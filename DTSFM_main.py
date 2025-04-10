import argparse
import itertools
import os
import sys
import time
import pickle
# os.environ["http_proxy"] = "http://172.16.48.64:10811"
# os.environ["https_proxy"] = "http://172.16.48.64:10811"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from collections import defaultdict, OrderedDict
from transformers import AutoTokenizer, AutoModel

# sys.path.append("..")

import utils
from utils import build_sub_graph
from knowledge_graph import _read_triplets_as_list
import scipy.sparse as sp
import requests
import json
import time
from dgl.nn.pytorch import GATConv



from concurrent.futures import ThreadPoolExecutor

class ExternalKnowledgeCache:
    def __init__(self, cache_dir='cache05', time_interval=100):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)  # 创建缓存目录
        self.time_interval = time_interval  # 每1000个时间一个缓存文件
        self.cache = {}  # 当前缓存数据
        self.cache_time_range = None  # 当前缓存时间区间

    def get_cache_filename(self, start_time):
        start_time = (start_time // self.time_interval) * self.time_interval
        return os.path.join(self.cache_dir, f"cache_{start_time}-{start_time + self.time_interval}.json")

    def load_cache(self, start_time):
        cache_file = self.get_cache_filename(start_time)
        # 只在时间区间更换时读取文件
        if self.cache_time_range != (start_time // self.time_interval) * self.time_interval:
            try:
                torch.cuda.empty_cache()
                self.cache.clear()
                # print(torch.cuda.memory_summary())
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        self.cache = json.load(f)
                    self.cache_time_range = (start_time // self.time_interval) * self.time_interval
                else:
                    self.cache = {}
                    self.cache_time_range = None
            except json.JSONDecodeError as e:
                print(f"缓存文件格式错误，清空并重新加载: {cache_file}")
                self.cache = {}
                self.cache_time_range = None

    def save_cache(self, start_time):
        cache_file = self.get_cache_filename(start_time)

        # 将字典复制一份，防止在遍历时修改字典
        cache_to_save = self.cache.copy()

        # 将键转换为字符串，确保可以保存为JSON
        cache_to_save = {str(key): value for key, value in cache_to_save.items()}

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_to_save, f, ensure_ascii=False, indent=4)
        except json.JSONDecodeError as e:
            print(f"缓存文件格式错误，重新创建: {cache_file}")
            self.cache = {}  # 清空当前缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def get(self, query, time):
        self.load_cache(time)

        return self.cache.get(str(query))

    def add(self, query, knowledge, time):
        self.load_cache(time)

        # 添加新的缓存数据
        self.cache[str(query)] = knowledge
        self.save_cache(time)  # 保存到文件中
# Initialize the cache outside of the function
external_knowledge_cache = ExternalKnowledgeCache()

def fetch_external_knowledge_batch(queries, batch_size=10, is_relation_query=True):
    """
    批量请求外部知识，支持自定义批次大小。使用线程池并行请求来加速。
    """
    url = 'http://172.16.51.231/v1/chat-messages'
    api_key = 'app-msQ7vsTXySMA0sYIm3IDHYvc'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    def fetch(query, time):
        # 从缓存中获取数据
        cached_knowledge = external_knowledge_cache.get(query, time)
        if cached_knowledge:
            return cached_knowledge  # 如果缓存中有结果，直接返回

        s_text, o_text, t = query
        q_text = f"Entity {s_text}; Relation ?; Entity {o_text}; Time {t}" if is_relation_query else f"Entity {s_text}; Relation {o_text}; Entity ?; Time {t}"
        data = {
            "inputs": {},
            "query": {"query": q_text},
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "DTSFM18",
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            answer_data = response.json()
            knowledge = answer_data.get('answer', '')
            # 将获取的知识加入缓存
            external_knowledge_cache.add(query, knowledge, t)
            return knowledge
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None

    # 使用线程池并行化批量请求
    with ThreadPoolExecutor(max_workers=5) as executor:
        responses = list(executor.map(lambda query: fetch(query, query[2]), queries))

    return responses


def fetch_external_knowledge(q, is_relation_query=True):
    # API的URL和Bearer Token
    url = 'http://172.16.51.231/v1/chat-messages'
    api_key = 'app-msQ7vsTXySMA0sYIm3IDHYvc'
    t1,t2,t3,t=q
    # 请求头
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    # 构造查询文本
    if is_relation_query:
        s_text = t1
        o_text = t3
        query = (s_text, o_text, t)
        q_text = f"Entity {s_text}; Relation ?; Entity {o_text}; Time {t}"
    else:
        s_text = t1
        r_text = t2
        query = (s_text, r_text, t)
        q_text = f"Entity {s_text}; Relation {r_text}; Entity ?; Time {t}"

    # 从缓存中获取数据
    cached_knowledge = external_knowledge_cache.get(query, t)
    if cached_knowledge:
        return cached_knowledge



    # 请求数据
    data = {
        "inputs": {},
        "query": {"query": q_text},
        "response_mode": "blocking",
        "conversation_id": "",
        "user": "DTSFM16",
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        answer_data = response.json()
        knowledge = answer_data.get('answer', '')

        external_knowledge_cache.add(query, knowledge, t)

        return knowledge
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None


class LMEncoder:
    def __init__(self, model_name='roberta-base', device='cuda:1', max_cache_size=10000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.model.eval()  # 设置为评估模式
        # self.cache = {}  # 缓存语义表示
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()

    def encode(self, text):
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
            if len(self.cache) >= self.max_cache_size:
                while len(self.cache) >= self.max_cache_size:
                    self.cache.popitem(last=False)
            self.cache[text] = cls_embedding.squeeze(0)  # 存入缓存
            return self.cache[text]

    def encode_batch(self, texts):
        """
        批量处理多个文本
        texts: 文本列表
        返回: 文本的嵌入（[batch_size, hidden_dim]）
        """
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("输入文本列表无效或为空")

        uncached_texts = []
        cached_embeddings = []
        for text in texts:
            if text in self.cache:
                cached_embeddings.append(self.cache[text])
            else:
                uncached_texts.append(text)

        if uncached_texts:
            with torch.no_grad():
                inputs = self.tokenizer(uncached_texts, return_tensors='pt', truncation=True, padding=True)
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

            # 将未缓存的文本嵌入加入缓存
            for i, text in enumerate(uncached_texts):
                self.cache[text] = cls_embeddings[i].squeeze(0)

            cached_embeddings.extend(cls_embeddings)

        return torch.stack(cached_embeddings, dim=0)  # 返回所有文本的嵌入




def time_semantic_score(fact, query, query_time, lm_encoder, id_to_entity, id_to_relation, alpha=1.0):

    # 计算时间差
    dt = query_time - fact[3]  # 计算所有历史事实与查询的时间差
    w_t = torch.exp(-alpha * torch.tensor(dt, device=lm_encoder.device))  # 批量计算时间衰减权重

    u_text = id_to_entity.get(fact[0], f"Entity{fact[0]}")
    r_f_text = id_to_relation.get(fact[1], f"Relation{fact[1]}")
    v_text = id_to_entity.get(fact[2], f"Entity{fact[2]}")
    fact_text = f"Entity {u_text}, Relation {r_f_text}, Entity {v_text}, Time {fact[3]}"

    fact_vec = lm_encoder.encode(fact_text)

    # 查询文本表示
    query_text = f"Entity {id_to_entity.get(query[0], f'Entity{query[0]}')}, Relation {id_to_relation.get(query[1], f'Relation{query[1]}')}, Time {query_time}"
    query_vec = lm_encoder.encode(query_text)  # 计算查询的表示

    # 计算语义匹配度并加权
    semantic_score = F.cosine_similarity(fact_vec.unsqueeze(0), query_vec.unsqueeze(0), dim=1)
    g_fq = w_t * semantic_score  # 加权后的语义评分

    return g_fq


def score_and_prune_subgraph(query, history_facts, m_hop_neighbors, top_n, lm_encoder, id_to_entity, id_to_relation, alpha=1.0, time_threshold=35):
    s = query[0]
    node_set = set([s]) | set(m_hop_neighbors)

    filtered_facts = [f for f in history_facts if (f[0] in node_set or f[2] in node_set) and (f[3] < query[3]) and (
            query[3] - f[3] <= time_threshold)]

    if not filtered_facts:
        return []

    fact_texts = [
        f"Entity {id_to_entity.get(fact[0], f'Entity{fact[0]}')}, Relation {id_to_relation.get(fact[1], f'Relation{fact[1]}')}, Entity {id_to_entity.get(fact[2], f'Entity{fact[2]}')}, Time {fact[3]}"
        for fact in filtered_facts
    ]
    fact_embeddings = lm_encoder.encode_batch(fact_texts)

    scored_facts = []
    for fact, fact_vec in zip(filtered_facts, fact_embeddings):
        g_fq = time_semantic_score(fact, query, query[3], lm_encoder, id_to_entity, id_to_relation, alpha=alpha)
        scored_facts.append((fact, g_fq))

    scored_facts.sort(key=lambda x: x[1], reverse=True)
    pruned_facts = [x[0] for x in scored_facts[:top_n]]

    return pruned_facts



class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1):
        super(RGATLayer, self).__init__()
        self.gat_conv = GATConv(in_dim, out_dim, num_heads,allow_zero_in_degree=True)

    def forward(self, g, node_feats):
        # g = dgl.add_self_loop(g)
        h = self.gat_conv(g, node_feats)
        return h


class DTSFMEncoder(nn.Module):


    def __init__(self, in_dim, hidden_dim, lm_encoder, use_cuda=False):
        super(DTSFMEncoder, self).__init__()
        self.use_cuda = use_cuda

        self.rgat_layer = RGATLayer(in_dim, hidden_dim)

        self.lm_encoder = lm_encoder

        self.proj_z_q = nn.Linear(768, hidden_dim)

        self.w_t = nn.Parameter(torch.Tensor([0.5]))
        self.w_s = nn.Parameter(torch.Tensor([0.5]))

        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, g, node_feats, context_text):

        h_temp = self.rgat_layer(g, node_feats)  # [num_nodes, hidden_dim]
        h_temp = h_temp.squeeze(1)

        z_q = self.lm_encoder.encode(context_text)  # [hidden_dim_LM=768]

        z_q = self.proj_z_q(z_q)  # [hidden_dim=200]

        beta_t = torch.exp(self.w_t) / (torch.exp(self.w_t) + torch.exp(self.w_s))
        beta_s = torch.exp(self.w_s) / (torch.exp(self.w_t) + torch.exp(self.w_s))
        beta_t = beta_t.to(h_temp.device)
        beta_s = beta_s.to(h_temp.device)


        h_fusion = beta_t * h_temp + beta_s * z_q

        h_fusion = h_fusion + (h_temp + z_q) * 0.1

        h_fusion = self.proj(h_fusion)


        h_fusion = torch.mean(h_fusion, dim=0, keepdim=True)

        return h_fusion




class ExternalKnowledgeAugment(nn.Module):


    def __init__(self, hidden_dim, lm_encoder):
        super(ExternalKnowledgeAugment, self).__init__()
        # 知识编码的投影层
        self.proj_ext = nn.Linear(768, hidden_dim)
        self.gamma_1 = nn.Parameter(torch.Tensor([0.33]))
        self.gamma_2 = nn.Parameter(torch.Tensor([0.33]))
        self.gamma_3 = nn.Parameter(torch.Tensor([0.34]))
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        self.lm_encoder = lm_encoder

        dataset_folder = os.path.join('../data', args.dataset)  # 替换为实际的数据集文件夹路径
        entity2id_path = os.path.join(dataset_folder, 'entity2id.txt')
        relation2id_path = os.path.join(dataset_folder, 'relation2id.txt')

        self.id_to_entity = load_id_mapping(entity2id_path)
        self.id_to_relation = load_id_mapping(relation2id_path)

    def forward(self, h_q_temp, z_q, external_facts, query):
        s, r_query, o, t = query
        is_relation_query = (r_query is None)  # 判断是否为关系预测

        # 批量构建查询列表
        # queries = []
        # for fact in external_facts:
        #     s, r_query, o, t = fact
        #     if is_relation_query:
        #         queries.append((self.id_to_entity.get(s, s), self.id_to_entity.get(o, o),t))
        #     else:
        #         queries.append((self.id_to_entity.get(s, s), self.id_to_relation.get(r_query, r_query),t))

        external_knowledge = fetch_external_knowledge(query, is_relation_query=is_relation_query)

        knowledge=external_knowledge
        if knowledge:
            z_ext = self.lm_encoder.encode(knowledge)
            z_ext = self.proj_ext(z_ext)  # 投影操作
            # 确保 z_ext 是 [1, hidden_dim] 形状
            if z_ext.dim() == 1:  # 如果 z_ext 是 [hidden_dim]
                z_ext = z_ext.unsqueeze(0)  # 增加一个维度，变为 [1, hidden_dim]
        else:
            z_ext = torch.zeros_like(h_q_temp)  # 保持维度一致
            print('no use knowledge')



        h_q_aug = self.gamma_1 * h_q_temp + self.gamma_2 * z_q + self.gamma_3 * z_ext
        return self.linear(h_q_aug)




class DTSFMModel(nn.Module):
    """
    DTSFM总模型
    - 结合3.4中多个损失项
    - 包含: 实体预测损失, 关系预测损失, 语言模型重构损失, 一致性正则化, 知识增强损失等
    """

    def __init__(self, in_dim, hidden_dim, lm_encoder, num_entities, num_relations, use_cuda=False):
        super(DTSFMModel, self).__init__()
        self.use_cuda = use_cuda
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(num_entities, in_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)  # Xavier初始化

        self.encoder = DTSFMEncoder(in_dim, hidden_dim, lm_encoder, use_cuda=use_cuda)

        self.ext_augment = ExternalKnowledgeAugment(hidden_dim,lm_encoder)

        # self.entity_classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, num_entities),  # 线性层
        #     nn.Softmax(dim=-1)  # Softmax 激活
        # )
        self.entity_classifier = nn.Linear(hidden_dim, num_entities)
        self.rel_classifier = nn.Linear(hidden_dim, num_relations)
        # self.rel_classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, num_relations),  # 线性层
        #     nn.Softmax(dim=-1)  # Softmax 激活
        # )

        self.lm_reconstruct_linear = nn.Linear(hidden_dim, hidden_dim)

        self.lambda_1 = 0.1  # 一致性正则化权重
        self.lambda_2 = 0.1  # 知识增强损失权重

    def forward(self, g, node_ids, context_text, query=None, external_facts=None):


        node_feats = self.entity_embedding(node_ids)  # [num_nodes, in_dim]

        h_fusion = self.encoder(g, node_feats, context_text)  # [1, hidden_dim]


        if external_facts is not None:
            h_fusion = self.ext_augment(
                h_q_temp=h_fusion,
                z_q=h_fusion,  # 使用编码器输出的语义向量
                external_facts=external_facts,
                query=query
            )


        return h_fusion

    def get_loss(self,
                 h_fusion,
                 target_entity=None,
                 target_relation=None,
                 prev_h_fusion=None,
                 external_knowledge_vec=None):
        loss_ent = 0.0
        loss_rel = 0.0
        loss_lm = 0.0
        loss_consistency = 0.0
        loss_ext = 0.0

        if target_entity is not None:
            logits_e = self.entity_classifier(h_fusion)
            logits_e = logits_e.squeeze(0)

            # Ensure the target label is a 1D tensor of shape [batch_size]
            label_e = torch.tensor(target_entity, dtype=torch.long, device=logits_e.device)  # Should be [batch_size]

            if label_e.dim() > 1:
                label_e = label_e.squeeze()  # 去除多余的维度

            loss_ent = F.cross_entropy(logits_e, label_e)  # [batch_size, num_entities]

        if target_relation is not None:
            logits_r = self.rel_classifier(h_fusion)
            logits_r = logits_r.squeeze(0)

            label_r = torch.tensor(target_relation, dtype=torch.long, device=logits_r.device)  # Should be [batch_size]

            if label_r.dim() > 1:
                label_r = label_r.squeeze()

            loss_rel = F.cross_entropy(logits_r, label_r)

        reconstruct = self.lm_reconstruct_linear(h_fusion)
        loss_lm = torch.mean(reconstruct ** 2)

        if prev_h_fusion is not None:
            loss_consistency = F.mse_loss(h_fusion, prev_h_fusion)

        if external_knowledge_vec is not None:
            cos_sim = F.cosine_similarity(h_fusion, external_knowledge_vec, dim=1)
            # 防止 cos_sim 为零或负值
            cos_sim = torch.clamp(cos_sim, min=1e-8)  # 确保 cos_sim 不小于 1e-8
            loss_ext = -torch.mean(torch.log(cos_sim))

        L_total = (loss_ent + loss_rel + loss_lm
                   + self.lambda_1 * loss_consistency
                   + self.lambda_2 * loss_ext)

        return L_total, (loss_ent, loss_rel, loss_lm, loss_consistency, loss_ext)



def train_and_evaluate_dtsfm(model, train_data, valid_data, test_data, args, id_to_entity, id_to_relation,
                             dataset_folder):
    device = next(model.parameters()).device  # 获取模型的设备
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_mrr = 0.0
    for epoch in range(args.n_epochs):
        model.train()
        epoch_losses = []
        for step, batch in enumerate(tqdm(train_data, desc=f"Epoch {epoch + 1}/{args.n_epochs}")):
            query = batch['query']  # (s, r, ?, t)
            history_facts = batch['history']
            m_hop_neighbors = batch['neighbors']
            top_n = args.top_n

            pruned_facts = score_and_prune_subgraph(
                query, history_facts, m_hop_neighbors, top_n,
                model.encoder.lm_encoder, id_to_entity, id_to_relation, alpha=1.0
            )

            if not pruned_facts:
                continue

            pruned_np = np.array(pruned_facts)
            if len(pruned_np.shape) < 2 or pruned_np.shape[0] == 0:
                continue

            pruned_graph = build_sub_graph(args.num_nodes, args.num_rels, pruned_np,
                                           model.use_cuda, args.gpu)

            pruned_graph = pruned_graph.to(device)

            node_ids = torch.arange(pruned_graph.num_nodes(), device=device)

            context_texts = []
            external_facts = []
            for pf in pruned_facts[:20]:
                u, r, v, t = pf
                u_text = id_to_entity.get(u, f"Entity{u}")
                r_text = id_to_relation.get(r, f"Relation{r}")
                v_text = id_to_entity.get(v, f"Entity{v}")
                context_text = f"Entity {u_text}, Relation {r_text}, Entity {v_text}, Time {t}"
                context_texts.append(context_text)
                external_facts.append(pf)  # 保持原始事实用于外部知识增强

            context_text = " ".join(context_texts)

            q0,q1,q2='','',''
            if query[1]==None:
                q0=id_to_entity.get(query[0],query[0])
                q1=None
                q2=id_to_entity.get(query[2],query[2])
            else:
                q0 = id_to_entity.get(query[0], query[0])
                q1 = id_to_relation.get(query[1], query[1])
                q2 = None
            q=(q0,q1,q2,query[3])

            h_fusion = model(pruned_graph, node_ids, context_text, query=q,external_facts=q)

            external_knowledge_vec = model.ext_augment(h_fusion, h_fusion, q, q)

            prev_h_fusion = None  # (示例) 没有跨时间的一致性
            loss, loss_tuple = model.get_loss(
                h_fusion,
                target_entity=batch.get('target_entity'),
                target_relation=batch.get('target_relation'),
                prev_h_fusion=prev_h_fusion,
                external_knowledge_vec=external_knowledge_vec
            )
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

            del h_fusion, pruned_graph, external_knowledge_vec

        print(f"Epoch {epoch + 1}, training loss: {np.mean(epoch_losses):.4f}")

        if (epoch + 1) % args.evaluate_every == 0:
            print(f"Evaluating model at epoch {epoch + 1}")
            entity_metrics, relation_metrics = evaluate(model, valid_data, args, mode="valid",
                                                        id_to_entity=id_to_entity, id_to_relation=id_to_relation)
            print(f"""
                    [Entity Prediction - Validation]
                    MRR: {entity_metrics[0]:.4f}  H@1: {entity_metrics[1]:.4f}
                    H@3: {entity_metrics[2]:.4f}  H@10: {entity_metrics[3]:.4f}

                    [Relation Prediction - Validation] 
                    MRR: {relation_metrics[0]:.4f}  H@1: {relation_metrics[1]:.4f}
                    H@3: {relation_metrics[2]:.4f}  H@10: {relation_metrics[3]:.4f}
                    """)
            print(f"""
                    [Entity and Relation Prediction - Validation]
                    MRR: {((entity_metrics[0]+relation_metrics[0])/2):.4f}  H@1: {((entity_metrics[1]+relation_metrics[1])/2):.4f}
                    H@3: {((entity_metrics[2]+relation_metrics[2])/2):.4f}  H@10: {((entity_metrics[3]+relation_metrics[3])/2):.4f}
                    """)
            # 保存指标数据
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"./{timestamp}_valid_metrics.txt"
            with open(filename, 'w') as f:
                f.write(f"[Entity Prediction - Validation]\n")
                f.write(f"MRR: {entity_metrics[0]:.4f}  H@1: {entity_metrics[1]:.4f}\n")
                f.write(f"H@3: {entity_metrics[2]:.4f}  H@10: {entity_metrics[3]:.4f}\n")
                f.write(f"\n[Relation Prediction - Validation]\n")
                f.write(f"MRR: {relation_metrics[0]:.4f}  H@1: {relation_metrics[1]:.4f}\n")
                f.write(f"H@3: {relation_metrics[2]:.4f}  H@10: {relation_metrics[3]:.4f}\n")
                f.write(f"\n[Entity and Relation Prediction - Validation]\n")
                f.write(f"MRR: {((entity_metrics[0]+relation_metrics[0])/2):.4f}  H@1: {((entity_metrics[1]+relation_metrics[1])/2):.4f}\n")
                f.write(f"H@3: {((entity_metrics[2]+relation_metrics[2])/2):.4f}  H@10: {((entity_metrics[3]+relation_metrics[3])/2):.4f}\n")

            if entity_metrics[0] > best_mrr:
                best_mrr = entity_metrics[0]
                print(f"Saving the best model with MRR {best_mrr:.4f}")
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                }, os.path.join(args.save_dir, f'{timestamp}_best_dtsfm.pth'))

    best_model_ckpt = torch.load(os.path.join(args.save_dir, "best_dtsfm.pth"), map_location='cpu')
    model.load_state_dict(best_model_ckpt['state_dict'])
    entity_metrics, relation_metrics = evaluate(model, test_data, args, mode="test", id_to_entity=id_to_entity, id_to_relation=id_to_relation)
    print(f"""
    [Entity Prediction]
    MRR: {entity_metrics[0]:.4f}  H@1: {entity_metrics[1]:.4f}
    H@3: {entity_metrics[2]:.4f}  H@10: {entity_metrics[3]:.4f}

    [Relation Prediction] 
    MRR: {relation_metrics[0]:.4f}  H@1: {relation_metrics[1]:.4f}
    H@3: {relation_metrics[2]:.4f}  H@10: {relation_metrics[3]:.4f}
    [Entity and Relation Prediction - Validation]
    MRR: {((entity_metrics[0]+relation_metrics[0])/2):.4f}  H@1: {((entity_metrics[1]+relation_metrics[1])/2):.4f}
    H@3: {((entity_metrics[2]+relation_metrics[2])/2):.4f}  H@10: {((entity_metrics[3]+relation_metrics[3])/2):.4f}
    """)



def evaluate(model, data, args, mode="valid", id_to_entity=None, id_to_relation=None):
    model.eval()
    entity_ranks = []
    relation_ranks = []
    all_scores = []
    all_test_triples = []

    external_knowledge_cache = ExternalKnowledgeCache()

    # 获取模型的设备
    device = next(model.parameters()).device

    # 批量处理
    with torch.no_grad():
        for batch in tqdm(data, desc=f"Evaluating {mode}"):
            query = batch['query']  # (s, r, ?, t)
            history_facts = batch['history']
            m_hop_neighbors = batch['neighbors']

            pruned_facts = score_and_prune_subgraph(
                query, history_facts, m_hop_neighbors, args.top_n,
                model.encoder.lm_encoder, id_to_entity, id_to_relation, alpha=1.0
            )
            pruned_np = np.array(pruned_facts)
            if len(pruned_np.shape) < 2 or pruned_np.shape[0] == 0:
                continue

            pruned_graph = build_sub_graph(args.num_nodes, args.num_rels, pruned_np,
                                           model.use_cuda, args.gpu)

            pruned_graph = pruned_graph.to(device)

            node_ids = torch.arange(pruned_graph.num_nodes(), device=device)

            context_texts = []
            external_facts = []
            for pf in pruned_facts:
                u, r, v, t = pf
                u_text = id_to_entity.get(u, f"Entity{u}")
                r_text = id_to_relation.get(r, f"Relation{r}")
                v_text = id_to_entity.get(v, f"Entity{v}")
                context_text = f"Entity {u_text}, Relation {r_text}, Entity {v_text}, Time {t}"
                context_texts.append(context_text)
                external_facts.append(pf)  # 保持原始事实用于外部知识增强
            context_text = " ".join(context_texts)

            q0, q1, q2 = '', '', ''
            if query[1] == None:
                q0 = id_to_entity.get(query[0], query[0])
                q1 = None
                q2 = id_to_entity.get(query[2], query[2])
            else:
                q0 = id_to_entity.get(query[0], query[0])
                q1 = id_to_relation.get(query[1], query[1])
                q2 = None
            q = (q0, q1, q2, query[3])

            h_fusion = model(pruned_graph, node_ids, context_text, query=q,external_facts=q)

            external_knowledge_vec = model.ext_augment(h_fusion, h_fusion, q, q)

            # prev_h_fusion = None
            # loss, loss_tuple = model.get_loss(
            #     h_fusion,
            #     target_entity=batch.get('target_entity'),
            #     target_relation=batch.get('target_relation'),
            #     prev_h_fusion=prev_h_fusion,
            #     external_knowledge_vec=external_knowledge_vec
            # )

            if query[1] is None:
                logits = model.rel_classifier(h_fusion)
                targets = torch.tensor(batch['target_relation'], dtype=torch.long).to(device)
            else:
                logits = model.entity_classifier(h_fusion)
                targets = torch.tensor(batch['target_entity'], dtype=torch.long).to(device)

            sorted_indices = torch.argsort(logits, dim=1, descending=True)
            ranks = (sorted_indices == targets.view(-1, 1)).nonzero()[:, 1] + 1

            if query[1] is None:
                relation_ranks.extend(ranks.cpu().tolist())
            else:
                entity_ranks.extend(ranks.cpu().tolist())

            del h_fusion, pruned_graph, external_knowledge_vec

        def calculate_metrics(ranks):
            # if not ranks:
            #     return (0.0, 0.0, 0.0, 0.0)
            mrr = np.mean(1.0 / np.array(ranks))
            hits1 = np.mean(np.array(ranks) <= 1)
            hits3 = np.mean(np.array(ranks) <= 3)
            hits10 = np.mean(np.array(ranks) <= 10)
            return mrr, hits1, hits3, hits10

        entity_metrics = calculate_metrics(entity_ranks) if entity_ranks else (0, 0, 0, 0)
        relation_metrics = calculate_metrics(relation_ranks) if relation_ranks else (0, 0, 0, 0)

        return entity_metrics, relation_metrics



def load_id_mapping(file_path):
    id_to_name = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # 跳过格式不正确的行
            name, id_str = parts
            id_to_name[int(id_str)] = name
    return id_to_name


def run_experiment(args):
    print("=== Loading data ===")
    data = utils.load_data(args.dataset)
    # train_list, train_times = utils.split_by_time(data.train)
    # valid_list, valid_times = utils.split_by_time(data.valid)
    # test_list, test_times = utils.split_by_time(data.test)

    # 加载实体和关系的映射
    dataset_folder = os.path.join('../data', args.dataset)  # 替换为实际的数据集文件夹路径
    entity2id_path = os.path.join(dataset_folder, 'entity2id.txt')
    relation2id_path = os.path.join(dataset_folder, 'relation2id.txt')

    id_to_entity = load_id_mapping(entity2id_path)
    id_to_relation = load_id_mapping(relation2id_path)


    print(f"Loaded {len(id_to_entity)} entities and {len(id_to_relation)} relations.")


    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    lm_encoder = LMEncoder(model_name='../roberta_base', device=device)

    # 构造模型
    model = DTSFMModel(in_dim=args.in_dim,
                       hidden_dim=args.hidden_dim,
                       lm_encoder=lm_encoder,
                       num_entities=data.num_nodes,
                       num_relations=data.num_rels,
                       use_cuda=(args.gpu >= 0))

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        model.to(device)


    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from checkpoint: {checkpoint_path}")

        test_list, test_times = utils.split_by_time(data.test)
        test_data = prepare_dtsfm_data(test_list, test_times, data.num_rels, data.num_nodes, mode='test', args=args)
        print("Evaluating model on test data...")
        entity_metrics, relation_metrics = evaluate(model, test_data, args, mode="test", id_to_entity=id_to_entity,
                                                    id_to_relation=id_to_relation)

        print(f"""
        [Entity Prediction - Test]
        MRR: {entity_metrics[0]:.4f}  H@1: {entity_metrics[1]:.4f}
        H@3: {entity_metrics[2]:.4f}  H@10: {entity_metrics[3]:.4f}
    
        [Relation Prediction - Test] 
        MRR: {relation_metrics[0]:.4f}  H@1: {relation_metrics[1]:.4f}
        H@3: {relation_metrics[2]:.4f}  H@10: {relation_metrics[3]:.4f}
        
        [Entity and Relation Prediction - test]
        MRR: {((entity_metrics[0]+relation_metrics[0])/2):.4f}  H@1: {((entity_metrics[1]+relation_metrics[1])/2):.4f}
        H@3: {((entity_metrics[2]+relation_metrics[2])/2):.4f}  H@10: {((entity_metrics[3]+relation_metrics[3])/2):.4f}
        """)
    else:
        train_list, train_times = utils.split_by_time(data.train)
        valid_list, valid_times = utils.split_by_time(data.valid)
        test_list, test_times = utils.split_by_time(data.test)
        train_data = prepare_dtsfm_data(train_list, train_times, data.num_rels, data.num_nodes, mode='train',
                                        args=args)
        valid_data = prepare_dtsfm_data(valid_list, valid_times, data.num_rels, data.num_nodes, mode='valid',
                                        args=args)

        test_data = prepare_dtsfm_data(test_list, test_times, data.num_rels, data.num_nodes, mode='test', args=args)
        train_and_evaluate_dtsfm(model, train_data, valid_data, test_data, args, id_to_entity, id_to_relation, dataset_folder)


def prepare_dtsfm_data(snapshots, times, num_rels, num_nodes, mode='train', args=None):
    data = []
    adjacency = defaultdict(set)
    all_facts = []

    for idx, snapshot in enumerate(snapshots):
        current_time = times[idx]
        print(f"Processing snapshot {idx + 1}/{len(snapshots)} at time {current_time}")
        all_facts_current_time=[]
        for fact in snapshot:
            s, r, o, t = fact
            all_facts_current_time.append((s, r, o, t))
            # all_facts.append((s, r, o, t))
            adjacency[s].add(o)
            adjacency[o].add(s)
        all_facts.append(all_facts_current_time)
        for fact in snapshot:
            history_facts = []
            s, r, o, t = fact
            if idx>10:
                for facts in all_facts[idx-10:]:
                    for f in facts:
                        if f[3] < t:
                            history_facts.append(f)
                # history_facts = [f for f in all_facts[idx-10:] if f[3] < t]
            else:
                for facts in all_facts:
                    for f in facts:
                        if f[3] < t:
                            history_facts.append(f)
                # history_facts = [f for f in all_facts if f[3] < t]
            neighbors = get_m_hop_neighbors(s, adjacency, m=1)

            # Entity prediction sample
            data.append({
                'query': (s, r, None, t),
                'history': history_facts,
                'neighbors': list(neighbors),
                'target_entity': o,
                'target_relation': None
            })

            # Relation prediction sample
            data.append({
                'query': (s, None, o, t),
                'history': history_facts,
                'neighbors': list(neighbors),
                'target_entity': None,
                'target_relation': r
            })

    return data




def get_m_hop_neighbors(entity, adjacency, m):
    visited = set()
    current_level = set([entity])
    for _ in range(m):
        next_level = set()
        for node in current_level:
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    next_level.add(neighbor)
        visited.update(current_level)
        current_level = next_level
    return visited


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='DTSFM')
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--dataset", type=str, default='ICEWS05-15')
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="../models/")
    parser.add_argument("--use-external-knowledge", action='store_true', default=True)
    parser.add_argument("--in_dim", type=int, default=200, help="输入特征维度")
    parser.add_argument("--hidden_dim", type=int, default=200, help="隐藏表示维度")
    parser.add_argument("--num_nodes", type=int, default=10488)
    parser.add_argument("--num_rels", type=int, default=251)
    parser.add_argument("--top_n", type=int, default=2000, help="剪枝时取最高分历史事实个数")
    parser.add_argument("--eval_bz", type=int, default=128, help="评估时的batch大小")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to the checkpoint file to load the model for testing")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    run_experiment(args)