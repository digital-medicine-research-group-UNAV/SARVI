import torch
import pandas as pd
from torch import Tensor
import torch.nn as nn
from asyncio import Semaphore
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any
from torch.utils.data import Dataset
from anytree import PostOrderIter

from .config import AppPaths

class DisabledOptionError(Exception):
    pass

class SpanDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Token-level embeddings
        token_embeddings = row["Embeddings"]
        if token_embeddings.dim() == 1:
            token_embeddings = token_embeddings.unsqueeze(0)
        span_repr, _ = torch.max(token_embeddings, dim=0)
        # CLS
        cls_repr = row["CLS Embedding"]
        # Width
        span_width = torch.tensor(token_embeddings.size(0), dtype=torch.long)

        return span_repr, cls_repr, span_width

class SpanClassifier(nn.Module):
    def __init__(self, span_dim, cls_dim, num_classes, max_span_width, width_emb_dim=25, dropout=0.1):
        super().__init__()

        # Width embeddings (TRAINABLE)
        self.width_embeddings = nn.Embedding(max_span_width + 1, width_emb_dim)
        nn.init.xavier_uniform_(self.width_embeddings.weight)

        # Clasificador (TRAINABLE)
        input_dim = span_dim + cls_dim + width_emb_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_classes)
        )

    def forward(self, span_repr, cls_repr, span_width):
        """
        span_repr:  [N, span_dim]
        cls_repr:   [N, cls_dim]
        span_width: [N]
        """

        # To NOT train prior encoder data
        span_repr = span_repr.detach()
        cls_repr = cls_repr.detach()

        width_emb = self.width_embeddings(span_width)  # [N, width_emb_dim]

        x = torch.cat([span_repr, cls_repr, width_emb], dim=-1)
        logits = self.classifier(x)

        return logits

class ICD10Dataset(Dataset):
    def __init__(self, inputs, queries, all_desc):
        self.inputs = inputs
        self.queries = queries
        self.all_desc = all_desc

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.queries[idx], self.all_desc[idx]

class ICD10Predictor_HS_Head(nn.Module):
    def __init__(self, root, decoder_query_dim=1024):
        super().__init__()

        self.root = root
        # ---------------------------
        # Projection head
        # ---------------------------
        self.proj = nn.Sequential(
            nn.LayerNorm(decoder_query_dim),

            nn.Linear(decoder_query_dim, decoder_query_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(decoder_query_dim * 2, decoder_query_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(decoder_query_dim * 4, decoder_query_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(decoder_query_dim * 2, decoder_query_dim)
        )

        nn.init.xavier_uniform_(self.proj[1].weight)

    def forward(self, output_query):
        # ---------------------------
        # Projection head
        # ---------------------------
        x = self.proj(output_query)
        B, Q, D = x.shape
        x = x.reshape(B * Q, D)  # (N, D)
        return x

class ICD10Predictor_HS_CrossEntropyLoss(nn.Module):
    def __init__(self, root, all_labels_desc, internal_K=1):
        super().__init__()
        self.internal_K = int(internal_K)

        if self.internal_K < 1:
            raise ValueError(f"internal_K must be >= 1, got {self.internal_K}")

        # -------------------------------------------------
        # Dense node indexing
        # -------------------------------------------------
        all_nodes = list(root.node_list)  # real nodes, root excluded
        self.num_real_nodes = len(all_nodes)
        self.root_dense_idx = -1

        self.node_to_dense_all = {node: i for i, node in enumerate(all_nodes)}
        self.dense_to_node_all = {i: node for node, i in self.node_to_dense_all.items()}

        # add root only after real-node indexing is fixed
        self.node_to_dense_all[root] = self.root_dense_idx
        self.dense_to_node_all[self.root_dense_idx] = root

        real_id_to_node = {idx: node for node, idx in root.node_to_id.items()}
        node_to_real_id = {node: idx for idx, node in real_id_to_node.items()}

        # -------------------------------------------------
        # Infer embedding dim from all_labels_desc
        # -------------------------------------------------
        self.embedding_dim = None
        for _, defs in all_labels_desc.items():
            if len(defs) > 0:
                sample_def = defs[0] if isinstance(defs, (list, tuple)) else defs
                if isinstance(sample_def, torch.Tensor):
                    self.embedding_dim = int(sample_def.shape[-1])
                    break

        if self.embedding_dim is None:
            raise ValueError("Could not infer embedding_dim from all_labels_desc.")

        # -------------------------------------------------
        # Node type info
        # -------------------------------------------------
        is_leaf = torch.zeros(self.num_real_nodes, dtype=torch.bool)
        for node in all_nodes:
            node_idx = self.node_to_dense_all[node]
            is_leaf[node_idx] = (len(node.children) == 0)
        self.register_buffer("is_leaf", is_leaf)

        # -------------------------------------------------
        # Fixed leaf definition bank
        # leaf_bank: [N_nodes, max_leaf_defs, D]
        # leaf_mask: [N_nodes, max_leaf_defs]
        # Only leaf rows are filled. Internal rows stay zero/False.
        # -------------------------------------------------
        leaf_counts = torch.zeros(self.num_real_nodes, dtype=torch.long)
        max_leaf_defs = 0

        for node in all_nodes:
            node_idx = self.node_to_dense_all[node]
            if not self.is_leaf[node_idx]:
                continue

            if node not in node_to_real_id:
                raise ValueError(f"Leaf node {node.name} not found in root.node_to_id.")

            real_leaf_id = node_to_real_id[node]
            if real_leaf_id not in all_labels_desc or len(all_labels_desc[real_leaf_id]) == 0:
                raise ValueError(f"Leaf node {node.name} has no definitions in all_labels_desc.")

            defs = self._defs_to_2d_tensor(all_labels_desc[real_leaf_id])
            leaf_counts[node_idx] = defs.size(0)
            max_leaf_defs = max(max_leaf_defs, defs.size(0))

        if max_leaf_defs == 0:
            raise ValueError("No leaf definitions found in all_labels_desc.")

        leaf_bank = torch.zeros(self.num_real_nodes, max_leaf_defs, self.embedding_dim, dtype=torch.float)

        for node in all_nodes:
            node_idx = self.node_to_dense_all[node]
            if not self.is_leaf[node_idx]:
                continue

            real_leaf_id = node_to_real_id[node]
            defs = self._defs_to_2d_tensor(all_labels_desc[real_leaf_id])
            defs = nn.functional.normalize(defs, dim=-1)

            k = defs.size(0)
            leaf_bank[node_idx, :k] = defs

        self.register_buffer("leaf_counts", leaf_counts)
        self.register_buffer("leaf_bank", leaf_bank)
        self.max_leaf_defs = max_leaf_defs

        # -------------------------------------------------
        # Internal prototype bank
        # internal_prototypes: [N_internal, internal_K, D]
        # internal_row_of_node: [N_nodes] -> row in internal bank, -1 for leaves
        # -------------------------------------------------
        internal_nodes = [node for node in all_nodes if len(node.children) > 0]
        num_internal = len(internal_nodes)

        internal_row_of_node = torch.full((self.num_real_nodes,), -1, dtype=torch.long)
        init_internal = []

        for row, node in enumerate(internal_nodes):
            node_idx = self.node_to_dense_all[node]
            internal_row_of_node[node_idx] = row

            desc_leaf_defs = []
            for leaf in node.leaves:
                if leaf.name == "Root":
                    continue
                if leaf not in node_to_real_id:
                    continue

                real_leaf_id = node_to_real_id[leaf]
                if real_leaf_id not in all_labels_desc or len(all_labels_desc[real_leaf_id]) == 0:
                    continue

                leaf_defs = self._defs_to_2d_tensor(all_labels_desc[real_leaf_id])  # [K_leaf, D]
                desc_leaf_defs.append(leaf_defs)

            if len(desc_leaf_defs) > 0:
                init = torch.cat(desc_leaf_defs, dim=0)   # [N_total, D]
                init = nn.functional.normalize(init, dim=-1)
                center = init.mean(dim=0)
                center = nn.functional.normalize(center, dim=0)
            else:
                center = torch.randn(self.embedding_dim)
                center = nn.functional.normalize(center, dim=0)

            init_k = center.unsqueeze(0).repeat(self.internal_K, 1)   # [K, D]
            if self.internal_K > 1:
                noise = 0.01 * torch.randn_like(init_k)
                init_k = init_k + noise
            init_k = nn.functional.normalize(init_k, dim=-1)
            init_internal.append(init_k)

        if num_internal > 0:
            internal_prototypes = torch.stack(init_internal, dim=0)  # [N_internal, K, D]
        else:
            internal_prototypes = torch.empty(0, self.internal_K, self.embedding_dim, dtype=torch.float)

        self.internal_prototypes = nn.Parameter(internal_prototypes)
        self.register_buffer("internal_row_of_node", internal_row_of_node)
        self.num_internal = num_internal

        # -------------------------------------------------
        # Tree helpers
        # children_indices[parent_dense_idx] = [child_dense_idx, ...]
        # Includes root as parent -1
        # -------------------------------------------------
        self.children_indices = {}

        valid_nodes = set(self.node_to_dense_all.keys())
        ordered_nodes = [root] + [n for n in PostOrderIter(root) if n is not root]

        for node in ordered_nodes:
            if node not in valid_nodes:
                continue

            node_idx = self.node_to_dense_all[node]
            child_nodes = [child for child in node.children if child in valid_nodes]

            if len(child_nodes) == 0:
                continue

            self.children_indices[node_idx] = [self.node_to_dense_all[child] for child in child_nodes]

        # -------------------------------------------------
        # Precompute static per-parent level packs
        #
        # For each parent:
        #   - child_global          [C]
        #   - target_to_local       [N_nodes]
        #   - proto_mask            [C, K_parent]
        #   - leaf_padded           [C, K_parent, D]
        #   - internal_local_idx    [n_internal_children]
        #   - internal_rows         [n_internal_children]
        #
        # K_parent = max(
        #   max leaf definitions among leaf children,
        #   internal_K for internal children
        # )
        # -------------------------------------------------
        self.level_info = {}

        for parent_idx, child_list in self.children_indices.items():
            key = self._level_key(parent_idx)

            child_global = torch.tensor(child_list, dtype=torch.long)
            C = child_global.numel()

            target_to_local = torch.full((self.num_real_nodes,), -1, dtype=torch.long)

            level_K = 0
            for local_idx, child_idx in enumerate(child_list):
                target_to_local[child_idx] = local_idx

                if self.is_leaf[child_idx]:
                    k = int(self.leaf_counts[child_idx].item())
                else:
                    k = self.internal_K

                level_K = max(level_K, k)

            leaf_padded = torch.zeros(C, level_K, self.embedding_dim, dtype=torch.float)
            proto_mask = torch.zeros(C, level_K, dtype=torch.bool)

            internal_local_idx = []
            internal_rows = []

            for local_idx, child_idx in enumerate(child_list):
                if self.is_leaf[child_idx]:
                    k = int(self.leaf_counts[child_idx].item())
                    leaf_padded[local_idx, :k] = self.leaf_bank[child_idx, :k]
                    proto_mask[local_idx, :k] = True
                else:
                    proto_mask[local_idx, :self.internal_K] = True
                    internal_local_idx.append(local_idx)
                    internal_rows.append(int(self.internal_row_of_node[child_idx].item()))

            internal_local_idx = torch.tensor(internal_local_idx, dtype=torch.long)
            internal_rows = torch.tensor(internal_rows, dtype=torch.long)

            child_global_name = f"_level_child_global_{key}"
            target_to_local_name = f"_level_target_to_local_{key}"
            proto_mask_name = f"_level_proto_mask_{key}"
            leaf_padded_name = f"_level_leaf_padded_{key}"
            internal_local_idx_name = f"_level_internal_local_idx_{key}"
            internal_rows_name = f"_level_internal_rows_{key}"

            self.register_buffer(child_global_name, child_global)
            self.register_buffer(target_to_local_name, target_to_local)
            self.register_buffer(proto_mask_name, proto_mask)
            self.register_buffer(leaf_padded_name, leaf_padded)
            self.register_buffer(internal_local_idx_name, internal_local_idx)
            self.register_buffer(internal_rows_name, internal_rows)

            self.level_info[parent_idx] = {
                "child_global": child_global_name,
                "target_to_local": target_to_local_name,
                "proto_mask": proto_mask_name,
                "leaf_padded": leaf_padded_name,
                "internal_local_idx": internal_local_idx_name,
                "internal_rows": internal_rows_name,
                "has_internal": bool(internal_rows.numel() > 0),
                "K": level_K,
                "C": C,
            }

    def _normalized_internal_prototypes(self, dtype=None):
        if self.num_internal == 0:
            return self.internal_prototypes

        proto = nn.functional.normalize(self.internal_prototypes, dim=-1)
        if dtype is not None and proto.dtype != dtype:
            proto = proto.to(dtype=dtype)
        return proto

    def _level_logits(self, parent_idx, x_sub, internal_proto_norm, return_target_to_local=False):
        info = self.level_info[parent_idx]

        child_global = getattr(self, info["child_global"])
        proto_mask = getattr(self, info["proto_mask"])
        leaf_padded = getattr(self, info["leaf_padded"])

        P = leaf_padded
        if P.dtype != x_sub.dtype:
            P = P.to(dtype=x_sub.dtype)

        if info["has_internal"]:
            internal_local_idx = getattr(self, info["internal_local_idx"])
            internal_rows = getattr(self, info["internal_rows"])

            P = P.clone()
            P[internal_local_idx, :self.internal_K] = internal_proto_norm[internal_rows]

        sim = torch.einsum("bd,ckd->bck", x_sub, P)      # [b, C, K_parent]
        sim = sim.masked_fill(~proto_mask.unsqueeze(0), -1.0)
        logits_cos = sim.amax(dim=2)                    # [b, C]

        if return_target_to_local:
            target_to_local = getattr(self, info["target_to_local"])
            return logits_cos, child_global, target_to_local

        return logits_cos, child_global, None
    
    def _defs_to_2d_tensor(self, defs):
        if isinstance(defs, torch.Tensor):
            if defs.dim() == 1:
                defs = defs.unsqueeze(0)
            elif defs.dim() != 2:
                raise ValueError(f"defs tensor must have dim 1 or 2, got shape={tuple(defs.shape)}")
            return defs

        if isinstance(defs, (list, tuple)):
            if len(defs) == 0:
                raise ValueError("Empty definitions list.")
            if isinstance(defs[0], torch.Tensor):
                return torch.stack(defs, dim=0)
            raise TypeError(f"defs elements must be tensors, got type={type(defs[0])}")

        raise TypeError(f"Unsupported defs format: {type(defs)}")
    
    def _level_key(self, parent_idx):
        return f"neg{abs(int(parent_idx))}" if int(parent_idx) < 0 else str(int(parent_idx))

    def forward(self):
        pass

    def predict(self, outputs):
        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack(outputs, dim=0)

        x = nn.functional.normalize(outputs, dim=-1)   # [B, D]
        B = x.size(0)

        internal_proto_norm = self._normalized_internal_prototypes(dtype=x.dtype)

        result_total = [[] for _ in range(B)]
        node_idx_total = torch.full((B,), self.root_dense_idx, device=x.device, dtype=torch.long)

        frontier = {self.root_dense_idx: torch.arange(B, device=x.device, dtype=torch.long)}

        while frontier:
            next_frontier = {}

            for parent_idx, batch_idx in frontier.items():
                if batch_idx.numel() == 0:
                    continue

                logits_cos, child_global, _ = self._level_logits(parent_idx=parent_idx, x_sub=x[batch_idx], internal_proto_norm=internal_proto_norm, return_target_to_local=False)

                values, indices = torch.max(logits_cos, dim=1)
                pred_global = child_global[indices]

                node_idx_total[batch_idx] = pred_global

                batch_idx_list = batch_idx.detach().cpu().tolist()
                pred_global_list = pred_global.detach().cpu().tolist()

                for row, b in enumerate(batch_idx_list):
                    result_total[b].append((values[row:row + 1], [pred_global_list[row]]))

                unique_children = torch.unique(pred_global)
                for child_tensor in unique_children:
                    child = int(child_tensor.item())

                    if child not in self.level_info:
                        continue

                    child_mask = (pred_global == child)
                    child_batch_idx = batch_idx[child_mask]

                    if child in next_frontier:
                        next_frontier[child] = torch.cat([next_frontier[child], child_batch_idx], dim=0)
                    else:
                        next_frontier[child] = child_batch_idx

            frontier = next_frontier

        return node_idx_total.detach().cpu().tolist(), result_total

class ICD10Predictor_NO_HS(nn.Module):
    def __init__(self, root, all_labels_desc, decoder_query_dim=1024):
        super().__init__()

        self.root = root
        # ---------------------------
        # Projection head
        # ---------------------------
        self.proj = nn.Sequential(
            nn.LayerNorm(decoder_query_dim),

            nn.Linear(decoder_query_dim, decoder_query_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(decoder_query_dim * 2, decoder_query_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(decoder_query_dim * 4, decoder_query_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(decoder_query_dim * 2, decoder_query_dim)
        )

        nn.init.xavier_uniform_(self.proj[1].weight)

        # ---------------------------
        # Build definition tensors (leafs only)
        # ---------------------------
        leaf_ids = sorted(all_labels_desc.keys())  # these are 36..2192

        leaf_id_to_dense = {leaf_id: i for i, leaf_id in enumerate(leaf_ids)}
        self.dense_to_leaf_leaf_nodes = {i: leaf_id for leaf_id, i in leaf_id_to_dense.items()}

        all_defs = []
        class_ids = []

        for cid, defs in all_labels_desc.items():
            for d in defs:
                all_defs.append(d)
                class_ids.append(leaf_id_to_dense[cid])

        self.definitions = nn.Parameter(torch.stack(all_defs), requires_grad=False)  # (N_defs, D)

        self.num_classes_leaf = len(leaf_ids)
        # print("num_classes:", self.num_classes)

        # ArcFace data
        subcenter_ids = []
        running_counts = {}
        for cid in class_ids:
            idx = running_counts.get(cid, 0)
            subcenter_ids.append(idx)
            running_counts[cid] = idx + 1

        # ---------------------------
        # Buffers
        # ---------------------------
        self.register_buffer("def_class_ids_leaf", torch.tensor(class_ids, dtype=torch.long))
        self.register_buffer("counts_leaf", torch.bincount(self.def_class_ids_leaf, minlength=self.num_classes_leaf).clamp(min=1))
        self.register_buffer("defs_norm_leaf", nn.functional.normalize(self.definitions, dim=-1))
        self.register_buffer("subcenter_ids_leaf", torch.tensor(subcenter_ids, dtype=torch.long))
        
        self.K_leaf = int(self.counts_leaf.max().item())

    def forward(self, output_query):
        # ---------------------------
        # Projection head
        # ---------------------------
        x = self.proj(output_query)
        B, Q, D = x.shape
        x = x.reshape(B * Q, D)  # (N, D)

        # ---------------------------
        # SIM value -> Per class
        # ---------------------------
        class_scores = self.predict(x, self.defs_norm_leaf, self.def_class_ids_leaf, self.subcenter_ids_leaf, self.num_classes_leaf, self.K_leaf)
        return class_scores
    
    def predict(self, input_data, defs_norm, def_class_ids, subcenter_ids, num_classes, K):
        # ---------------------------
        # Cosine similarity
        # ---------------------------
        x_norm = nn.functional.normalize(input_data, dim=-1)
        sim = x_norm @ defs_norm.T  # (N, N_defs)

        # ---------------------------
        # Build S: (N, C, K)
        # ---------------------------
        N = sim.size(0)
        S = sim.new_full((N, num_classes, K), -1.0)

        # place each definition score into its class/subcenter slot
        S[:, def_class_ids, subcenter_ids] = sim    # [B*Q, C, K]
        S_prime = S.max(dim=2).values               # [B*Q, C]
        return S_prime

    
class LLMConfig(BaseModel):
    service: str
    model: str
    lora_model: str|None = None
    device: str
    num_threads: int = 16
    num_interop_threads: int = 2

class DOCXToJSONSConfig(BaseModel):
    report_list: list[Path]
    prompt: str|None
    llm: Any
    semaforo: Semaphore
    modelo_ner: list[SpanClassifier]|None
    node_list: dict|None
    modelo_icd10_head: list[ICD10Predictor_HS_Head|ICD10Predictor_NO_HS]|None
    modelo_icd10_prediction: list[ICD10Predictor_HS_CrossEntropyLoss|None]|None
    label2id_ICD10: list[dict]|None
    id2label_ICD10: list[dict]|None
    id_no_hs_to_id_hs: dict|None
    icd10_thresholds: dict|None

    model_config = {
        "arbitrary_types_allowed": True
    }

class JSONToXLSXConfig(BaseModel):
    df_reference: pd.DataFrame
    CIE10_full_list: list[str]
    df_reference_embeddings: Tensor
    llm: Any
    semaforo: Semaphore
    docx_lista: dict[str, str]
    prompts: dict[str, str]

    model_config = {
        "arbitrary_types_allowed": True
    }

class PipelineContext(BaseModel):
    paths: AppPaths
    ussage: str

    # MUTABLES
    folder_and_archive_name: str
    json_parse: bool = True
    llm_config: LLMConfig
    cie_10_version: str
    
    # MUTABLES (async ONLY)
    MAX_CONCURRENCY: int = 5

    # Variables auxiliares
    vars: dict[str, Any] = Field(default_factory=dict)