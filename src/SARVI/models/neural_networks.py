import torch
import torch.nn as nn
import pandas as pd

from typing import Any
from anytree import PostOrderIter

from .losses import (
    CombinedMarginLoss
)

from ..services.common.utils.ner_utils import (
    average_overlapping_hidden_states
)
from ..services.common.span_funcs import (
    construct_span_data
)

class NERClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, freeze_encoder: bool = False):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch: dict | None = None, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, global_token_indices: torch.Tensor | None = None, **encoder_kwargs) -> dict:
        if batch is not None:
            input_ids = batch["input_ids"] if input_ids is None else input_ids
            attention_mask = batch.get("attention_mask", attention_mask)
            global_token_indices = batch["global_token_indices"] if global_token_indices is None else global_token_indices

        if input_ids is None:
            raise ValueError("input_ids is required")
        if global_token_indices is None:
            raise ValueError("global_token_indices is required")

        if any(param.requires_grad for param in self.encoder.parameters()):
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs)
        else:
            with torch.no_grad():
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs)

        last_hidden_state = encoder_outputs.last_hidden_state
        merged_hidden_state = average_overlapping_hidden_states(last_hidden_state=last_hidden_state, global_token_indices=global_token_indices, attention_mask=attention_mask)

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "global_token_indices": global_token_indices, "last_hidden_state": merged_hidden_state, "encoder_outputs": encoder_outputs}

        if batch is not None:
            for key in ("window_to_text", "window_local_index", "text_indices", "file_names", "text_token_offsets", "text_token_lengths", "window_labels", "window_offset_mapping"):
                if key in batch:
                    out[key] = batch[key]

        return out

class Span_NERClassifier(NERClassifier):
    def __init__(self, encoder: nn.Module, num_labels: int, id2label: dict, label2id: dict, dropout=0.1, tokenizer: Any|None = None, skip_incomplete_spans: bool = True, lexicon: pd.DataFrame|None = None, freeze_encoder: bool = False):
        super().__init__(encoder=encoder, freeze_encoder=freeze_encoder)

        self.id2label = id2label
        self.label2id = label2id
        self.skip_incomplete_spans = skip_incomplete_spans
        self.tokenizer = tokenizer
        self.lexicon = lexicon

        input_dim = encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_labels)
        )

    def forward(self, batch: dict) -> dict:
        base_out = super().forward(batch=batch)
        base_out = construct_span_data(base_out, id2label=self.id2label, label2id=self.label2id, tokenizer=self.tokenizer, skip_incomplete_spans=self.skip_incomplete_spans, lexicon=self.lexicon)

        hidden_states = base_out["last_hidden_state"]
        logits = self.classifier(hidden_states)
        out = {**base_out, "logits": logits}

        return out
    
class BIO_NERClassifier(NERClassifier):
    def __init__(self, encoder: nn.Module, num_labels: int, dropout=0.1):
        super().__init__(encoder=encoder)

        input_dim = encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_labels)
        )

    def forward(self, batch: dict) -> dict:
        base_out = super().forward(batch=batch)

        hidden_states = base_out["last_hidden_state"]
        logits = self.classifier(hidden_states)
        out = {**base_out, "logits": logits}

        return out

class REClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_labels: int, middle_token_count_bin_embeddings: int | None = None, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()

        if middle_token_count_bin_embeddings is None or middle_token_count_bin_embeddings < 1:
            raise ValueError("middle_token_count_bin_embeddings is required and must be >= 1")

        self.encoder = encoder
        self.num_labels = num_labels
        self.middle_token_count_bin_embedding_count = int(middle_token_count_bin_embeddings)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        input_dim = encoder.config.hidden_size
        classifier_input_dim = input_dim * 3
        
        self.middle_token_count_bin_embeddings = nn.Embedding(self.middle_token_count_bin_embedding_count, input_dim)
        nn.init.xavier_uniform_(self.middle_token_count_bin_embeddings.weight)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim * 2, num_labels)
        )

    def gather_token_embeddings(self, hidden_states: torch.Tensor, positions: torch.Tensor, name: str) -> torch.Tensor:
        if positions.dim() != 1:
            raise ValueError(f"{name} must have shape [batch_size], got {tuple(positions.shape)}")
        if positions.numel() != hidden_states.size(0):
            raise ValueError(f"{name} must have one position per example.")
        if positions.lt(0).any():
            raise ValueError(f"{name} contains missing positions: {positions.detach().cpu().tolist()}")
        if positions.ge(hidden_states.size(1)).any():
            raise ValueError(f"{name} contains positions outside sequence length {hidden_states.size(1)}")

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    def forward(self, batch: dict) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        marker_position_ids = batch["marker_position_ids"].to(device=input_ids.device, dtype=torch.long)
        sep_position_ids = batch["sep_position_ids"].to(device=input_ids.device, dtype=torch.long)
        middle_token_count_bin = batch["middle_token_count_bin"].to(device=input_ids.device, dtype=torch.long)

        if marker_position_ids.dim() != 2 or marker_position_ids.size(1) != 4:
            raise ValueError(f"marker_position_ids must have shape [batch_size, 4], got {tuple(marker_position_ids.shape)}")

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        subject_start = self.gather_token_embeddings(hidden_states, marker_position_ids[:, 0], "subject_start marker positions")
        subject_end = self.gather_token_embeddings(hidden_states, marker_position_ids[:, 1], "subject_end marker positions")
        object_start = self.gather_token_embeddings(hidden_states, marker_position_ids[:, 2], "object_start marker positions")
        object_end = self.gather_token_embeddings(hidden_states, marker_position_ids[:, 3], "object_end marker positions")
        sep_embedding = self.gather_token_embeddings(hidden_states, sep_position_ids, "sep positions")

        middle_token_count_bin_for_embedding = middle_token_count_bin.clamp(min=0, max=self.middle_token_count_bin_embedding_count - 1)
        middle_embedding = self.middle_token_count_bin_embeddings(middle_token_count_bin_for_embedding)

        subject_embedding = torch.stack([subject_start, subject_end], dim=1).max(dim=1).values
        object_embedding = torch.stack([object_start, object_end], dim=1).max(dim=1).values
        relation_token_embeddings = torch.stack([subject_embedding, object_embedding, sep_embedding + middle_embedding,], dim=1)

        logits = self.classifier(relation_token_embeddings)

        return {"logits": logits}
    
class ATTClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()

        self.encoder = encoder
        self.num_labels = num_labels

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        input_dim = encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_labels)
        )

    def gather_token_embeddings(self, hidden_states: torch.Tensor, positions: torch.Tensor, name: str) -> torch.Tensor:
        if positions.dim() != 1:
            raise ValueError(f"{name} must have shape [batch_size], got {tuple(positions.shape)}")
        if positions.numel() != hidden_states.size(0):
            raise ValueError(f"{name} must have one position per example.")
        if positions.lt(0).any():
            raise ValueError(f"{name} contains missing positions: {positions.detach().cpu().tolist()}")
        if positions.ge(hidden_states.size(1)).any():
            raise ValueError(f"{name} contains positions outside sequence length {hidden_states.size(1)}")

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    def forward(self, batch: dict) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        marker_position_ids = batch["marker_position_ids"].to(device=input_ids.device, dtype=torch.long)

        if marker_position_ids.dim() != 2 or marker_position_ids.size(1) != 2:
            raise ValueError(f"marker_position_ids must have shape [batch_size, 2], got {tuple(marker_position_ids.shape)}")

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        subject_start = self.gather_token_embeddings(hidden_states, marker_position_ids[:, 0], "subject_start marker positions")
        subject_end = self.gather_token_embeddings(hidden_states, marker_position_ids[:, 1], "subject_end marker positions")

        subject_embedding = torch.stack([subject_start, subject_end], dim=1).max(dim=1).values

        logits = self.classifier(subject_embedding)

        return {"logits": logits}


class ICD10Predictor(nn.Module):
    def __init__(self, encoder, tokenizer, device, freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = encoder
        self.tokenizer = tokenizer if tokenizer is not None else getattr(encoder, "tokenizer", None)
        self.device = device
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, diagnoses):
        if torch.is_tensor(diagnoses):
            if diagnoses.dim() == 2:
                diagnoses = diagnoses.unsqueeze(0)
            return diagnoses

        if self.encoder is None or self.tokenizer is None:
            raise ValueError("encoder and tokenizer are required when ICD10Predictor receives raw diagnosis text")
        
        texts, batch_size, query_count = self.flatten_diagnosis_texts(diagnoses)
        proj_dtype = next(self.proj.parameters()).dtype
        cache = {}
        new_texts = list(set(texts))

        tokens = self.tokenizer(new_texts, padding=True, truncation=True, max_length=self.encoder.config.max_position_embeddings, return_tensors="pt", add_special_tokens=False)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}

        encoder_is_trainable = torch.is_grad_enabled() and any(param.requires_grad for param in self.encoder.parameters())
        context = torch.enable_grad() if encoder_is_trainable else torch.no_grad()
        with context:
            outputs = self.encoder(**tokens)

        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

        for text, emb in zip(new_texts, pooled):
            cache[text] = emb

        embeddings = torch.stack([cache[text] for text in texts])
        return embeddings.to(dtype=proj_dtype).reshape(batch_size, query_count, -1)

    def flatten_diagnosis_texts(self, diagnoses):
        if isinstance(diagnoses, str):
            return [diagnoses], 1, 1

        if not isinstance(diagnoses, (list, tuple)):
            raise TypeError(f"Unsupported diagnoses type: {type(diagnoses)}")

        if len(diagnoses) == 0:
            raise ValueError("diagnoses cannot be empty")

        if all(isinstance(item, str) for item in diagnoses):
            return list(diagnoses), 1, len(diagnoses)

        if all(isinstance(item, (list, tuple)) for item in diagnoses):
            query_counts = [len(item) for item in diagnoses]
            if len(set(query_counts)) != 1:
                raise ValueError("Raw ICD diagnosis batches must have the same number of diagnoses per item")
            texts = [text for item in diagnoses for text in item]
            if not all(isinstance(text, str) for text in texts):
                raise TypeError("Raw ICD diagnoses must be strings")
            return texts, len(diagnoses), query_counts[0]

        raise TypeError("Raw ICD diagnoses must be a string, a list of strings, or a nested list of strings")


class ICD10Predictor_HS_Head(ICD10Predictor):
    def __init__(self, root, device, decoder_query_dim=1024, encoder=None, tokenizer=None, freeze_encoder: bool = False):
        super().__init__(encoder=encoder, tokenizer=tokenizer, device=device, freeze_encoder=freeze_encoder)

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

    def forward(self, diagnoses):
        diagnoses = super().forward(diagnoses=diagnoses)
        # ---------------------------
        # Projection head
        # ---------------------------
        x = self.proj(diagnoses)
        B, Q, D = x.shape
        x = x.reshape(B * Q, D)  # (N, D)
        return x

class ICD10Predictor_HS_CrossEntropyLoss(nn.Module):
    def __init__(self, root, all_labels_desc, device, weight=None, reduction="mean", internal_K=1):
        super().__init__()
        self.register_buffer("weight", weight)
        self.reduction = reduction
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

        # -------------------------------------------------
        # Target paths for ArcFace training
        # -------------------------------------------------
        max_depth = 0
        for node in all_nodes:
            max_depth = max(max_depth, len(node.path) - 1)

        path_parents = torch.full((self.num_real_nodes, max_depth), self.root_dense_idx, dtype=torch.long)
        path_children = torch.full((self.num_real_nodes, max_depth), -1, dtype=torch.long)
        path_alphas = torch.zeros((self.num_real_nodes, max_depth), dtype=torch.float)
        path_valid = torch.zeros((self.num_real_nodes, max_depth), dtype=torch.bool)

        for node in all_nodes:
            node_idx = self.node_to_dense_all[node]
            correct_nodes = [self.node_to_dense_all[n] for n in node.path[1:]]

            for step_idx, (correct_node_level, ancestor) in enumerate(zip(correct_nodes, node.ancestors)):
                parent_idx = self.node_to_dense_all[ancestor]
                alpha = float(self.dense_to_node_all[correct_node_level].alpha)

                path_parents[node_idx, step_idx] = parent_idx
                path_children[node_idx, step_idx] = correct_node_level
                path_alphas[node_idx, step_idx] = alpha
                path_valid[node_idx, step_idx] = True

        self.register_buffer("path_parents", path_parents)
        self.register_buffer("path_children", path_children)
        self.register_buffer("path_alphas", path_alphas)
        self.register_buffer("path_valid", path_valid)
        self.max_depth = max_depth

        # Official ArcFace settings in arcface_torch: m1=1.0, m2=0.5, m3=0.0  -> ArcFace branch
        self.arcface = CombinedMarginLoss(s=64.0, m1=1.0, m2=0.5, m3=0.0, interclass_filtering_threshold=0.0).to(device)

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

    def forward(self, outputs, targets=None, inference=False):
        if inference and targets is None:
            return self.predict(outputs)

        if targets is None:
            raise ValueError("targets are required when inference=False")

        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack(outputs, dim=0)
        if isinstance(targets, (list, tuple)):
            targets = torch.as_tensor(targets, device=outputs.device)

        x = nn.functional.normalize(outputs, dim=-1)
        targets = targets.to(device=outputs.device).view(-1).long()
        B = x.size(0)

        path_parents = self.path_parents[targets]
        path_children = self.path_children[targets]
        path_alphas = self.path_alphas[targets]
        path_valid = self.path_valid[targets]

        internal_proto_norm = self._normalized_internal_prototypes(dtype=x.dtype)

        if inference:
            num_steps_per_sample = path_valid.sum(dim=1).tolist()
            result_total = [[None] * int(num_steps_per_sample[b]) for b in range(B)]

            for step_idx in range(self.max_depth):
                valid = path_valid[:, step_idx]
                if not torch.any(valid):
                    continue

                batch_idx_all = torch.nonzero(valid, as_tuple=False).squeeze(1)
                step_parents = path_parents[batch_idx_all, step_idx]

                for parent_tensor in torch.unique(step_parents):
                    parent_idx = int(parent_tensor.item())
                    group_mask = (step_parents == parent_tensor)
                    sample_idx = batch_idx_all[group_mask]

                    logits_cos, child_global, _ = self._level_logits(parent_idx=parent_idx, x_sub=x[sample_idx], internal_proto_norm=internal_proto_norm, return_target_to_local=False)

                    values, indices = torch.max(logits_cos, dim=1)
                    pred_global = child_global[indices]

                    for row, b in enumerate(sample_idx.detach().cpu().tolist()):
                        result_total[b][step_idx] = (values[row:row + 1], [int(pred_global[row].detach().cpu().item())])

            return result_total

        per_sample_loss = torch.zeros(B, device=x.device, dtype=x.dtype)

        for step_idx in range(self.max_depth):
            valid = path_valid[:, step_idx]
            if not torch.any(valid):
                continue

            batch_idx_all = torch.nonzero(valid, as_tuple=False).squeeze(1)
            step_parents = path_parents[batch_idx_all, step_idx]
            step_children = path_children[batch_idx_all, step_idx]
            step_alpha = path_alphas[batch_idx_all, step_idx].to(dtype=x.dtype)

            for parent_tensor in torch.unique(step_parents):
                parent_idx = int(parent_tensor.item())
                group_mask = (step_parents == parent_tensor)

                sample_idx = batch_idx_all[group_mask]
                correct_child_global = step_children[group_mask]
                alpha = step_alpha[group_mask]

                logits_cos, _, target_to_local = self._level_logits(parent_idx=parent_idx, x_sub=x[sample_idx], internal_proto_norm=internal_proto_norm, return_target_to_local=True)

                target_local = target_to_local[correct_child_global]
                logits_cos = logits_cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                logits = self.arcface(logits_cos.clone(), target_local)

                ce = nn.functional.cross_entropy(logits, target_local, weight=self.weight, reduction="none")
                per_sample_loss.index_add_(0, sample_idx, alpha * ce)

        if self.reduction == "none":
            return per_sample_loss
        if self.reduction == "sum":
            return per_sample_loss.sum()
        if self.reduction == "mean":
            return per_sample_loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")

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

class ICD10Predictor_NO_HS(ICD10Predictor):
    def __init__(self, root, device, all_labels_desc, decoder_query_dim=1024, encoder=None, tokenizer=None, freeze_encoder: bool = False):
        super().__init__(encoder=encoder, tokenizer=tokenizer, device=device, freeze_encoder=freeze_encoder)

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

        # Official ArcFace settings in arcface_torch: m1=1.0, m2=0.5, m3=0.0  -> ArcFace branch
        self.arcface = CombinedMarginLoss(s=64.0, m1=1.0, m2=0.5, m3=0.0, interclass_filtering_threshold=0.0).to(device)

    def forward(self, diagnoses, inference, targets):
        diagnoses = super().forward(diagnoses=diagnoses)
        # ---------------------------
        # Projection head
        # ---------------------------
        x = self.proj(diagnoses)
        B, Q, D = x.shape
        x = x.reshape(B * Q, D)  # (N, D)

        # ---------------------------
        # SIM value -> Per class
        # ---------------------------
        class_scores = self.predict(x, self.defs_norm_leaf, self.def_class_ids_leaf, self.subcenter_ids_leaf, self.num_classes_leaf, self.K_leaf, inference, targets)
        return class_scores
    
    def predict(self, input_data, defs_norm, def_class_ids, subcenter_ids, num_classes, K, inference, targets):
        # ---------------------------
        # Cosine similarity
        # ---------------------------
        x_norm = nn.functional.normalize(input_data, dim=-1)
        defs_norm = defs_norm.to(device=input_data.device, dtype=input_data.dtype)
        def_class_ids = def_class_ids.to(device=input_data.device)
        subcenter_ids = subcenter_ids.to(device=input_data.device)
        sim = x_norm @ defs_norm.T  # (N, N_defs)

        # ---------------------------
        # Build S: (N, C, K)
        # ---------------------------
        N = sim.size(0)
        S = sim.new_full((N, num_classes, K), -1.0)

        # place each definition score into its class/subcenter slot
        S[:, def_class_ids, subcenter_ids] = sim    # [B*Q, C, K]
        S_prime = S.max(dim=2).values               # [B*Q, C]
        if inference:
            return S_prime
        
        S_prime = S_prime.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        logits = self.arcface(S_prime.clone(), targets)
        return logits
