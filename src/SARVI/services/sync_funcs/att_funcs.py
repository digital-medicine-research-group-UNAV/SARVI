from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report as classification_report_sklearn,
    f1_score as f1_score_sklearn,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...models.datasets import (
    ATTDataset
)
from ...models.neural_networks import (
    ATTClassifier
)
from ..common.att_funcs import (
    att_collate_fn,
    attribute_label_for_entity,
    build_att_entity_window,
    parse_att_attributes,
    parse_att_target_entities,
)
from ..common.utils.nn_utils import (
    move_batch_to_device,
    ensure_entity_marker_tokens
)
from ..common.utils.re_utils import (
    resolve_max_length
)


def prepare_data(data: pd.DataFrame, data_ann: pd.DataFrame, entity_label: str, tokenizer: Any, padding: bool = True, max_length: int | None = None, w: int = 1, print_warnings: bool = True, file_col: str = "archivo_origen", text_col: str = "Text", t_col: str = "T", a_col: str = "A", attribute_label: str | None = "Status", outside_label: str = "O", return_warnings: bool = False, return_tokenizer: bool = False) -> tuple:
    """
    Prepares one ATT classification example per target entity.

    Each example marks a single `entity_label` entity with subject markers
    (`<S:...>`, `</S:...>`) and uses `w` sentences on both sides of that
    entity. Attribute labels are read from the brat `A` column.
    """
    if file_col not in data.columns:
        raise ValueError(f"data must contain '{file_col}'.")
    if text_col not in data.columns:
        raise ValueError(f"data must contain '{text_col}'.")
    if file_col not in data_ann.columns:
        raise ValueError(f"data_ann must contain '{file_col}'.")
    if t_col not in data_ann.columns:
        raise ValueError(f"data_ann must contain '{t_col}'.")

    resolved_max_length = resolve_max_length(tokenizer, max_length=max_length)
    ensure_entity_marker_tokens(tokenizer, [entity_label], "att")

    data_prepared = []
    all_attribute_labels = []
    preparation_warnings = []
    file_names = []
    ann_by_file = {row[file_col]: row for _, row in data_ann.iterrows()}

    for _, text_info in tqdm(data.iterrows(), total=len(data), desc="Preparing data for ATT prediction: Entities and marked contexts", unit="text"):
        file_name = text_info[file_col]
        text = text_info[text_col]
        ann_info = ann_by_file.get(file_name)

        if ann_info is None:
            raise ValueError(f"No annotation row found for file {file_name}")

        entities = parse_att_target_entities(ann_info, entity_label, t_col=t_col)
        attributes = parse_att_attributes(ann_info, a_col=a_col) if a_col in data_ann.columns else []

        examples = []
        attribute_labels = []
        for entity in entities:
            example, warning_message = build_att_entity_window(text=text, entity=entity, tokenizer=tokenizer, max_length=resolved_max_length, w=w, padding=padding)
            if warning_message is not None:
                warning = {"file_name": file_name, "entity_id": entity.get("id"), "message": warning_message}
                preparation_warnings.append(warning)
                if print_warnings:
                    warnings.warn(f"{file_name} {entity.get('id')}: {warning_message}", RuntimeWarning)
                continue

            example["file_name"] = file_name
            example["entity_label"] = entity_label
            attribute_value = attribute_label_for_entity(entity, attributes, attribute_label=attribute_label, outside_label=outside_label)
            example["attribute_label"] = attribute_value

            examples.append(example)
            attribute_labels.append(attribute_value)

        data_prepared.append(examples)
        all_attribute_labels.append(attribute_labels)
        file_names.append(file_name)

    if return_tokenizer:
        return data_prepared, all_attribute_labels, file_names, preparation_warnings, tokenizer

    return data_prepared, all_attribute_labels, file_names, preparation_warnings


def construct_loader_att(data: list[list[dict[str, Any]]], attribute_labels: list[list[str | int]] | None = None, file_names: list[Any] | None = None, label2id: dict[str, int] | None = None, id2label: dict[int, str] | None = None, outside_label: str = "O", batch_size: int = 8, shuffle: bool = False, seed: int = 8, flatten_examples: bool = True) -> tuple[ATTDataset, DataLoader, dict[str, int], dict[int, str]]:
    """
    Builds the ATT dataset and DataLoader.

    By default `flatten_examples=True`, so each DataLoader item is one target
    entity example instead of one full text with all of its entities.
    """
    if file_names is None:
        file_names = [examples[0].get("file_name") if examples else None for examples in data]

    attribute_label_ids = None
    if attribute_labels is not None:
        if label2id is None:
            labels = [outside_label]
            labels.extend(label for labels_per_text in attribute_labels for label in labels_per_text if isinstance(label, str) and label not in labels)
            label2id = {label: idx for idx, label in enumerate(labels)}
            id2label = {idx: label for label, idx in label2id.items()}
        elif id2label is None:
            id2label = {idx: label for label, idx in label2id.items()}

        attribute_label_ids = [[label2id[label] if isinstance(label, str) else int(label) for label in labels_per_text] for labels_per_text in attribute_labels]
    else:
        if label2id is None:
            label2id = {outside_label: 0}
        if id2label is None:
            id2label = {idx: label for label, idx in label2id.items()}

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    data_dataset = ATTDataset(data, attribute_label_ids, file_names, flatten_examples=flatten_examples)
    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=att_collate_fn, generator=generator, worker_init_fn=seed_worker)

    return data_dataset, data_loader, label2id, id2label


def run_att_classifier(model: "ATTClassifier", data_loader: DataLoader, device: torch.device, id2label: dict, train: bool = False, dev_data_loader: DataLoader | None = None, optimizer: torch.optim.Optimizer | None = None, criterion: torch.nn.Module | None = None, epochs: int = 1, label_key: str = "attribute_labels", ignore_index: int = -100, patience: int = 10, return_predictions: bool = True):
    """
    Runs ATTClassifier in training or inference mode.
    """
    model.to(device)

    loss_values = []
    development_loss_values = []
    development_f1_values = []

    all_pred = None
    all_value_preds = None
    eval_loss = None
    final_classification_report = None

    best_macro_f1 = -float("inf")
    best_epoch = None
    best_model_state = None
    epochs_without_improvement = 0

    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        if criterion is None:
            raise ValueError("criterion is required when train=True")

        for epoch in tqdm(range(epochs), desc="ATT / Training", total=epochs, unit="epoch"):
            model.train()
            epoch_losses = []

            for batch in tqdm(data_loader, desc=f"ATT / Training epoch {epoch + 1} / Train set", unit="batch", total=len(data_loader), leave=False):
                if batch["input_ids"].numel() == 0:
                    continue
                batch = move_batch_to_device(batch, device)
                labels = batch[label_key]
                if labels is None:
                    raise ValueError(f"Training requires `{label_key}` in the batch")

                optimizer.zero_grad(set_to_none=True)

                output = model(batch=batch)
                logits = output["logits"]

                loss = criterion(logits, labels)
                loss_value = loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss_value)

                del output, logits, loss, labels, batch

            train_loss_epoch = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0
            loss_values.append(train_loss_epoch)

            if dev_data_loader is not None:
                model.eval()

                dev_losses = []
                dev_pred_batches = []
                dev_value_pred_batches = []
                dev_y_true_all = []
                dev_y_pred_all = []

                with torch.no_grad():
                    for dev_batch in tqdm(dev_data_loader, desc=f"ATT / Training epoch {epoch + 1} / Dev set", unit="batch", total=len(dev_data_loader), leave=False):
                        if dev_batch["input_ids"].numel() == 0:
                            continue
                        dev_batch = move_batch_to_device(dev_batch, device)

                        dev_output = model(batch=dev_batch)
                        dev_logits = dev_output["logits"]

                        dev_value_preds = torch.softmax(dev_logits, dim=-1)
                        dev_preds = torch.argmax(dev_value_preds, dim=-1)

                        dev_labels = dev_batch.get(label_key)
                        if dev_labels is not None:
                            dev_loss = criterion(dev_logits, dev_labels)
                            dev_losses.append(dev_loss.item())

                            dev_mask = dev_labels != ignore_index
                            dev_preds_cpu = dev_preds.detach().cpu()
                            dev_labels_cpu = dev_labels.detach().cpu()
                            dev_mask_cpu = dev_mask.detach().cpu()

                            for pred_id, true_id, valid_position in zip(dev_preds_cpu, dev_labels_cpu, dev_mask_cpu):
                                if not bool(valid_position):
                                    continue

                                true_id = int(true_id.item())
                                pred_id = int(pred_id.item())

                                if true_id == ignore_index:
                                    continue

                                dev_y_true_all.append(id2label[true_id])
                                dev_y_pred_all.append(id2label[pred_id])

                            if return_predictions:
                                dev_pred_batches.append(dev_preds[dev_mask].detach().cpu().numpy())
                                dev_value_pred_batches.append(dev_value_preds[dev_mask].detach().cpu().numpy())

                dev_loss_epoch = float(np.mean(dev_losses)) if len(dev_losses) > 0 else 0.0
                development_loss_values.append(dev_loss_epoch)

                if len(dev_y_true_all) > 0:
                    dev_macro_f1 = f1_score_sklearn(dev_y_true_all, dev_y_pred_all, average="macro", zero_division=0)
                    dev_report = classification_report_sklearn(dev_y_true_all, dev_y_pred_all, digits=4, zero_division=0)
                else:
                    dev_macro_f1 = None
                    dev_report = None

                development_f1_values.append(dev_macro_f1)
                final_classification_report = dev_report

                if return_predictions and len(dev_pred_batches) > 0:
                    all_pred = np.concatenate(dev_pred_batches, axis=0)
                    all_value_preds = np.concatenate(dev_value_pred_batches, axis=0)

                if dev_report is not None:
                    print("\nDevelopment classification report:")
                    print(dev_report)

                if dev_macro_f1 is not None and dev_macro_f1 > best_macro_f1:
                    best_macro_f1 = dev_macro_f1
                    best_epoch = epoch + 1
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best epoch: {best_epoch} | Best dev macro F1: {best_macro_f1:.6f}")
                    break

                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f} | dev loss: {dev_loss_epoch:.6f} | Best F1 (macro): {best_macro_f1:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f}")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)
    else:
        model.eval()

        eval_losses = []
        pred_batches = []
        value_pred_batches = []
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="ATT / Inference", total=len(data_loader), unit="batch"):
                if batch["input_ids"].numel() == 0:
                    continue
                batch = move_batch_to_device(batch, device)

                output = model(batch=batch)
                logits = output["logits"]

                value_preds = torch.softmax(logits, dim=-1)
                preds = torch.argmax(value_preds, dim=-1)

                labels = batch.get(label_key)
                if labels is not None:
                    if criterion is not None:
                        loss = criterion(logits, labels)
                        eval_losses.append(loss.item())

                    mask = labels != ignore_index
                    preds_cpu = preds.detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    mask_cpu = mask.detach().cpu()

                    for pred_id, true_id, valid_position in zip(preds_cpu, labels_cpu, mask_cpu):
                        if not bool(valid_position):
                            continue

                        true_id = int(true_id.item())
                        pred_id = int(pred_id.item())

                        if true_id == ignore_index:
                            continue

                        y_true_all.append(id2label[true_id])
                        y_pred_all.append(id2label[pred_id])

                    if return_predictions:
                        pred_batches.append(preds[mask].detach().cpu().numpy())
                        value_pred_batches.append(value_preds[mask].detach().cpu().numpy())
                elif return_predictions:
                    pred_batches.append(preds.detach().cpu().numpy())
                    value_pred_batches.append(value_preds.detach().cpu().numpy())

        if len(eval_losses) > 0:
            eval_loss = float(np.mean(eval_losses))

        if len(y_true_all) > 0:
            best_macro_f1 = f1_score_sklearn(y_true_all, y_pred_all, average="macro", zero_division=0)
            final_classification_report = classification_report_sklearn(y_true_all, y_pred_all, digits=4, zero_division=0)

            print("\nClassification report:")
            print(final_classification_report)
        else:
            best_macro_f1 = None
            final_classification_report = None

        if return_predictions and len(pred_batches) > 0:
            all_pred = np.concatenate(pred_batches, axis=0)
            all_value_preds = np.concatenate(value_pred_batches, axis=0)

    return {"all_pred": all_pred, "all_value_preds": all_value_preds, "loss_values": loss_values, "development_loss_values": development_loss_values, "development_f1_values": development_f1_values, "eval_loss": eval_loss, "classification_report": final_classification_report, "best_macro_f1": best_macro_f1, "best_epoch": best_epoch, "best_model_state": best_model_state}
