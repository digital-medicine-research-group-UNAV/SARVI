from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from ..common.utils.icd_preds_utils import (
    annotation_row_for_data_row, remap_icd_prediction_ids, as_prediction_id_list,
    flatten_optional_nested, normalize_icd_code_label
)
from ..common.utils.nn_utils import(
    tensor_to_python
)
from ..common.utils.ann_utils import(
    label_from_id
)

from ..common.icd_pred_funcs import (
    build_icd_label_maps, extract_diso_diagnoses, extract_flattened_predictions, classification_report_icd_flatten
)

from ...models.datasets import (
    ICD10Dataset
)
if TYPE_CHECKING:
    from ...models.neural_networks import ICD10Predictor_HS_Head, ICD10Predictor_HS_CrossEntropyLoss, ICD10Predictor_NO_HS

def prepare_data(data: pd.DataFrame, data_ann: pd.DataFrame | None = None, file_col: str = "archivo_origen", text_col: str = "Text", t_col: str = "T", code_col: str = "#", allow_incomplete_code_extraction: bool = False, output_only_max_conf: bool = False, output_only_judged_if_possible: bool = False):
    """
    Prepara datos ICD-10 para construir `ICD10Dataset`.

    Extrae entidades DISO desde las líneas BRAT de `T`, usa sus offsets de
    caracteres sobre el texto original como diagnósticos, y opcionalmente extrae
    códigos ICD desde las líneas `AnnotatorNotes` de la columna `#`.
    """
    if data_ann is None:
        raise ValueError("data_ann is required to extract ICD diagnoses from BRAT T annotations")
    if text_col not in data.columns:
        raise ValueError(f"data must contain `{text_col}`")
    if file_col not in data.columns:
        raise ValueError(f"data must contain `{file_col}`")
    if t_col not in data_ann.columns:
        raise ValueError(f"data_ann must contain `{t_col}`")

    data = data.reset_index(drop=True)

    diagnoses = []
    icd_codes = []
    file_names = []
    has_codes = True

    for row_pos, (_, row) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Preparing data for ICD10 prediction", unit="text")):
        ann_row = annotation_row_for_data_row(row, data_ann, row_pos)
        diagnosis_list, code_list = extract_diso_diagnoses(str(row[text_col]), ann_row, t_col=t_col, code_col=code_col, output_only_max_conf=output_only_max_conf, output_only_judged_if_possible=output_only_judged_if_possible)

        diagnoses.extend(diagnosis_list)
        file_names.extend([row[file_col]] * len(diagnosis_list))

        if code_list is None and allow_incomplete_code_extraction == False:
            has_codes = False
        elif has_codes and code_list is not None:
            icd_codes.extend(code_list)

    if len(diagnoses) == 0:
        raise ValueError("No DISO diagnoses found in data_ann")

    data_prepared = {
        "diagnoses": diagnoses,
        "file_names": file_names,
    }

    all_icd_codes = None
    if allow_incomplete_code_extraction == True or (has_codes and len(icd_codes) == len(diagnoses)):
        all_icd_codes = icd_codes
        data_prepared["icd_codes"] = all_icd_codes

    return data_prepared, all_icd_codes, file_names

def icd10_collate_fn(batch):
    """Collate function for diagnosis-level ICD10 batches.
    
    Parameters
    ----------
    batch : list
        Batch from DataLoader containing diagnosis-level items.
        
    Returns
    -------
    tuple
        `(diagnoses, icd_codes)` with proper batch dimensions:
        - diagnoses: diagnosis strings
        - icd_code: ICD labels if present
    """
    diagnoses, icd_codes = zip(*batch)

    if torch.is_tensor(diagnoses[0]):
        diagnoses = torch.stack(diagnoses)
    else:
        diagnoses = list(diagnoses)

    if all(code is None for code in icd_codes):
        icd_codes = None
    elif all(torch.is_tensor(code) for code in icd_codes):
        icd_codes = torch.stack(icd_codes).reshape(-1)
    else:
        icd_codes = torch.tensor(icd_codes, dtype=torch.long)

    return diagnoses, icd_codes

def construct_dataset_icd10(data: pd.DataFrame, data_ann: pd.DataFrame, **prepare_kwargs):
    """
    Builds the ICD-10 dataset from the prepared BRAT dataframe.

    Parameters
    ----------
        `data`: pd.DataFrame
            - Input dataframe containing the records to process with `archivo_origen` and `Text`.
        `data_ann`: pd.DataFrame
            - Annotation data used to align entities, attributes, or relations. It has BRAT annotations in `T`, and optionally in `#`

    Returns
    -------
        tuple
            - `(diagnoses, file_names, icd_codes)`
    """
    data_prepared, _, _ = prepare_data(data, data_ann=data_ann, **prepare_kwargs)
    return data_prepared["diagnoses"], data_prepared["file_names"], data_prepared.get("icd_codes")

def construct_loaders_icd(data_prepared: dict, all_window_labels: list | None, file_names: list | None, hs: bool, root, label2id: dict | None = None, id2label: dict | None = None, batch_size: int = 1, seed: int = 8, shuffle: bool = True):
    """
    Creates ICD-10 training and evaluation data loaders.

    Parameters
    ----------
        `data_prepared`: dict
            - Prepared examples grouped by source document.
        `all_window_labels`: list | None
            - Entity or relation labels used by the model.
        `file_names`: list | None
            - Argument controlling file names.
        `hs`: bool
            - Whether hierarchical softmax is used.
        `root`: Any
            - Root node of the ICD hierarchy.
        `label2id`: dict | None
            - Mapping from labels to numeric identifiers.
        `id2label`: dict | None
            - Mapping from numeric identifiers to labels.
        `batch_size`: int
            - Number of examples processed in one batch.
        `seed`: int
            - Seed used to make random sampling reproducible.
        `shuffle`: bool
            - Whether to shuffle the examples.

    Returns
    -------
        `Any`
            - Constructed data ready for the next pipeline step.
    """
    label2id, id2label = build_icd_label_maps(root=root, hs=hs, label2id=label2id, id2label=id2label)

    labels_flat = flatten_optional_nested(all_window_labels)
    window_labels_clean = None if labels_flat is None else [normalize_icd_code_label(label) for label in labels_flat]
    window_label_ids = None if window_labels_clean is None else [label2id[label.split(".")[0]] for label in window_labels_clean if label.split(".")[0] in label2id]

    data_prepared = dict(data_prepared)
    if file_names is not None:
        data_prepared["file_names"] = file_names
    if window_label_ids is not None:
        data_prepared["icd_codes"] = window_label_ids
    else:
        data_prepared.pop("icd_codes", None)

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    data_dataset = ICD10Dataset(data_prepared)
    data_loader = DataLoader(
        data_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=icd10_collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    return data_dataset, data_loader, label2id, id2label

def run_icd_classifier(model: ICD10Predictor_HS_Head | ICD10Predictor_NO_HS, data_loader: DataLoader, device: torch.device, id2label: dict, hs: bool, train: bool = False, dev_data_loader: DataLoader | None = None, optimizer: torch.optim.Optimizer | None = None, criterion: torch.nn.Module | ICD10Predictor_HS_CrossEntropyLoss | None = None, epochs: int = 1, ignore_index: int = -100, patience: int = 10, return_predictions: bool = True, print_full_report_level: int = 0):
    """
    Run the ICD classifier completely

    Parameters
    ----------
        `model`: ICD10Predictor_HS_Head | ICD10Predictor_NO_HS
            - Model used to perform the requested inference or initialization.
        `data_loader`: DataLoader
            - DataLoader providing model examples.
        `device`: torch.device
            - Device used for tensor computation.
        `id2label`: dict
            - Mapping from numeric identifiers to labels.
        `hs`: bool
            - Whether hierarchical softmax is used.
        `train`: bool
            - Whether to run training instead of evaluation.
        `dev_data_loader`: DataLoader | None
            - Optional DataLoader used for validation.
        `optimizer`: torch.optim.Optimizer | None
            - Optimizer used during training.
        `criterion`: torch.nn.Module | ICD10Predictor_HS_CrossEntropyLoss | None
            - Loss function used during training.
        `epochs`: int
            - Number of passes over the training data.
        `ignore_index`: int
            - Label ID ignored by the loss function.
        `patience`: int
            - Number of epochs allowed without validation improvement.
        `return_predictions`: bool
            - Whether to return predictions in addition to metrics.
        `print_full_report_level`: int
            - Verbosity level for the classification report.

    Returns
    -------
        `Any`
            - Processed output for downstream pipeline steps.
    """
    model.to(device)
    if hs:
        if criterion is None:
            raise ValueError("criterion must be an ICD10Predictor_HS_CrossEntropyLoss when hs=True")
        criterion.to(device)

    loss_values = []
    development_loss_values = []
    development_f1_values = []
    development_f1_micro_values = []
    epoch_results = []

    all_pred = None
    all_value_preds = None
    all_desc_global = None
    eval_loss = None

    best_micro_f1 = -float("inf")
    best_epoch = None
    best_model_state = None
    best_prediction_state = None
    epochs_without_improvement = 0

    if train:

        for epoch in tqdm(range(epochs), desc=f"ICD10 {'HS' if hs else 'NO HS'} / Training", total=epochs, unit="epoch"):
            # ---- TRAIN ----
            model.train()
            if hs:
                criterion.train()

            epoch_losses = []
            train_y_true_all = []
            train_y_pred_all = []

            for batch in tqdm(data_loader, desc=f"ICD10 {'HS' if hs else 'NO HS'} / Training epoch {epoch + 1} / Train set", unit="batch", total=len(data_loader), leave=False):
                diagnoses, icd_code = batch

                if torch.is_tensor(diagnoses):
                    dtype = next(model.parameters()).dtype
                    diagnoses = diagnoses.to(device, dtype=dtype, non_blocking=True)

                labels = None
                if torch.is_tensor(icd_code):
                    labels = icd_code.to(device, non_blocking=True)
                    labels = labels.reshape(-1).long()

                if labels is None:
                    raise ValueError("Training requires ICD target labels as the second DataLoader item")

                optimizer.zero_grad(set_to_none=True)

                if hs:
                    outputs = model(diagnoses)
                    loss = criterion(outputs, inference=False, targets=labels)
                else:
                    outputs = model(diagnoses, False, labels)
                    loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if hs:
                    torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)

                optimizer.step()
                epoch_losses.append(loss.item())

                if hs:
                    preds, path_results = criterion.predict(outputs)
                    preds = tensor_to_python(preds)
                    if not isinstance(preds, list):
                        preds = [preds]
                else:
                    preds_tensor = torch.argmax(outputs, dim=1)
                    preds = preds_tensor.detach().cpu().tolist()

                labels_cpu = labels.detach().cpu()
                valid_mask = labels_cpu != ignore_index
                valid_labels = labels_cpu[valid_mask].tolist()
                valid_preds = [pred for pred, valid in zip(preds, valid_mask.tolist()) if valid]

                for label_id in valid_labels:
                    label_id = int(label_id)
                    if label_id in id2label:
                        train_y_true_all.append(id2label[label_id])
                    else:
                        train_y_true_all.append(id2label[str(label_id)])

                for label_id in valid_preds:
                    label_id = int(label_id)
                    if label_id in id2label:
                        train_y_pred_all.append(id2label[label_id])
                    else:
                        train_y_pred_all.append(id2label[str(label_id)])

                del outputs, loss, diagnoses, labels

            train_loss_epoch = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0
            loss_values.append(train_loss_epoch)

            if len(train_y_true_all) > 0:
                train_precision_macro, train_precision_micro, train_recall_macro, train_recall_micro, train_macro_f1, train_micro_f1 = classification_report_icd_flatten(y_true=train_y_true_all, y_pred=train_y_pred_all, print_full_report_level=print_full_report_level)
            else:
                train_precision_macro, train_precision_micro, train_recall_macro, train_recall_micro, train_macro_f1, train_micro_f1 = None, None, None, None, None, None

            if dev_data_loader is not None:
                # ---- DEV / VALIDATION ----
                model.eval()
                if hs:
                    criterion.eval()

                dev_losses = []
                dev_pred_batches = []
                dev_value_pred_batches = []
                dev_diagnoses_text_global = []

                dev_y_true_all = []
                dev_y_pred_all = []

                with torch.inference_mode():
                    for dev_batch in tqdm(dev_data_loader, desc=f"ICD10 {'HS' if hs else 'NO HS'} / Training epoch {epoch + 1} / Dev set", unit="batch", total=len(dev_data_loader), leave=False):
                        diagnoses, icd_code = dev_batch

                        if torch.is_tensor(diagnoses):
                            dtype = next(model.parameters()).dtype
                            diagnoses = diagnoses.to(device, dtype=dtype, non_blocking=True)

                        labels = None
                        if torch.is_tensor(icd_code):
                            labels = icd_code.to(device, non_blocking=True)
                            labels = labels.reshape(-1).long()

                        if hs:
                            outputs = model(diagnoses)
                        else:
                            outputs = model(diagnoses, True, labels)

                        if labels is not None:
                            if hs:
                                dev_loss = criterion(outputs, inference=False, targets=labels)
                            else:
                                dev_loss = criterion(outputs, labels)
                            dev_losses.append(dev_loss.item())

                        if hs:
                            preds, path_results = criterion.predict(outputs)
                            preds = tensor_to_python(preds)
                            if not isinstance(preds, list):
                                preds = [preds]
                            path_results = tensor_to_python(path_results)

                            if return_predictions:
                                dev_pred_batches.append(preds)
                                dev_value_pred_batches.append((None, path_results))
                        else:
                            preds_tensor = torch.argmax(outputs, dim=1)
                            max_vals, argmax_vals = torch.max(outputs, dim=1)
                            preds = preds_tensor.detach().cpu().tolist()

                            if return_predictions:
                                dev_pred_batches.append(preds_tensor.detach().cpu().numpy())
                                dev_value_pred_batches.append((None, [[(max_vals[i].item(), argmax_vals[i].item())] for i in range(outputs.size(0))]))

                        if labels is not None:
                            labels_cpu = labels.detach().cpu()
                            valid_mask = labels_cpu != ignore_index
                            valid_labels = labels_cpu[valid_mask].tolist()
                            valid_preds = [pred for pred, valid in zip(preds, valid_mask.tolist()) if valid]

                            for label_id in valid_labels:
                                label_id = int(label_id)
                                if label_id in id2label:
                                    dev_y_true_all.append(id2label[label_id])
                                else:
                                    dev_y_true_all.append(id2label[str(label_id)])

                            for label_id in valid_preds:
                                label_id = int(label_id)
                                if label_id in id2label:
                                    dev_y_pred_all.append(id2label[label_id])
                                else:
                                    dev_y_pred_all.append(id2label[str(label_id)])

                        if not torch.is_tensor(diagnoses):
                            dev_diagnoses_text_global.extend(diagnoses)

                        del outputs, diagnoses, labels

                dev_loss_epoch = float(np.mean(dev_losses)) if len(dev_losses) > 0 else None
                development_loss_values.append(dev_loss_epoch)

                if len(dev_y_true_all) > 0:
                    dev_precision_macro, dev_precision_micro, dev_recall_macro, dev_recall_micro, dev_macro_f1, dev_micro_f1 = classification_report_icd_flatten(y_true=dev_y_true_all, y_pred=dev_y_pred_all, print_full_report_level=print_full_report_level)
                else:
                    dev_precision_macro, dev_precision_micro, dev_recall_macro, dev_recall_micro, dev_macro_f1, dev_micro_f1 = None, None, None, None, None, None

                development_f1_values.append(dev_macro_f1)
                development_f1_micro_values.append(dev_micro_f1)

                if return_predictions and len(dev_pred_batches) > 0:
                    if hs:
                        all_pred = [pred for batch_preds in dev_pred_batches for pred in batch_preds]
                    else:
                        all_pred = np.concatenate(dev_pred_batches, axis=0)

                    all_value_preds = tensor_to_python(dev_value_pred_batches)
                    all_desc_global = dev_diagnoses_text_global

                # ---- EARLY STOPPING ----
                if dev_micro_f1 is not None and dev_micro_f1 > best_micro_f1:
                    best_micro_f1 = dev_micro_f1
                    best_epoch = epoch + 1
                    best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    if hs:
                        best_prediction_state = {key: value.detach().cpu().clone() for key, value in criterion.state_dict().items()}
                    else:
                        best_prediction_state = None
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                epoch_results.append({"epoch": epoch + 1, "train_loss": train_loss_epoch, "development_loss": dev_loss_epoch, "f1_macro": dev_macro_f1, "f1_micro": dev_micro_f1, "best_micro_f1": best_micro_f1})

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best epoch: {best_epoch} | Best dev F1 (micro): {best_micro_f1:.6f}")
                    break

                dev_loss_text = f"{dev_loss_epoch:.6f}" if dev_loss_epoch is not None else "n/a"
                macro_precision_text = f"{dev_precision_macro:.4f}" if dev_precision_macro is not None else "n/a"
                micro_precision_text = f"{dev_precision_micro:.4f}" if dev_precision_micro is not None else "n/a"
                macro_recall_text = f"{dev_recall_macro:.4f}" if dev_recall_macro is not None else "n/a"
                micro_recall_text = f"{dev_recall_micro:.4f}" if dev_recall_micro is not None else "n/a"
                macro_f1_text = f"{dev_macro_f1:.4f}" if dev_macro_f1 is not None else "n/a"
                micro_f1_text = f"{dev_micro_f1:.4f}" if dev_micro_f1 is not None else "n/a"
                best_f1_text = f"{best_micro_f1:.4f}" if best_micro_f1 != -float("inf") else "n/a"

                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f} | dev loss: {dev_loss_text} | Precision (macro): {macro_precision_text} | Precision (micro): {micro_precision_text} | Recall (macro): {macro_recall_text} | Recall (micro): {micro_recall_text} | F1 (macro): {macro_f1_text} | F1 (micro): {micro_f1_text} | Best F1 (micro): {best_f1_text}")

            else:
                epoch_results.append({"epoch": epoch + 1, "train_loss": train_loss_epoch, "development_loss": None, "f1_macro": train_macro_f1, "f1_micro": train_micro_f1, "best_micro_f1": best_micro_f1})
                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f}")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

            if hs and best_prediction_state is not None:
                criterion.load_state_dict(best_prediction_state)
                criterion.to(device)
        else:
            best_epoch = len(loss_values)
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if hs:
                best_prediction_state = {key: value.detach().cpu().clone() for key, value in criterion.state_dict().items()}
            else:
                best_prediction_state = None

    else:
        # ---- INFERENCE / EVALUATION ----
        model.eval()
        if hs and criterion is not None:
            criterion.eval()

        eval_losses = []
        pred_batches = []
        value_pred_batches = []
        diagnoses_text_global = []

        y_true_all = []
        y_pred_all = []

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc=f"ICD10 {'HS' if hs else 'NO HS'} / Inference", total=len(data_loader), unit="batch"):
                diagnoses, icd_code = batch

                if torch.is_tensor(diagnoses):
                    dtype = next(model.parameters()).dtype
                    diagnoses = diagnoses.to(device, dtype=dtype, non_blocking=True)

                labels = None
                if torch.is_tensor(icd_code):
                    labels = icd_code.to(device, non_blocking=True)
                    labels = labels.reshape(-1).long()

                if hs:
                    outputs = model(diagnoses)
                else:
                    outputs = model(diagnoses, True, labels)

                if criterion is not None and labels is not None:
                    if hs:
                        loss = criterion(outputs, inference=False, targets=labels)
                    else:
                        loss = criterion(outputs, labels)
                    eval_losses.append(loss.item())

                if hs:
                    preds, path_results = criterion.predict(outputs)
                    preds = tensor_to_python(preds)
                    if not isinstance(preds, list):
                        preds = [preds]
                    path_results = tensor_to_python(path_results)

                    if return_predictions:
                        pred_batches.append(preds)
                        value_pred_batches.append((None, path_results))
                else:
                    preds_tensor = torch.argmax(outputs, dim=1)
                    max_vals, argmax_vals = torch.max(outputs, dim=1)
                    preds = preds_tensor.detach().cpu().tolist()

                    if return_predictions:
                        pred_batches.append(preds_tensor.detach().cpu().numpy())
                        value_pred_batches.append((None, [[(max_vals[i].item(), argmax_vals[i].item())] for i in range(outputs.size(0))]))

                if labels is not None:
                    labels_cpu = labels.detach().cpu()
                    valid_mask = labels_cpu != ignore_index
                    valid_labels = labels_cpu[valid_mask].tolist()
                    valid_preds = [pred for pred, valid in zip(preds, valid_mask.tolist()) if valid]

                    for label_id in valid_labels:
                        label_id = int(label_id)
                        if label_id in id2label:
                            y_true_all.append(id2label[label_id])
                        else:
                            y_true_all.append(id2label[str(label_id)])

                    for label_id in valid_preds:
                        label_id = int(label_id)
                        if label_id in id2label:
                            y_pred_all.append(id2label[label_id])
                        else:
                            y_pred_all.append(id2label[str(label_id)])

                if not torch.is_tensor(diagnoses):
                    diagnoses_text_global.extend(diagnoses)

                del outputs, diagnoses, labels

        if len(eval_losses) > 0:
            eval_loss = float(np.mean(eval_losses))

        if len(y_true_all) > 0:
            precision_macro, precision_micro, recall_macro, recall_micro, macro_f1, micro_f1 = classification_report_icd_flatten(y_true=y_true_all, y_pred=y_pred_all, print_full_report_level=print_full_report_level)
            print(f"Precision (macro): {precision_macro:.4f} | Precision (micro): {precision_micro:.4f} | Recall (macro): {recall_macro:.4f} | Recall (micro): {recall_micro:.4f} | F1 (macro): {macro_f1:.4f} | F1 (micro): {micro_f1:.4f}")
        else:
            precision_macro, precision_micro, recall_macro, recall_micro, macro_f1, micro_f1 = None, None, None, None, None, None

        if return_predictions and len(pred_batches) > 0:
            if hs:
                all_pred = [pred for batch_preds in pred_batches for pred in batch_preds]
            else:
                all_pred = np.concatenate(pred_batches, axis=0)

            all_value_preds = tensor_to_python(value_pred_batches)
            all_desc_global = diagnoses_text_global
        elif return_predictions:
            all_pred = []
            all_value_preds = []
            all_desc_global = diagnoses_text_global

    return {"all_pred": all_pred, "all_value_preds": all_value_preds, "all_desc_global": all_desc_global, "loss_values": loss_values, "development_loss_values": development_loss_values, "development_f1_values": development_f1_values, "development_f1_micro_values": development_f1_micro_values, "epoch_results": epoch_results, "eval_loss": eval_loss, "best_micro_f1": best_micro_f1, "best_epoch": best_epoch, "best_model_state": best_model_state, "best_prediction_state": best_prediction_state}

def apply_thresholds_icd10(flattened_predictions: list, thresholds: list):
    """
    Aplica una lista de umbrales a secuencias de predicciones para determinar en qué paso debe detenerse cada secuencia.

    Parameters
    ----------
        `flattened_predictions`: list
            - Lista de secuencias de predicciones. Cada secuencia contiene los valores obtenidos en distintos pasos o niveles

        `thresholds`: list
            - Lista de umbrales que se comparan con los valores de cada secuencia. Cada posición representa el umbral correspondiente a ese paso

    Returns
    -------
        `Any`
            - Processed output for downstream pipeline steps.
    """
    result = []

    for seq in flattened_predictions:
        if len(seq) == 0:
            result.append((False, None, None))
            continue

        max_steps = min(len(seq), len(thresholds))

        stop_step = None
        stop_value = None
        reached_end = True

        for step in range(max_steps):
            value = seq[step]

            if value < thresholds[step]:
                stop_step = step
                stop_value = value
                reached_end = False
                break

        if stop_step is None:
            stop_step = max_steps
            stop_value = seq[max_steps - 1]

        result.append((reached_end, stop_step, stop_value))

    return result

def create_mixed_icd10_threshold_results(results_hs: dict, results_nohs: dict, thresholds_hs: list, thresholds_nohs: list, id2label_hs: dict, id_no_hs_to_id_hs: dict):
    """
    Combines hierarchical and non-hierarchical predictions using their thresholds.

    Parameters
    ----------
        `results_hs`: dict
            - Model output or predictions to transform.
        `results_nohs`: dict
            - Model output or predictions to transform.
        `thresholds_hs`: list
            - Similarity or decision threshold applied by the function.
        `thresholds_nohs`: list
            - Similarity or decision threshold applied by the function.
        `id2label_hs`: dict
            - Entity or relation labels used by the model.
        `id_no_hs_to_id_hs`: dict
            - Mapping from non-hierarchical IDs to hierarchical IDs.

    Returns
    -------
        `Any`
            - Constructed data ready for the next pipeline step.
    """
    pred_hs = as_prediction_id_list(results_hs.get("all_pred"))
    pred_nohs = as_prediction_id_list(results_nohs.get("all_pred"))
    pred_nohs_as_hs = [id_no_hs_to_id_hs[pred_id] for pred_id in pred_nohs]

    value_preds_hs = results_hs.get("all_value_preds")
    value_preds_nohs_as_hs = remap_icd_prediction_ids(results_nohs.get("all_value_preds"), id_no_hs_to_id_hs)

    desc_hs = results_hs.get("all_desc_global") or []
    desc_nohs = results_nohs.get("all_desc_global") or []
    if desc_hs and desc_nohs and desc_hs != desc_nohs:
        raise ValueError("ICD10 HS and NO-HS descriptions are different. Build both loaders with shuffle=False.")

    diagnoses = desc_hs or desc_nohs
    if not (len(pred_hs) == len(pred_nohs_as_hs) == len(diagnoses)):
        raise ValueError(f"ICD10 HS/NO-HS prediction lengths do not match: HS={len(pred_hs)}, NO-HS={len(pred_nohs_as_hs)}, diagnoses={len(diagnoses)}")

    flattened_hs = extract_flattened_predictions(value_preds_hs, pred_hs)
    flattened_nohs = extract_flattened_predictions(value_preds_nohs_as_hs, pred_nohs_as_hs)

    threshold_results_hs = apply_thresholds_icd10(flattened_hs, thresholds_hs)
    threshold_results_nohs = apply_thresholds_icd10(flattened_nohs, thresholds_nohs)

    mixed_items = []
    mixed_pred_ids = []
    mixed_sources = []

    for idx, (hs_threshold, nohs_threshold) in enumerate(zip(threshold_results_hs, threshold_results_nohs)):
        hs_reached_end = bool(hs_threshold[0])
        nohs_reached_end = bool(nohs_threshold[0])

        if hs_reached_end:
            selected_pred_id = pred_hs[idx]
            selected_source = "HS"
            selected_threshold = hs_threshold
        elif nohs_reached_end:
            selected_pred_id = pred_nohs_as_hs[idx]
            selected_source = "NO_HS"
            selected_threshold = nohs_threshold
        else:
            selected_pred_id = pred_hs[idx]
            selected_source = "HS_FALLBACK"
            selected_threshold = hs_threshold

        selected_pred_id = int(selected_pred_id)
        mixed_pred_ids.append(selected_pred_id)
        mixed_sources.append(selected_source)
        mixed_items.append((
            diagnoses[idx],
            id2label_hs[selected_pred_id],
            {
                "selected_source": selected_source,
                "selected_threshold": selected_threshold,
                "hs_threshold": hs_threshold,
                "nohs_threshold": nohs_threshold,
            },
        ))

    return {"items": mixed_items, "all_pred": mixed_pred_ids, "all_desc_global": diagnoses, "sources": mixed_sources, "threshold_results_hs": threshold_results_hs, "threshold_results_nohs": threshold_results_nohs, "flattened_predictions_hs": flattened_hs, "flattened_predictions_nohs": flattened_nohs}

def add_standalone_icd_metadata(results, id2label, source, node_list):
    """
    Adds ICD-10 code and hierarchy metadata to standalone predictions.

    Parameters
    ----------
        `results`: Any
            - Model outputs to convert or enrich.
        `id2label`: Any
            - Mapping from numeric identifiers to labels.
        `source`: Any
            - Source identifier attached to the predictions.
        `node_list`: Any
            - ICD hierarchy nodes used to enrich predictions.

    Returns
    -------
        `Any`
            - Processed output for downstream pipeline steps.
    """
    predictions = tensor_to_python(results.get("all_pred")) or []

    results["sources"] = [source for _ in predictions]
    results["items"] = []

    for prediction in predictions:
        code = label_from_id(id2label, prediction)

        if source == "HS" and code != "O":
            step = max(len(node_list[code].path) - 1, 0)
        else:
            step = 0

        results["items"].append(
            (
                None,
                code,
                {
                    "selected_threshold": (True, step, None),
                },
            )
        )

    return results
