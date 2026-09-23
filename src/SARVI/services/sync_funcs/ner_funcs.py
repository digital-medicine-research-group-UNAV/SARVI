from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Any, TYPE_CHECKING
from functools import partial

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from seqeval.metrics import (
    f1_score as f1_score_sequeval,
    classification_report as classification_report_sequeval
)
from sklearn.metrics import (
    f1_score as f1_score_sklearn,
    classification_report as classification_report_sklearn
)

from ..common.ner_funcs import (
    series_to_striding_ner_windows, ner_collate_fn, span_labels_present_in_window_labels,
    NER_WINDOW_LABELS
)
from ..common.utils.ner_utils import (
    normalize_list, get_final_entity_components_from_group, normalize_bio_sequence, average_overlapping_hidden_states
)

from ..common.utils.nn_utils import (
    move_batch_to_device, detach_output
)

from ...models.datasets import (
    NERDataset
)

if TYPE_CHECKING:
    from ...models.neural_networks import Span_NERClassifier, BIO_NERClassifier

def prepare_data(data: pd.DataFrame, padding: bool, tokenizer: Any, model: Any, device: torch.device, data_ann: pd.DataFrame | None = None, window_tokens_no_special: int = 510, stride: int = 128, strict: bool = False, lemma: bool = False):
    """
    Prepara el conjunto de datos a utilizar de manera determinista para la extracción de entidades. 
    Genera ventanas con stride, calculando los embeddings de cada ventana con el modelo y fusionando los estados ocultos solapados.

    Parameters
    ----------
        `data`: pd.DataFrame
            - DataFrame con los datos de entrada. Debe contar unicamente con las columnas `archivo_origen` y `Text`

        `padding`: bool
            - Indica si se debe aplicar padding a las ventanas generadas por el tokenizer

        `tokenizer`: Any
            - Tokenizer utilizado para transformar el texto en tokens y para generar las ventanas de entrada del modelo

        `model`: Any
            - Modelo utilizado para calcular los estados ocultos de cada ventana

        `device`: torch.device
            - Dispositivo donde se ejecutará el modelo, por ejemplo **cpu** o **cuda**

        `window_tokens_no_special`: int
            - Número máximo de tokens por ventana sin contar tokens especiales. Por defecto es **510** *(512-2 tokens especiales)*

        `stride`: int
            - Desplazamiento entre ventanas consecutivas, medido en tokens del texto completo. Por defecto es **128**

        `strict`: bool
            - Si es `True`, aplica comprobaciones estrictas al fusionar los estados ocultos solapados. Por defecto es **False**
        
        `lemma`: bool
            - Si se necesita hacer una lemmatización del texto o no. Por defecto es **False**

    Returns
    -------
        ``: list
            - Lista con las ventanas preparadas para cada fila del DataFrame. Cada ventana incluye su embedding calculado y el nombre del archivo de origen.
    """
    data_prepared = []
    all_window_labels = [] if data_ann is not None else None
    file_names = []

    ##############################
    ## Compatibility inputs kept for the existing notebooks. The previous embedding precomputation that used model/device/strict is commented below.
    # model = model
    # device = device
    # strict = strict
    ##############################

    ann_by_file = None
    if data_ann is not None and "archivo_origen" in data_ann.columns:
        ann_by_file = {row["archivo_origen"]: row for _, row in data_ann.iterrows()}

    for row_pos, (_, text_info) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Preparing data for NER prediction: Window slicing and overlapping tokens", unit="text")):
        if ann_by_file is not None:
            ann_info = ann_by_file.get(text_info["archivo_origen"])
        elif data_ann is not None:
            ann_info = data_ann.iloc[row_pos]
        else:
            ann_info = None

        if data_ann is not None and ann_info is None:
            raise ValueError(f"No annotation row found for file {text_info['archivo_origen']}")

        windows_result = series_to_striding_ner_windows(text_info, ann_info, tokenizer=tokenizer, window_tokens=window_tokens_no_special, stride=stride, padding=padding, lemma=lemma)
        if ann_info is not None:
            windows, window_labels = windows_result
            all_window_labels.append(window_labels)
        else:
            windows = windows_result

        ##############################
        # last_hidden_state_list = []
        # for w in windows:
        #     inputs = {
        #         "input_ids": torch.tensor([w["input_ids"]], device=device),
        #         "attention_mask": torch.tensor([w["attention_mask"]], device=device),
        #     }
        #     with torch.no_grad():
        #         outputs = model(**inputs)

        #     last_hidden_states = outputs.last_hidden_state
        #     last_hidden_state_list.append(last_hidden_states)

        # merged, logs = average_overlapping_hidden_states(windows=windows, last_hidden_state_list=last_hidden_state_list, window_tokens_no_special=window_tokens_no_special, stride=stride, strict=strict)

        # for w,m in zip(windows, merged):
        #     w["embedding"] = m
        #     w["file_name"] = info["archivo_origen"]
        ##############################

        data_prepared.append(windows)
        file_names.append(text_info["archivo_origen"])

    return data_prepared, all_window_labels, file_names

def construct_loaders_ner(data: list, window_labels: list | None = None, file_names: list | None = None, ner_type: str = "", label2id: dict | None = None, id2label: dict | None = None, seed: int = 8, ignore_o_labels: bool = False, batch_size: int = 8):
    """
    Construye el data loader para poder realizar la predicción de entidades.

    Parameters
    ----------
        `data`: list
            - Lista con los datos de entrada que se usarán para construir el dataset

        `window_labels`: list
            - Lista donde cada elemento es otro conjunto de listas que indida que etiquetas tiene cada token

        `file_names`: list
            - Lista de elementos donde cada uno indica el nombre del archivo de donde provienen los datos
        
        `ner_type`: str
            - Tiene que ser "span" o "bio" para saber como tratar con las etiquetas y que id2label y label2id devolver

        `ignore_o_labels`: bool
            - Si es `True`, las etiquetas "O" se convierten a `-100` en el collator para ignorarlas en la loss y métricas enmascaradas.

    Returns
    -------
        `data_loader`: DataLoader
            - DataLoader que permite iterar sobre el dataset por lotes
    """
    if ner_type == "span":
        if label2id is None:
            labels = span_labels_present_in_window_labels(window_labels)
            if labels is None:
                labels = ["O"] + sorted(NER_WINDOW_LABELS)
            label2id = {label: idx for idx, label in enumerate(labels)}
            id2label = {idx: label for label, idx in label2id.items()}
        elif id2label is None:
            id2label = {idx: label for label, idx in label2id.items()}
        window_label_ids = window_labels
    elif ner_type == "bio":
        if window_labels is None:
            window_labels_clean = None
        else:
            window_labels_clean = [[normalize_bio_sequence(sublist) for sublist in window] for window in window_labels]

        if label2id is None and window_labels_clean is not None:
            labels = sorted({label for window in window_labels_clean for sublist in window for label in sublist})
            label2id = {label: idx for idx, label in enumerate(labels)}
            id2label = {idx: label for label, idx in label2id.items()}
        elif id2label is None:
            id2label = {idx: label for label, idx in label2id.items()}

        window_label_ids = None if window_labels_clean is None else [[[label2id[label] for label in sublist] for sublist in window] for window in window_labels_clean]
    else:
        raise KeyError("Must be `span` or `bio`")

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    data_dataset = NERDataset(data, window_label_ids, file_names)
    data_loader = DataLoader(
        data_dataset,
        batch_size=batch_size,
        collate_fn=partial(ner_collate_fn, ignore_o_labels=ignore_o_labels, o_label_id=(label2id or {}).get("O")),
        generator=generator,
        worker_init_fn=seed_worker,
    )


    return data_dataset, data_loader, label2id, id2label

def run_span_nerclassifier(model: "Span_NERClassifier", data_loader: DataLoader, device: torch.device, id2label: dict, train: bool = False, dev_data_loader: DataLoader | None = None, optimizer: torch.optim.Optimizer | None = None, criterion: torch.nn.Module | None = None, epochs: int = 1, label_key: str = "window_labels", ignore_index: int = -100, patience: int = 10, return_predictions: bool = True, ignore_o_labels: bool = False):
    """
    Ejecuta Span_NERClassifier en modo entrenamiento o inferencia.

    Parameters
    ----------
        model: Span_NERClassifier
            Modelo Span NER.

        data_loader: DataLoader
            DataLoader principal. Si train=True, será el train_loader.
            Si train=False, será el loader de inferencia/evaluación.

        device: torch.device
            Dispositivo de ejecución.
        
        id2label: dict
            Diccionario con la traducción de ids a labels

        train: bool
            Si True, entrena el modelo.
            Si False, solo hace inferencia/evaluación.

        dev_data_loader: DataLoader | None
            DataLoader de desarrollo/validación. Solo se usa si train=True.

        optimizer: torch.optim.Optimizer | None
            Optimizador. Obligatorio si train=True.

        criterion: torch.nn.Module | None
            Loss function. Por ejemplo MoMLoss.

        epochs: int
            Número de epochs de entrenamiento.

        label_key: str
            Clave del batch donde están las etiquetas BIO.
            Normalmente "window_labels" o "labels".

        ignore_index: int
            Índice ignorado en la loss, normalmente -100.

        patiente: int
            Cuantas epocas sin mejora se espera para parar el entrenamiento del modelo

        return_predictions: bool
            Si True, devuelve predicciones y probabilidades.

        ignore_o_labels: bool
            Si True, no incluye "O" en el classification report ni en el macro F1.

    Returns
    -------
        dict
            Diccionario con:
                - all_pred
                - all_value_preds
                - loss_values
                - development_loss_values
                - eval_loss
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
    model_outputs = []
    report_labels = [label for _, label in sorted(id2label.items(), key=lambda item: int(item[0])) if not (ignore_o_labels and label == "O")]

    def report_labels_present_in_gold(y_true: list[str]) -> list[str]:
        present_labels = set(y_true)
        return [label for label in report_labels if label in present_labels]

    if train:
        for epoch in tqdm(range(epochs), desc="Span NER / Training", total=epochs, unit="epoch"):
            # ---- TRAIN ----
            model.train()

            epoch_losses = []

            for batch in tqdm(data_loader, desc=f"Span NER / Training epoch {epoch + 1} / Train set", unit="batch", total=len(data_loader), leave=False):
                batch = move_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                output = model(batch=batch)
                logits = output["logits"]
                labels = output.get("y_true")
                weights = output.get("weights")

                loss = criterion(logits, labels, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                epoch_losses.append(loss.item())

            train_loss_epoch = float(np.mean(epoch_losses))
            loss_values.append(train_loss_epoch)

            if dev_data_loader is not None:
                model.eval()

                dev_losses = []
                dev_pred_batches = []
                dev_value_pred_batches = []

                dev_y_true_all = []
                dev_y_pred_all = []

                with torch.inference_mode():
                    for dev_batch in tqdm(dev_data_loader, desc=f"Span NER / Training epoch {epoch + 1} / Dev set", unit="batch", total=len(dev_data_loader), leave=False):
                        dev_batch = move_batch_to_device(dev_batch, device)

                        dev_output = model(batch=dev_batch)
                        model_outputs.append(detach_output(dev_output))

                        dev_logits = dev_output["logits"]

                        dev_value_preds = torch.softmax(dev_logits, dim=-1)
                        dev_preds = torch.argmax(dev_value_preds, dim=-1)
                        dev_labels = dev_output.get("y_true")
                        dev_weights = dev_output.get("weights")

                        if dev_labels is not None:
                            dev_loss = criterion(dev_logits, dev_labels, dev_weights)
                            dev_losses.append(dev_loss.item())

                            dev_mask = dev_labels != ignore_index

                            dev_preds_cpu = dev_preds.detach().cpu()
                            dev_labels_cpu = dev_labels.detach().cpu()
                            dev_mask_cpu = dev_mask.detach().cpu()

                            for pred_id, true_id in zip(dev_preds_cpu[dev_mask_cpu], dev_labels_cpu[dev_mask_cpu]):
                                true_id = int(true_id.item())
                                pred_id = int(pred_id.item())

                                dev_y_true_all.append(id2label[true_id])
                                dev_y_pred_all.append(id2label[pred_id])

                            if return_predictions:
                                dev_pred_batches.append(dev_preds[dev_mask].detach().cpu().numpy())
                                dev_value_pred_batches.append(dev_value_preds[dev_mask].detach().cpu().numpy())

                dev_loss_epoch = float(np.mean(dev_losses))
                development_loss_values.append(dev_loss_epoch)

                dev_report_labels = report_labels_present_in_gold(dev_y_true_all)
                if len(dev_report_labels) > 0:
                    dev_macro_f1 = f1_score_sklearn(dev_y_true_all, dev_y_pred_all, labels=dev_report_labels, average="macro", zero_division=0)
                    dev_report = classification_report_sklearn(dev_y_true_all, dev_y_pred_all, labels=dev_report_labels, digits=4, zero_division=0)
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

                # ---- EARLY STOPPING ----
                if dev_macro_f1 is not None and dev_macro_f1 > best_macro_f1:
                    best_macro_f1 = dev_macro_f1
                    best_epoch = epoch + 1
                    best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best epoch: {best_epoch} | Best dev macro F1: {best_macro_f1:.6f}")
                    break

                best_f1_text = f"{best_macro_f1:.4f}" if best_macro_f1 != -float("inf") else "n/a"
                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f} | dev loss: {dev_loss_epoch:.6f} | Best F1 (macro): {best_f1_text}")

            else:
                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f}")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)
    else:
        # ---- INFERENCE / EVALUATION ----
        model.eval()

        eval_losses = []
        pred_batches = []
        value_pred_batches = []

        y_true_all = []
        y_pred_all = []

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="Span NER / Inference", total=len(data_loader), unit="batch"):
                batch = move_batch_to_device(batch, device)

                output = model(batch=batch)
                logits = output["logits"]

                value_preds = torch.softmax(logits, dim=-1)
                preds = torch.argmax(value_preds, dim=-1)
                pred_scores = torch.max(value_preds, dim=-1).values
                labels = output.get("y_true")
                weights = output.get("weights")
                output = {**output, "pred_id": preds, "pred_label": [id2label[int(pred_id)] for pred_id in preds.detach().cpu().tolist()], "pred_score": pred_scores, "value_preds": value_preds}
                model_outputs.append(detach_output(output))

                if labels is not None:
                    if criterion is not None:
                        loss = criterion(logits, labels, weights)
                        eval_losses.append(loss.item())

                    mask = labels != ignore_index

                    preds_cpu = preds.detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    mask_cpu = mask.detach().cpu()

                    for pred_id, true_id in zip(preds_cpu[mask_cpu], labels_cpu[mask_cpu]):
                        true_id = int(true_id.item())
                        pred_id = int(pred_id.item())

                        y_true_all.append(id2label[true_id])
                        y_pred_all.append(id2label[pred_id])

                    if return_predictions:
                        pred_batches.append(preds[mask].detach().cpu().numpy())
                        value_pred_batches.append(value_preds[mask].detach().cpu().numpy())
                else:
                    if return_predictions:
                        pred_batches.append(preds.detach().cpu().numpy())
                        value_pred_batches.append(value_preds.detach().cpu().numpy())

        if len(eval_losses) > 0:
            eval_loss = float(np.mean(eval_losses))

        eval_report_labels = report_labels_present_in_gold(y_true_all)
        if len(eval_report_labels) > 0:
            best_macro_f1 = f1_score_sklearn(y_true_all, y_pred_all, labels=eval_report_labels, average="macro", zero_division=0)
            final_classification_report = classification_report_sklearn(y_true_all, y_pred_all, labels=eval_report_labels, digits=4, zero_division=0)

            print("\nClassification report:")
            print(final_classification_report)
        else:
            best_macro_f1 = None
            final_classification_report = None

        if return_predictions and len(pred_batches) > 0:
            all_pred = np.concatenate(pred_batches, axis=0)
            all_value_preds = np.concatenate(value_pred_batches, axis=0)

    return {"all_pred": all_pred, "all_value_preds": all_value_preds, "model_outputs": model_outputs, "loss_values": loss_values, "development_loss_values": development_loss_values, "development_f1_values": development_f1_values, "eval_loss": eval_loss, "classification_report": final_classification_report, "best_macro_f1": best_macro_f1, "best_epoch": best_epoch, "best_model_state": best_model_state}


def run_bio_nerclassifier(model: "BIO_NERClassifier", data_loader: DataLoader, device: torch.device, id2label: dict, train: bool = False, dev_data_loader: DataLoader | None = None, optimizer: torch.optim.Optimizer | None = None, criterion: torch.nn.Module | None = None, epochs: int = 1, label_key: str = "window_labels", ignore_index: int = -100, patience: int = 10, return_predictions: bool = True, return_probabilities: bool = False):
    """
    Ejecuta BIO_NERClassifier en modo entrenamiento o inferencia.

    Parameters
    ----------
        model: BIO_NERClassifier
            Modelo BIO NER.

        data_loader: DataLoader
            DataLoader principal. Si train=True, será el train_loader.
            Si train=False, será el loader de inferencia/evaluación.

        device: torch.device
            Dispositivo de ejecución.
        
        id2label: dict
            Diccionario con la traducción de ids a labels

        train: bool
            Si True, entrena el modelo.
            Si False, solo hace inferencia/evaluación.

        dev_data_loader: DataLoader | None
            DataLoader de desarrollo/validación. Solo se usa si train=True.

        optimizer: torch.optim.Optimizer | None
            Optimizador. Obligatorio si train=True.

        criterion: torch.nn.Module | None
            Loss function. Por ejemplo MoMLoss.

        epochs: int
            Número de epochs de entrenamiento.

        label_key: str
            Clave del batch donde están las etiquetas BIO.
            Normalmente "window_labels" o "labels".

        ignore_index: int
            Índice ignorado en la loss, normalmente -100.

        patiente: int
            Cuantas epocas sin mejora se espera para parar el entrenamiento del modelo

        return_predictions: bool
            Si True, devuelve predicciones y probabilidades.

        return_probabilities: bool
            Si True, devuelve probabilidades por clase. Consume bastante más memoria
            porque materializa un tensor softmax completo.

    Returns
    -------
        dict
            Diccionario con:
                - all_pred
                - all_value_preds
                - loss_values
                - development_loss_values
                - eval_loss
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

    def valid_token_mask(batch: dict, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            if "attention_mask" in batch:
                mask = batch["attention_mask"].bool()
            else:
                logits_shape = batch["input_ids"].shape
                mask = torch.ones(logits_shape, dtype=torch.bool, device=device)
        else:
            mask = labels != ignore_index

        if "attention_mask" in batch:
            mask = mask & batch["attention_mask"].bool()
        if "global_token_indices" in batch:
            mask = mask & (batch["global_token_indices"] >= 0)

        return mask

    if train:
        if optimizer is None or criterion is None:
            raise ValueError("optimizer and criterion are required when train=True")

        for epoch in tqdm(range(epochs), desc="BIO NER / Training", total=epochs, unit="epoch"):
            # ---- TRAIN ----
            model.train()

            epoch_losses = []

            for batch in tqdm(data_loader, desc=f"BIO NER / Training epoch {epoch + 1} / Train set", unit="batch", total=len(data_loader), leave=False):
                batch = move_batch_to_device(batch, device)
                labels = batch.get(label_key)
                if labels is None:
                    raise ValueError(f"Training requires `{label_key}` in the batch")

                optimizer.zero_grad(set_to_none=True)

                output = model(batch=batch)
                logits = output["logits"]

                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                epoch_losses.append(loss.item())
                del output, logits, loss, batch, labels

            train_loss_epoch = float(np.mean(epoch_losses))
            loss_values.append(train_loss_epoch)


            if dev_data_loader is not None:
                model.eval()

                dev_losses = []
                dev_pred_batches = []
                dev_value_pred_batches = []

                dev_y_true_all = []
                dev_y_pred_all = []

                with torch.inference_mode():
                    for dev_batch in tqdm(dev_data_loader, desc=f"BIO NER / Training epoch {epoch + 1} / Dev set", unit="batch", total=len(dev_data_loader), leave=False):
                        dev_batch = move_batch_to_device(dev_batch, device)

                        dev_output = model(batch=dev_batch)
                        dev_logits = dev_output["logits"]

                        dev_preds = torch.argmax(dev_logits, dim=-1)

                        dev_labels = dev_batch.get(label_key)
                        if dev_labels is not None:
                            dev_loss = criterion(dev_logits, dev_labels)
                            dev_losses.append(dev_loss.item())

                            dev_mask = valid_token_mask(dev_batch, dev_labels)

                            dev_preds_cpu = dev_preds.detach().cpu()
                            dev_labels_cpu = dev_labels.detach().cpu()
                            dev_mask_cpu = dev_mask.detach().cpu()
                            batch_size = dev_preds_cpu.shape[0]
                            for i in range(batch_size):
                                true_seq = []
                                pred_seq = []

                                valid_positions = dev_mask_cpu[i].bool()

                                for pred_id, true_id in zip(dev_preds_cpu[i][valid_positions], dev_labels_cpu[i][valid_positions]):
                                    true_id = int(true_id.item())
                                    pred_id = int(pred_id.item())

                                    if true_id == ignore_index:
                                        continue

                                    true_seq.append(id2label[true_id])
                                    pred_seq.append(id2label[pred_id])

                                if len(true_seq) > 0:
                                    dev_y_true_all.append(true_seq)
                                    dev_y_pred_all.append(pred_seq)

                            if return_predictions:
                                dev_prediction_mask = valid_token_mask(dev_batch)
                                dev_pred_batches.append(dev_preds[dev_prediction_mask].detach().cpu().numpy())
                                if return_probabilities:
                                    dev_value_preds = torch.softmax(dev_logits, dim=-1)
                                    dev_value_pred_batches.append(dev_value_preds[dev_prediction_mask].detach().cpu().numpy())

                        del dev_output, dev_logits, dev_preds, dev_batch

                dev_loss_epoch = float(np.mean(dev_losses)) if len(dev_losses) > 0 else None
                development_loss_values.append(dev_loss_epoch)

                if len(dev_y_true_all) > 0:
                    dev_macro_f1 = f1_score_sequeval(dev_y_true_all, dev_y_pred_all, average="macro", zero_division=0)
                    dev_report = classification_report_sequeval(dev_y_true_all, dev_y_pred_all, digits=4, zero_division=0)
                else:
                    dev_macro_f1 = None
                    dev_report = None

                development_f1_values.append(dev_macro_f1)
                final_classification_report = dev_report

                if return_predictions and len(dev_pred_batches) > 0:
                    all_pred = np.concatenate(dev_pred_batches, axis=0)
                    if return_probabilities and len(dev_value_pred_batches) > 0:
                        all_value_preds = np.concatenate(dev_value_pred_batches, axis=0)

                if dev_report is not None:
                    print("\nDevelopment classification report:")
                    print(dev_report)

                # ---- EARLY STOPPING ----
                if dev_macro_f1 is not None and dev_macro_f1 > best_macro_f1:
                    best_macro_f1 = dev_macro_f1
                    best_epoch = epoch + 1
                    best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best epoch: {best_epoch} | Best dev macro F1: {best_macro_f1:.6f}")
                    break

                dev_loss_text = f"{dev_loss_epoch:.6f}" if dev_loss_epoch is not None else "n/a"
                best_f1_text = f"{best_macro_f1:.4f}" if best_macro_f1 != -float("inf") else "n/a"
                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f} | dev loss: {dev_loss_text} | Best F1 (macro): {best_f1_text}")

            else:
                print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss_epoch:.6f}")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            best_epoch = len(loss_values)
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    else:
        # ---- INFERENCE / EVALUATION ----
        model.eval()

        eval_losses = []
        pred_batches = []
        value_pred_batches = []

        y_true_all = []
        y_pred_all = []

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="BIO NER / Inference", total=len(data_loader), unit="batch"):
                batch = move_batch_to_device(batch, device)

                output = model(batch=batch)
                logits = output["logits"]

                preds = torch.argmax(logits, dim=-1)

                labels = batch.get(label_key)
                if labels is not None:

                    if criterion is not None:
                        loss = criterion(logits, labels)
                        eval_losses.append(loss.item())

                    mask = valid_token_mask(batch, labels)

                    preds_cpu = preds.detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    mask_cpu = mask.detach().cpu()
                    batch_size = preds_cpu.shape[0]

                    for i in range(batch_size):
                        true_seq = []
                        pred_seq = []

                        valid_positions = mask_cpu[i].bool()

                        for pred_id, true_id in zip(preds_cpu[i][valid_positions], labels_cpu[i][valid_positions]):
                            true_id = int(true_id.item())
                            pred_id = int(pred_id.item())

                            if true_id == ignore_index:
                                continue

                            true_seq.append(id2label[true_id])
                            pred_seq.append(id2label[pred_id])

                        if len(true_seq) > 0:
                            y_true_all.append(true_seq)
                            y_pred_all.append(pred_seq)

                    if return_predictions:
                        prediction_mask = valid_token_mask(batch)
                        pred_batches.append(preds[prediction_mask].detach().cpu().numpy())
                        if return_probabilities:
                            value_preds = torch.softmax(logits, dim=-1)
                            value_pred_batches.append(value_preds[prediction_mask].detach().cpu().numpy())
                else:
                    if return_predictions:
                        mask = valid_token_mask(batch)
                        pred_batches.append(preds[mask].detach().cpu().numpy())
                        if return_probabilities:
                            value_preds = torch.softmax(logits, dim=-1)
                            value_pred_batches.append(value_preds[mask].detach().cpu().numpy())

                del output, logits, preds, batch

        if len(eval_losses) > 0:
            eval_loss = float(np.mean(eval_losses))

        if len(y_true_all) > 0:
            best_macro_f1 = f1_score_sequeval(y_true_all, y_pred_all, average="macro", zero_division=0)
            final_classification_report = classification_report_sequeval(y_true_all, y_pred_all, digits=4, zero_division=0)

            print("\nClassification report:")
            print(final_classification_report)
        else:
            best_macro_f1 = None
            final_classification_report = None

        if return_predictions and len(pred_batches) > 0:
            all_pred = np.concatenate(pred_batches, axis=0)
            if return_probabilities and len(value_pred_batches) > 0:
                all_value_preds = np.concatenate(value_pred_batches, axis=0)

    return {"all_pred": all_pred, "all_value_preds": all_value_preds, "loss_values": loss_values, "development_loss_values": development_loss_values, "development_f1_values": development_f1_values, "eval_loss": eval_loss, "classification_report": final_classification_report, "best_macro_f1": best_macro_f1, "best_epoch": best_epoch, "best_model_state": best_model_state}

def update_df_with_final_pred_entities(df: pd.DataFrame, file_col: str = "File", token_idx_col: str = "Token idx", text_instance_col: str = "Text instance", label_col: str = "pred_label", id_col: str = "pred_id", outside_label: str = "O", outside_id: int = 2, fallback_when_final_span_missing: str = "max_existing"):
    """
    Actualiza un DataFrame de predicciones uniendo spans solapados o adyacentes que pertenecen a la misma entidad final.

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame original con las predicciones de spans. Debe contener columnas para archivo, instancia de texto, índices de tokens, etiqueta predicha e ID de predicción

        `file_col`: str
            - Nombre de la columna que identifica el archivo de origen. Por defecto es **File**

        `token_idx_col`: str
            - Nombre de la columna que contiene los índices de tokens del span. Por defecto es **Token idx**

        `text_instance_col`: str
            - Nombre de la columna que identifica la instancia de texto. Por defecto es **Text instance**

        `label_col`: str
            - Nombre de la columna que contiene la etiqueta predicha. Por defecto es **pred_label**

        `id_col`: str
            - Nombre de la columna que contiene el ID de la etiqueta predicha. Por defecto es **pred_id**

        `outside_label`: str
            - Etiqueta usada para marcar spans que no forman parte de ninguna entidad. Por defecto es **O**

        `outside_id`: int
            - ID asociado a la etiqueta **outside_label**. Por defecto es **2**

        `fallback_when_final_span_missing`: str
            - Estrategia a usar cuando el span final unido no existe como fila en el DataFrame. Puede ser **keep_original** para mantener las predicciones originales o **max_existing** para etiquetar el span existente más largo dentro del componente. Por defecto es **max_existing**

    Returns
    -------
        `df_out`: pd.DataFrame
            - Copia del DataFrame original con las columnas de predicción actualizadas. Mantiene exactamente las mismas filas, columnas, índice y forma que el DataFrame de entrada
    """
    if fallback_when_final_span_missing not in {"keep_original", "max_existing"}: raise ValueError("fallback_when_final_span_missing must be 'keep_original' or 'max_existing'")
    df_out = df.copy()
    label_to_id = df[df[label_col].ne(outside_label)].drop_duplicates(subset=[label_col]).set_index(label_col)[id_col].to_dict()
    normalized_token_idx = {idx: tuple(int(x) for x in normalize_list(row[token_idx_col])) for idx, row in df.iterrows()}
    token_sets = {idx: set(tok) for idx, tok in normalized_token_idx.items()}
    ent_df = df[df[label_col].ne(outside_label)].copy()
    decisions = []
    for (file_name, text_instance, entity_label), group in ent_df.groupby([file_col, text_instance_col, label_col], sort=False):
        components = get_final_entity_components_from_group(group=group, token_idx_col=token_idx_col)
        entity_id = label_to_id[entity_label]
        mask_same_context = (df[file_col].eq(file_name) & df[text_instance_col].eq(text_instance))
        context_indices = list(df[mask_same_context].index)
        for component in components:
            source_indices = component["source_row_indices"]
            if len(source_indices) <= 1: continue
            final_token_idx = tuple(component["final_token_idx"])
            final_set = set(final_token_idx)
            exact_candidates = [idx for idx in context_indices if normalized_token_idx[idx] == final_token_idx]
            if exact_candidates:
                final_row_idx = exact_candidates[0]
                decisions.append({"label": entity_label, "id": entity_id, "final_idx": final_row_idx, "source_indices": source_indices, "final_len": len(final_token_idx)})
                continue
            if fallback_when_final_span_missing == "keep_original": continue
            possible_candidates = [idx for idx in context_indices if token_sets[idx] and token_sets[idx].issubset(final_set)]
            if not possible_candidates: continue
            best_idx = max(possible_candidates, key=lambda idx: (len(token_sets[idx]), -abs(min(token_sets[idx]) - min(final_set))))
            best_len = len(token_sets[best_idx])
            max_source_len = max(len(token_sets[idx]) for idx in source_indices)
            if best_len < max_source_len: continue
            decisions.append({"label": entity_label, "id": entity_id, "final_idx": best_idx, "source_indices": source_indices, "final_len": best_len})
    decisions = sorted(decisions, key=lambda x: x["final_len"], reverse=True)
    protected_final_indices = set()
    for decision in decisions:
        final_idx = decision["final_idx"]
        source_indices = decision["source_indices"]
        entity_label = decision["label"]
        entity_id = decision["id"]
        if final_idx in protected_final_indices: continue
        for idx in source_indices:
            if idx != final_idx:
                df_out.at[idx, label_col] = outside_label
                df_out.at[idx, id_col] = outside_id
        df_out.at[final_idx, label_col] = entity_label
        df_out.at[final_idx, id_col] = entity_id
        protected_final_indices.add(final_idx)
    assert list(df_out.columns) == list(df.columns)
    assert list(df_out.index) == list(df.index)
    assert df_out.shape == df.shape
    return df_out
