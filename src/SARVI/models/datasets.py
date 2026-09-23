from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, data_prepared, window_labels=None, file_names=None):
        self.data_prepared = data_prepared
        self.window_labels = window_labels
        self.file_names = file_names

        if self.window_labels is not None and len(self.window_labels) != len(self.data_prepared):
            raise ValueError("window_labels must have same length as data_prepared")

        if self.file_names is not None and len(self.file_names) != len(self.data_prepared):
            raise ValueError("file_names must have same length as data_prepared")

    def __len__(self):
        return len(self.data_prepared)

    def __getitem__(self, idx):
        windows = self.data_prepared[idx]

        if self.window_labels is not None:
            labels = self.window_labels[idx]

            if len(labels) != len(windows):
                raise ValueError(f"Text {idx} has {len(windows)} windows but {len(labels)} labels")
        else:
            labels = None

        if self.file_names is not None:
            file_name = self.file_names[idx]
        else:
            file_name = None

        return {"text_index": idx, "file_name": file_name, "windows": windows, "window_labels": labels}

class REDataset(Dataset):
    def __init__(self, data_prepared, relation_labels=None, file_names=None, flatten_examples=True):
        self.data_prepared = data_prepared
        self.relation_labels = relation_labels
        self.file_names = file_names
        self.flatten_examples = flatten_examples

        if self.relation_labels is not None and len(self.relation_labels) != len(self.data_prepared):
            raise ValueError("relation_labels must have same length as data_prepared")

        if self.file_names is not None and len(self.file_names) != len(self.data_prepared):
            raise ValueError("file_names must have same length as data_prepared")

        self.flat_indices = []
        if self.flatten_examples:
            for text_idx, examples in enumerate(self.data_prepared):
                if self.relation_labels is not None:
                    labels = self.relation_labels[text_idx]
                    if len(labels) != len(examples):
                        raise ValueError(f"Text {text_idx} has {len(examples)} RE examples but {len(labels)} labels")
                
                for example_idx in range(len(examples)):
                    self.flat_indices.append((text_idx, example_idx))

    def __len__(self):
        if self.flatten_examples:
            return len(self.flat_indices)
        return len(self.data_prepared)

    def __getitem__(self, idx):
        if self.flatten_examples:
            text_idx, example_idx = self.flat_indices[idx]
            example = self.data_prepared[text_idx][example_idx]

            if self.relation_labels is not None:
                labels = [self.relation_labels[text_idx][example_idx]]
            else:
                labels = None

            if self.file_names is not None:
                file_name = self.file_names[text_idx]
            else:
                file_name = None

            return {"text_index": text_idx, "file_name": file_name, "examples": [example], "relation_labels": labels}
        else:
            examples = self.data_prepared[idx]

            if self.relation_labels is not None:
                labels = self.relation_labels[idx]

                if len(labels) != len(examples):
                    raise ValueError(f"Text {idx} has {len(examples)} RE examples but {len(labels)} labels")
            else:
                labels = None

            if self.file_names is not None:
                file_name = self.file_names[idx]
            else:
                file_name = None

            return {"text_index": idx, "file_name": file_name, "examples": examples, "relation_labels": labels}

class ATTDataset(Dataset):
    def __init__(self, data_prepared, attribute_labels=None, file_names=None, flatten_examples=True):
        self.data_prepared = data_prepared
        self.attribute_labels = attribute_labels
        self.file_names = file_names
        self.flatten_examples = flatten_examples

        if self.attribute_labels is not None and len(self.attribute_labels) != len(self.data_prepared):
            raise ValueError("attribute_labels must have same length as data_prepared")

        if self.file_names is not None and len(self.file_names) != len(self.data_prepared):
            raise ValueError("file_names must have same length as data_prepared")

        self.flat_indices = []
        if self.flatten_examples:
            for text_idx, examples in enumerate(self.data_prepared):
                if self.attribute_labels is not None:
                    labels = self.attribute_labels[text_idx]
                    if len(labels) != len(examples):
                        raise ValueError(f"Text {text_idx} has {len(examples)} ATT examples but {len(labels)} labels")

                for example_idx in range(len(examples)):
                    self.flat_indices.append((text_idx, example_idx))

    def __len__(self):
        if self.flatten_examples:
            return len(self.flat_indices)
        return len(self.data_prepared)

    def __getitem__(self, idx):
        if self.flatten_examples:
            text_idx, example_idx = self.flat_indices[idx]
            example = self.data_prepared[text_idx][example_idx]

            if self.attribute_labels is not None:
                labels = [self.attribute_labels[text_idx][example_idx]]
            else:
                labels = None

            if self.file_names is not None:
                file_name = self.file_names[text_idx]
            else:
                file_name = None

            return {"text_index": text_idx, "file_name": file_name, "examples": [example], "attribute_labels": labels}

        examples = self.data_prepared[idx]

        if self.attribute_labels is not None:
            labels = self.attribute_labels[idx]

            if len(labels) != len(examples):
                raise ValueError(f"Text {idx} has {len(examples)} ATT examples but {len(labels)} labels")
        else:
            labels = None

        if self.file_names is not None:
            file_name = self.file_names[idx]
        else:
            file_name = None

        return {"text_index": idx, "file_name": file_name, "examples": examples, "attribute_labels": labels}

class ICD10Dataset(Dataset):
    def __init__(self, data_prepared):
        self.diagnoses = data_prepared["diagnoses"]
        self.icd_codes = data_prepared.get("icd_codes")
        self.file_names = data_prepared.get("file_names")
 
        if self.icd_codes is not None and len(self.icd_codes) != len(self.diagnoses):
            raise ValueError("icd_codes must have the same length as diagnoses")
        if self.file_names is not None and len(self.file_names) != len(self.diagnoses):
            raise ValueError("file_names must have the same length as diagnoses")

    def __len__(self):
        return len(self.diagnoses)

    def __getitem__(self, idx):
        icd_code = self.icd_codes[idx] if self.icd_codes is not None else None
        return self.diagnoses[idx], icd_code

# class ICD10TrainingDataset(Dataset):
#     def __init__(self, inputs, queries, targets):
#         self.inputs = inputs
#         self.queries = queries
#         self.targets = targets

#         if not (len(self.inputs) == len(self.queries) == len(self.targets)):
#             raise ValueError("inputs, queries, and targets must have the same length")

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.queries[idx], self.targets[idx]
