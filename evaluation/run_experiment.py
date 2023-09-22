import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy

import pandas as pd

from lfma.utils import LitProgressBar, compute_annot_perf_clf

from sacred import Experiment

from evaluation.data_utils import load_data, DATA_PATH
from evaluation.experiment_utils import instantiate_model

from lightning_lite.utilities.seed import seed_everything

from skactiveml.pool import RandomSampling
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.utils import is_labeled, is_unlabeled, majority_vote

RESULT_PATH = "/mnt/work/crowd/crowd_results"

ex = Experiment()


@ex.config
def config():
    # Random seed for reproducibility.
    seed = 0

    # Further data set names are list in `evaluation/data_utils.py`.
    data_set_name = "letter"

    # Corresponds to the annotator sets in the accompanied article, where we have the following mapping:
    # none=independent, correlated=interdependent, rand_dep_10_100=random-interdependent, and inductive=inductive_25.
    data_type = "none"

    # Number of repeated experiments, i.e., train, validation, and test splits.
    n_repeats = 5

    # Fraction of test samples, which is only relevant for data sets without predefined splits.
    test_size = 0.2

    # Number of active learning cycles.
    n_al_cycles = 20

    # Active learning batch size, i.e., number of annotations queried per active learning cycle.
    al_batch_size = 128

    # Initial label size
    initial_label_size = 100

    # Flag to determine whether annotators should be selected according to their estimated performances.
    annot_perf_sel = True

    # Parameters passed to the `pytorch_lightning.Trainer` object.
    trainer_dict = {
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "enable_progress_bar": True,
        "logger": False,
        "enable_checkpointing": False,
    }

    # Name of the possible optimizer. See `evaluation/experiment_utils.py` for details.
    optimizer = "AdamW"

    # Parameters passed to the `torch.optim.Optimizer` object.
    optimizer_dict = {"lr": 0.01, "weight_decay": 0.0}

    # Name of the possible learning rate schedulers. See `evaluation/experiment_utils.py` for details.
    lr_scheduler = "CosineAnnealing"

    # Parameters passed to the `torch.optim.lr_scheduler.LRScheduler` object.
    lr_scheduler_dict = None

    # Batch sized used during training.
    batch_size = 64

    # Dropout rate used during training.
    dropout_rate = 0.0

    # Name of the multi-annotator learning technique. See `evaluation/experiment_utils.py` for details.
    model_name = "mr"

    # Parameters passed to the module of the multi-annotator learning technique. See the respective moudle class
    # in `lfma/modules` for details.
    model_dict = {}


@ex.automain
def run_al_experiment(
    data_set_name,
    data_type,
    n_repeats,
    test_size,
    n_al_cycles,
    al_batch_size,
    initial_label_size,
    annot_perf_sel,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    batch_size,
    dropout_rate,
    model_name,
    model_dict,
    seed,
):
    # Copy configuration.
    optimizer_dict = deepcopy(optimizer_dict)
    model_dict = deepcopy(model_dict)

    # Load data.
    use_annotator_features = model_dict.pop("use_annotator_features", False)
    ds = load_data(
        data_path=DATA_PATH,
        data_set_name=data_set_name,
        data_type=data_type,
        use_annotator_features=use_annotator_features,
        preprocess=True,
        n_repeats=n_repeats,
        test_size=test_size,
    )
    n_annotators = ds["y"].shape[1]
    classes = np.unique(ds["y_true"])
    print(compute_annot_perf_clf(y_true=ds["y_true"], y=ds["y"], missing_label=-1).values[0])
    best_idx = compute_annot_perf_clf(y_true=ds["y_true"], y=ds["y"], missing_label=-1).values[0].argsort()[-10:]

    # Performance dictionary.
    perf_dict = {"accuracy": [], "correct_label_ratio": []}
    for a in range(n_annotators):
        perf_dict[f"annotator_{a}"] = []

    n_iter = 0
    for tr, te in zip(ds["train"], ds["test"]):
        for key, item in perf_dict.items():
            perf_dict[key].append([])
        print(f"Train {model_name} on {n_iter}-th fold of {data_set_name}.")

        # Add callbacks.
        td = deepcopy(trainer_dict)
        td["callbacks"] = []
        if trainer_dict["enable_progress_bar"]:
            td["callbacks"].append(LitProgressBar())

        # Set global random seed.
        seed_everything(seed + n_iter)

        # Randomly add missing labels.
        n_samples = len(tr)
        random_state = np.random.RandomState(seed + n_iter)
        y_partial = np.full_like(ds["y"][tr], fill_value=-1)
        for a_idx in range(n_annotators):
            is_lbld_a = is_labeled(ds["y"][tr, a_idx], missing_label=-1)
            p_a = is_lbld_a / is_lbld_a.sum()
            initial_label_size_a = initial_label_size if is_lbld_a.sum() >= initial_label_size else is_lbld_a.sum()
            selected_idx_a = random_state.choice(np.arange(n_samples), size=initial_label_size_a, p=p_a, replace=False)
            y_partial[selected_idx_a, a_idx] = ds["y"][tr][selected_idx_a, a_idx]

        # Generate model according to configuration.
        data_loader_dict = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
        fit_dict = {
            "X": ds["X"][tr],
            "y": y_partial,
            "data_loader_dict": data_loader_dict,
            "transform": ds["transform"],
            "val_transform": None,
        }
        if model_name in ["madl", "gt", "mr"]:
            fit_dict["A"] = ds["A"]

        # Create query strategy.
        rs = RandomSampling(missing_label=-1, random_state=seed + n_iter)
        strategy = SingleAnnotatorWrapper(strategy=rs, missing_label=-1, random_state=seed + n_iter)
        candidate_indices = np.arange(n_samples)

        # Function to be able to index via an array of indices
        idx = lambda M: (M[:, 0], M[:, 1])

        # Fallback for random annotator selection.
        A_random = np.ones_like(y_partial)

        # Perform active learning.
        for c in range(n_al_cycles+1):
            if c > 0:
                if annot_perf_sel:
                    A_perf = model.predict_annotator_perf(fit_dict["X"], data_loader_dict={"batch_size": 512})
                else:
                    A_perf = A_random
                y_query = np.copy(fit_dict["y"])
                is_ulbld_query = np.copy(is_ulbld)
                if data_set_name in ["music", "label-me"]:
                    no_label_available = is_unlabeled(ds["y"][tr], missing_label=-1)
                    y_query[no_label_available] = 0
                    is_ulbld_query = is_unlabeled(y_query, missing_label=-1)
                is_candidate = is_ulbld_query.all(axis=-1)
                candidates = candidate_indices[is_candidate]
                query_indices = strategy.query(
                    X=fit_dict["X"],
                    y=y_query,
                    candidates=candidates,
                    A_perf=A_perf[candidates],
                    batch_size=al_batch_size,
                    n_annotators_per_sample=1,
                )
                fit_dict["y"][idx(query_indices)] = ds["y"][tr][idx(query_indices)]

            # Update aggregated annotations.
            if model_name in ["mr"]:
                fit_dict["y_agg"] = majority_vote(
                    y=y_partial,
                    classes=classes,
                    missing_label=-1,
                    random_state=seed + n_iter
                )

            # Generate model.
            model = instantiate_model(
                data_set_name=data_set_name,
                data_set=ds,
                model_name=model_name,
                trainer_dict=deepcopy(td),
                optimizer=optimizer,
                optimizer_dict=deepcopy(optimizer_dict),
                lr_scheduler=lr_scheduler,
                lr_scheduler_dict=deepcopy(lr_scheduler_dict),
                dropout_rate=dropout_rate,
                model_dict=deepcopy(model_dict),
                random_state=seed + n_iter + c,
            )

            # Fit model on initial labeled data.
            model.fit(**fit_dict)

            # Evaluate results.
            print(f"========================={c}. AL CYCLE=========================")
            y_pred = model.predict_proba(ds["X"][te], data_loader_dict={"batch_size": 512}).argmax(axis=-1)
            acc = np.mean(ds["y_true"][te] == y_pred)
            print(f"Accuracy: {acc}")
            perf_dict["accuracy"][n_iter].append(acc)
            is_ulbld = is_unlabeled(fit_dict["y"], missing_label=-1)
            print((~is_ulbld).any(axis=-1).sum())
            labels_per_annotator = (~is_ulbld).sum(axis=0)
            print(f"Labels per annotators: {labels_per_annotator}")
            print(labels_per_annotator.argsort()[-10:])
            print(best_idx)
            for a in range(n_annotators):
                perf_dict[f"annotator_{a}"][n_iter].append(labels_per_annotator[a]/labels_per_annotator.sum())
            is_lbld = np.array(~is_ulbld, dtype=int)
            correct_label_ratio = is_lbld * (y_partial == ds["y_true"][tr][:, None])
            correct_label_ratio = correct_label_ratio.sum() / is_lbld.sum()
            print(f"Correct label ratio: {correct_label_ratio}")
            perf_dict["correct_label_ratio"][n_iter].append(correct_label_ratio)

        n_iter += 1

    # Plot results.
    title = "| "
    for key in list(perf_dict.keys()):
        item = perf_dict[key]
        del perf_dict[key]
        key = key.replace('_', '-')
        item = np.array(item)
        means = item.mean(axis=0)
        stds = item.std(axis=0)
        perf_dict[f"mean-{key}"] = means
        perf_dict[f"std-{key}"] = stds
        if "annotator" in key:
            continue
        plt.errorbar(x=np.arange(n_al_cycles+1), y=means, yerr=stds, label=key, fmt='-o')
        title += f"mean-{key}: {means.mean().round(2)} | "
    plt.xlabel(f"active learning cycle with batch size of {al_batch_size}")
    plt.ylabel(f"accuracy")
    plt.title(title)
    plt.legend()
    plt.show()

    # Store results.
    perf_df = pd.DataFrame(perf_dict)
    filename = f"{data_set_name}-{model_name}-{annot_perf_sel}-{al_batch_size}-new.csv".lower()
    perf_df.to_csv(path_or_buf=f"{RESULT_PATH}/{filename}", index=False)
    print(perf_df.mean(axis=0))
