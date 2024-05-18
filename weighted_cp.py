import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import numpy as np

from ..lightning_module.SegmPLModuleV2 import SegmPLModuleV2
from ..data_management.mri_datamodule import LightningMRIModule
from ..routines.utils import read_monitor_mode_from_folder, get_best_ckpt_from_mode_and_monitor, string_to_torch_device
from ..data_management.utils import concatenate_channels
from ..feature_extractor import Extractor


from monai.metrics import DiceMetric
from monai.networks import one_hot

from scipy.optimize import brentq

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


"""
Perform weighted Conformal Prediction (WCP) using either oracle covariates or latent representations
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--id-datasets', type=str, nargs='+', default=['val'], required=True, help='path to in-distribution datasets')
    parser.add_argument('--ds-datasets', type=str, nargs='+', default=None, required=False, help='path to shifted dataset')
    parser.add_argument('--run-folder', type=str, required=True, help='path to trained segmentation model')
    parser.add_argument('--device', type=int, default=None, help='gpu device')
    parser.add_argument('--trials', type=int, default=10,
                        help='number of trials - shuffling calibration and test ID datapoints')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.10, help='level of coverage desired')
    parser.add_argument('--dev-run', action='store_true', default=False)
    parser.add_argument('--save-suffix', type=str, default='')
    parser.add_argument('--covariate', type=str, default=None)
    parser.add_argument('--classifier', type=str, required=False, choices=['log', 'forest'],
                        default='log', help='auxiliary classifier used to compute the weights')

    args = parser.parse_args()
    args.device = string_to_torch_device(args.device)

    return args


def get_weights(latent_train, latent_test, labels_train, labels_test, clip=True,
                clip_min=0.01, clip_max=0.99, cv=20, classifier='log'):
    """
    Function to compute the weights used in Weighted Conformal Prediction
    :param latent_train: latent feature vectors for calibration datapoints
    :param latent_test: latent feature vectors for test datapoints
    :param labels_train: class labels for calibration datapoints
    :param labels_test: class labels for test datapoints
    :param weight_mode: string defining the auxiliary classification model
    :param clip_min: lower value used to clip the probability for stability
    :param clip_max: lower value used to clip the probability for stability
    :param cv: number of folds for the cross-validation
    :return:
    """

    n_train, n_test = len(latent_train), len(latent_test)

    # weights correspond to the density ratio dPtest/dPtrain
    # in the case the weights are not bound in the range [0, 1]
    # weights are computed for all calibration and test samples

    if classifier == 'log':
        cmodel = LogisticRegression(max_iter=1000)
    else:
        raise NotImplementedError

    cat_latent = np.concatenate([latent_train, latent_test])
    cat_labels = np.concatenate([labels_train, labels_test])

    # train classification model using cross-validation + prediction
    prob_distribution = cross_val_predict(cmodel, cat_latent, cat_labels, cv=cv, verbose=1,
                                          method='predict_proba')  # n_samples, 2
    prob1 = prob_distribution[:, 1]
    predicted_classes = (prob1 >= 0.5).astype(np.uint8)  # C=1 if pred as test, otherwise C=0
    acc = accuracy_score(cat_labels, predicted_classes)

    if clip:
        # clip probas to avoid infinite weights
        prob1 = np.clip(prob1, clip_min, clip_max)

    weights = prob1 / (1 - prob1)

    weights_train = weights[0:n_train]
    weights_test = weights[n_train:]

    return weights_train, weights_test, acc


def run_experiment(df_calib, df_test, classifier='log', alpha=0.10, foreground_classes=[1],
                   cv=20, oracle_covariate=False, correction='none'):

    if oracle_covariate is True:
        infeat_calib = np.asarray(df_calib['covariate'])[:, None]
        infeat_test = np.asarray(df_test['covariate'])[:, None]
    else:
        infeat_calib = torch.concat(df_calib['latent'].tolist(), dim=0)
        infeat_test = torch.concat(df_test['latent'].tolist(), dim=0)

    ws_calib, ws_test, acc = get_weights(latent_train=infeat_calib, latent_test=infeat_test,
                                         labels_train=[0] * len(infeat_calib),
                                         labels_test=[1] * len(infeat_test), classifier=classifier, cv=cv,
                                         correction=correction)

    # compute effective size
    neff = np.sum(np.abs(ws_calib)) ** 2 / np.sum(np.abs(ws_test) ** 2)

    n_calib = len(infeat_calib)
    row = {'alpha': alpha, 'accuracy': acc, 'neff': neff, 'nfeat': infeat_calib.shape[1]}

    df_calib = df_calib.assign(weights=ws_calib)
    df_test = df_test.assign(weights=ws_test)

    for c in foreground_classes:
        class_calib_df = df_calib.loc[df_calib[f'cal_{c}'].notnull()]
        class_test_df = df_test.loc[df_test[f'cal_{c}'].notnull()]

        cal_scores = np.asarray(class_calib_df[f'cal_{c}'])

        lower_bound_test = class_test_df[f'low_{c}'].tolist()
        upper_bound_test = class_test_df[f'up_{c}'].tolist()
        mean_vol_test = class_test_df[f'mean_{c}'].tolist()
        true_test = class_test_df[f'true_{c}'].tolist()

        class_test_weights = class_test_df['weights'].tolist()
        class_calib_weights = class_calib_df['weights'].tolist()

        ceil = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        naive_qhat = np.quantile(cal_scores, ceil, method='higher')
        row['naive_qhat'] = naive_qhat

        naive_prediction_sets = [(max(0, low - naive_qhat), up + naive_qhat) for low, up in zip(lower_bound_test,
                                                                                                upper_bound_test)]
        # empirical coverages
        naive_covered = [(truth >= naive_sets[0]) & (truth <= naive_sets[1]) for truth, naive_sets in
                         zip(true_test, naive_prediction_sets)]
        naive_widths = [np.abs(naive_sets[1] - naive_sets[0]) for naive_sets in naive_prediction_sets]

        naive_coverage = sum(naive_covered) / len(naive_covered)
        maes = [np.abs(true - pred) for true, pred in zip(true_test, mean_vol_test)]
        mean_mae = sum(maes) / len(maes)
        naive_width = sum(naive_widths) / len(naive_widths)

        # weighted CP happens here
        qhats_weighted = []
        for wtest in tqdm(class_test_weights):
            piws = np.asarray([w / (sum(class_calib_weights) + wtest) for w in class_calib_weights])
            q_w = get_weighted_quantile(cal_scores, piws, alpha)
            qhats_weighted.append(q_w)

        weighted_prediction_sets = [(max(0, low - w_qhat), up + w_qhat) for low, up, w_qhat in
                                    zip(lower_bound_test, upper_bound_test, qhats_weighted)]
        weighted_covered = [(truth >= w_sets[0]) & (truth <= w_sets[1]) for truth, w_sets in
                            zip(true_test, weighted_prediction_sets)]
        weighted_widths = [np.abs(w_sets[1] - w_sets[0]) for w_sets in weighted_prediction_sets]

        weighted_coverage = sum(weighted_covered) / len(weighted_covered)
        weighted_width = sum(weighted_widths) / len(weighted_widths)

        class_row = {f'naive_coverage_{c}': naive_coverage,
                     f'naive_qhat_{c}': naive_qhat,
                     f'naive_width_{c}': naive_width,
                     f'weighted_coverage_{c}': weighted_coverage,
                     f'weighted_width_{c}': weighted_width,
                     f'mae_{c}': mean_mae
                     }
        row.update(class_row)

    return row, ws_calib, ws_test


def get_weighted_quantile(cal_scores, weights, alpha, low=0, max=20000):
    def critical_point_quantile(q):
        return (weights * (cal_scores <= q)).sum() - (1 - alpha)

    # check bounds
    a = critical_point_quantile(low)
    b = critical_point_quantile(max)
    if a * b < 0:  # different signs
        return brentq(critical_point_quantile, low, max)
    else:
        return max  # no solution, return max


def get_data(model, df, device):
    """
    Run inferences on df data using "model" on GPU: "device"
    return probs, gts, and weights
    :param model:
    :param df:
    :param device:
    :return:
    """

    df = df.assign(train=[1] * len(df))
    dm = LightningMRIModule(df=df,
                            visit_key=model.hparams.visit_key,
                            channel_names=model.hparams.channel_names,
                            segm_name=model.hparams.segm_name,
                            augmentation_mode="none",
                            batch_size=1,
                            val_batch_size=1,
                            crop_size=model.hparams.crop_size if not model.hparams.patch_training else model.hparams.patch_size,
                            crop_type=model.hparams.crop_type,
                            num_workers=model.hparams.num_workers,
                            normalization=model.hparams.normalization,
                            patch_size=model.hparams.patch_size)

    dm.setup(stage='fit')
    data_loader = dm.train_dataloader(shuffle=False)  # use training mode with "none" augmentation

    predictions = []
    umaps = []
    for n in foreground_classes:
        umaps.append(f'lower_{n}')
        umaps.append(f'upper_{n}')
        umaps.append(f'mean_{n}')

    layer_name = 'net.backbone.upsamples.3.conv_block.conv2.conv'
    extractor = Extractor(layer=[layer_name])  # little tool to colect the intermediate activations
    dice_metric = DiceMetric(include_background=False, reduction='mean_batch')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, 0)):
                image_dict = concatenate_channels(batch, model.hparams.channel_names)
                x = image_dict['images']
                visits = batch['visit']
                y = image_dict['seg'].long().to(device)
                yone_hot = one_hot(y, model.hparams.n_classes)  # b n h w d
                x = x.to(device)
                batch_size = len(x)

                for b in range(batch_size):
                    xb = x[b][None, ...]  # 1 n h w d
                    yb = yone_hot[b][None, ...]  # 1 n h w d
                    present_classes = torch.unique(y[b]).tolist()

                    # extract latent representation
                    latentb = extractor.predict(model, xb)[0][layer_name].detach().cpu()  # latent representation
                    row = {'id': visits[b], 'latent': latentb}

                    pred_dict = model.prediction_wrapper(xb, **{'umaps': umaps})
                    pmean = torch.sigmoid(pred_dict['logits'])

                    # COMPUTE Dice
                    seg = torch.argmax(pmean, 1, keepdim=True)
                    seg_one_hot = one_hot(seg, n_classes)
                    dice_metric(seg_one_hot, yb)
                    dice_classes = dice_metric.aggregate()
                    dice_metric.reset()

                    for _class in range(1, model.hparams.n_classes):
                        # compute stats per class
                        pmean_class = pmean[:, _class]
                        pup_class = pred_dict[f'uncertainty_upper_{_class}']
                        plow_class = pred_dict[f'uncertainty_lower_{_class}']

                        # area are expressed in % of the image size
                        lower_estimate = (plow_class >= 0.5).sum().item()
                        upper_estimate = (pup_class >= 0.5).sum().item()
                        mean_estimate = (pmean_class >= 0.5).sum().item()
                        true_estimate = yb[:, _class].sum().item()

                        # conformal risk
                        cal_score = np.maximum(true_estimate - upper_estimate, lower_estimate - true_estimate)

                        class_row = {f'low_{_class}': lower_estimate, f'mean_{_class}': mean_estimate,
                                     f'up_{_class}': upper_estimate, f'true_{_class}': true_estimate,
                                     f'cal_{_class}': cal_score}

                        if _class in present_classes:
                            class_row[f'dice_{_class}'] = dice_classes[_class - 1].item()
                        else:
                            class_row[f'dice_{_class}'] = np.nan

                        row.update(class_row)

                    predictions.append(row)

        return predictions


if __name__ == '__main__':
    args = parse_args()
    device = args.device
    run_folder = args.run_folder
    alpha = args.alpha
    np.random.seed(24)

    # load trained model
    monitor, mode = read_monitor_mode_from_folder(run_folder)
    checkpoint_dir = os.path.join(run_folder, 'checkpoints')
    ckpt_path = get_best_ckpt_from_mode_and_monitor(checkpoint_dir, mode=mode, monitor=monitor)

    model = SegmPLModuleV2.load_from_checkpoint(ckpt_path)
    n_classes = model.hparams.n_classes

    model.eval()
    model.to(device)

    out_folder = os.path.join(run_folder, 'risk_control')
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    print(f'Will save at {out_folder}.')

    foreground_classes = range(1, n_classes)
    id_datasets = []
    for id_dataset in args.id_datasets:
        id_datasets.append(pd.read_csv(id_dataset))

    df_cat = pd.concat(id_datasets)
    if args.dev_run:
        df_cat = df_cat.sample(5)

    df_cat = df_cat.assign(source=['id'] * len(df_cat))

    n_id_samples = len(df_cat)
    print(f'Got {n_id_samples} in-distribution samples for the conformal step')

    # now load domain shift data if provided
    if args.ds_datasets is not None:
        print('Load DS datasets for robustness testing')
        for idx, ds in enumerate(args.ds_datasets):
            ds_df = pd.read_csv(ds)
            if args.dev_run:
                ds_df = ds_df.sample(5)
            ds_df = ds_df.assign(source=[f'ds_{idx}'] * len(ds_df))
            df_cat = pd.concat([df_cat, ds_df])

    # first step is to gather predictions on all datapoints (segmentation, probabilities, and latent vectors)
    all_predictions = get_data(model, df_cat, device)
    all_predictions = pd.DataFrame(all_predictions)
    all_predictions = all_predictions.assign(source=df_cat['source'].tolist())

    del model  # no need segm model anymore, free CUDA memory
    all_covariates = []
    if args.covariate is not None:
        covariate_id = df_cat[df_cat['source'] == 'id'][args.covariate].tolist()
        print('id : ', sum(covariate_id) / len(covariate_id), len(covariate_id))
        all_covariates += covariate_id
        if args.ds_datasets is not None:
            for idx, ds in enumerate(args.ds_datasets):
                covariate_ds = pd.read_csv(ds)[args.covariate].tolist()
                all_covariates += covariate_ds
                print('ds : ', sum(covariate_ds) / len(covariate_ds), len(covariate_ds))

        all_predictions = all_predictions.assign(covariate=all_covariates)
        all_predictions = all_predictions[all_predictions['covariate'].notna()]  # drop rows with NaN covarites

    N_RUNS = args.trials if not args.dev_run else 1
    n_calib = n_id_samples // 2
    n_test = n_id_samples - n_calib

    out_data = {'id': []}
    if args.ds_datasets is not None:
        for idx in range(len(args.ds_datasets)):
            out_data[f'ds_{idx}'] = []

    n_cv = 2 if args.dev_run else 20
    for run in tqdm(range(N_RUNS)):
        current_df = all_predictions[all_predictions['source'] == 'id'].sample(frac=1)  # shuffle id
        df_calib, df_test = train_test_split(current_df, test_size=n_test)

        row_id, wcalib_id, wtest_id = run_experiment(df_calib, df_test, classifier=args.classifier,
                                                     foreground_classes=range(1, n_classes), alpha=alpha, cv=n_cv,
                                                     oracle_covariate=(args.covariate is not None))
        row_id['run'] = run
        row_id['dataset'] = 'id'

        df_weight_calib_id = pd.DataFrame({'w_calib': wcalib_id})
        df_weight_test_id = pd.DataFrame({'w_test': wtest_id})
        if args.covariate is not None:
            df_weight_calib_id = df_weight_calib_id.assign(covariate=df_calib['covariate'].tolist())

        out_data['id'].append(row_id)
        if args.ds_datasets is not None:
            for idx in range(len(args.ds_datasets)):
                ds_df = all_predictions[all_predictions['source'] == f'ds_{idx}']
                row_ds, wcalib_ds, wtest_ds = run_experiment(df_calib, ds_df, classifier=args.classifier,
                                                             foreground_classes=range(1, n_classes), alpha=alpha,
                                                             cv=n_cv, oracle_covariate=(args.covariate is not None))
                row_ds['run'] = run
                row_ds['dataset'] = f'ds_{idx}'
                out_data[f'ds_{idx}'].append(row_ds)

    # save experiments results
    for key in out_data.keys():
        df = pd.DataFrame(out_data[key])
        # save all metrics in a DataFrame
        sp = os.path.join(out_folder, f'data_cov_{args.save_suffix}_{key}.csv')
        df.to_csv(sp)

    all_predictions.drop('latent', axis=1)
    spath = f'wcp_abc_{args.save_suffix}.csv'
    all_predictions.to_csv(spath)

