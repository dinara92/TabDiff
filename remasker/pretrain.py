"""Pretrain the ReMasker-style tabular MAE (remasker/model.py) on Adult.

One run, frozen and reused unchanged across every drift-generator training
seed downstream (drifting_true/main.py --remasker-ckpt) -- no seed sweep
needed here, same reasoning as why decoder_phi's weights are a fixed,
once-computed artifact (see the plan, velvet-cuddling-finch.md).

--pretrain-data is the open call recorded in the plan, not resolved in
code: 'real' (default) trains on real Adult rows, numeric columns from
preprocess()'s quantile-normalized X_num and categorical columns one-hot
from X_cat -- the same representation ReMasker's own paper uses, no VAE in
the loop. 'decoded' trains on drifting_true/decoder_phi.py's reconstruction
of those same rows instead (decode(encode(real))), closing any pretrain/
drift-time domain gap at the cost of never seeing genuinely real data.
"""

import os
import time
import argparse
import warnings

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from utils_train import preprocess
from remasker.model import ReMaskerMAE

warnings.filterwarnings('ignore')


def load_pretrain_data(args, device):
    dataset_dir = f'data/{args.dataname}'
    import json
    with open(f'{dataset_dir}/info.json') as f:
        info = json.load(f)

    if args.pretrain_data == 'real':
        # TabDiff's preprocess returns an extra int_inverse compared with
        # TabSyn's, so unpack positionally rather than by fixed arity.
        out = preprocess(dataset_dir, task_type=info['task_type'], inverse=True)
        (X_train_num, _), (X_train_cat, _), categories, d_numerical = out[:4]
        x_num = torch.tensor(X_train_num).float()
        x_cat_idx = torch.tensor(X_train_cat).long()
        x_cat_blocks = [F.one_hot(x_cat_idx[:, i], num_classes=c).float() for i, c in enumerate(categories)]
    else:
        from drifting_true.decoder_phi import DecoderPhi
        from remasker.model import split_flat_features

        class A:
            pass
        a = A()
        a.dataname = args.dataname
        # Only the 'decoded' branch needs TabSyn's VAE latents; imported
        # lazily so 'real' pretraining runs in the TabDiff repo unchanged.
        from tabsyn.latent_utils import get_input_train
        train_z, _, _, _, _ = get_input_train(a)
        phi = DecoderPhi(args.dataname, device=device)
        with torch.no_grad():
            flat = phi(train_z.to(device)).cpu()
        d_numerical, categories = phi.d_numerical, phi.categories
        x_num, x_cat_blocks = split_flat_features(flat, d_numerical, categories)

    return x_num, x_cat_blocks, d_numerical, categories


def masked_reconstruction_loss(x_num, x_cat_blocks, x_num_pred, x_cat_pred, mask, d_numerical):
    """Masked-positions-only loss: MSE for numeric columns, cross-entropy
    (against the true class, i.e. argmax of the one-hot/softmax input) for
    categorical columns -- standard MAE convention."""
    num_mask = mask[:, :d_numerical]
    if num_mask.any():
        num_loss = F.mse_loss(x_num_pred[num_mask], x_num[num_mask])
    else:
        num_loss = x_num.new_zeros(())

    cat_losses = []
    for i, (block, logits) in enumerate(zip(x_cat_blocks, x_cat_pred)):
        col_mask = mask[:, d_numerical + i]
        if col_mask.any():
            target = block[col_mask].argmax(dim=-1)
            cat_losses.append(F.cross_entropy(logits[col_mask], target))
    cat_loss = torch.stack(cat_losses).mean() if cat_losses else x_num.new_zeros(())

    return num_loss + cat_loss, num_loss, cat_loss


def main(args):
    device = args.device
    x_num, x_cat_blocks, d_numerical, categories = load_pretrain_data(args, device)
    n_cat = len(categories)
    print(f"pretrain data: {args.pretrain_data}, {x_num.shape[0]} rows, "
          f"{d_numerical} numeric + {n_cat} categorical columns")

    model = ReMaskerMAE(
        d_numerical, categories, embed_dim=args.embed_dim,
        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        n_head=args.n_head, d_ffn_factor=args.d_ffn_factor,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print("the number of parameters", num_params)

    dataset = TensorDataset(x_num, *x_cat_blocks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

    ckpt_dir = f'remasker/ckpt/{args.dataname}' + (f'-{args.run_name}' if args.run_name else '')
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.random_init:
        # Random-phi control: same architecture, frozen at its random
        # initialization, no training at all. Isolates whether the
        # decode_then_featurize + ReMasker gain (remasker_phi_2026-08-25.md)
        # is specifically about *learned* features, or just "a second,
        # independent drift_loss term in a different feature space helps,
        # learned or not." Saved in the exact same format RemaskerPhi
        # expects, so it's a drop-in --remasker-ckpt swap, no other changes.
        torch.save({
            'model': model.state_dict(),
            'd_numerical': d_numerical,
            'categories': categories,
            'embed_dim': args.embed_dim,
            'encoder_layers': args.encoder_layers,
            'decoder_layers': args.decoder_layers,
            'n_head': args.n_head,
            'd_ffn_factor': args.d_ffn_factor,
            'pretrain_data': 'random_init',
            'epoch': -1,
        }, f'{ckpt_dir}/model.pt')
        print(f'Saved random-init (untrained) checkpoint to {ckpt_dir}/model.pt')
        return

    start_time = time.time()
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in loader:
            xn, xc = batch[0].to(device), [b.to(device) for b in batch[1:]]
            optimizer.zero_grad()
            xn_pred, xc_pred, mask = model(xn, xc, args.mask_ratio_min, args.mask_ratio_max)
            loss, num_loss, cat_loss = masked_reconstruction_loss(xn, xc, xn_pred, xc_pred, mask, d_numerical)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        epoch_loss = running / n_batches
        scheduler.step(epoch_loss)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}/{args.epochs} loss={epoch_loss:.6f} "
                  f"num={num_loss.item():.6f} cat={cat_loss.item():.6f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'model': model.state_dict(),
                'd_numerical': d_numerical,
                'categories': categories,
                'embed_dim': args.embed_dim,
                'encoder_layers': args.encoder_layers,
                'decoder_layers': args.decoder_layers,
                'n_head': args.n_head,
                'd_ffn_factor': args.d_ffn_factor,
                'pretrain_data': args.pretrain_data,
                'epoch': epoch,
            }, f'{ckpt_dir}/model.pt')

    print('Time:', time.time() - start_time)
    print(f'Saved best checkpoint (loss={best_loss:.6f}) to {ckpt_dir}/model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain ReMasker-style tabular MAE')
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--pretrain-data', type=str, default='real', choices=['real', 'decoded'],
                         help='real: real Adult rows (one-hot categoricals). decoded: '
                              'decoder_phi(train_z) reconstructions (softmax categoricals). '
                              'Open call, see the plan -- not resolved in code, a flag instead.')
    parser.add_argument('--random-init', action='store_true',
                         help='Skip training entirely -- save the freshly-initialized model as-is. '
                              'The random-phi control: isolates whether the ReMasker gain is about '
                              'learned features specifically, or any second drift_loss term in a '
                              'different (even untrained) feature space. --pretrain-data still '
                              'selects which data to build d_numerical/categories from, but it is '
                              'never actually trained on.')
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--encoder-layers', type=int, default=8)
    parser.add_argument('--decoder-layers', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--d-ffn-factor', type=int, default=4)
    parser.add_argument('--mask-ratio-min', type=float, default=0.3)
    parser.add_argument('--mask-ratio-max', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--log-every', type=int, default=20)
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    main(args)
