# GraViti

This repository accompanies the paper:
**"GraViti: Graph-Level Variational Autoencoders with Relaxed Permutation Invariance".**

---

## Training a model

To launch training:

python job.py --arguments

### Example configuration
```
python job.py \
    --dataset PubChem16 \
    --chunk_size 524288 \
    --splits 0.9736 0.003 0.0234 \
    --batch_size 64 \
    --with_aromatic 1 \
    --latent_size 256 \
    --encoder_hidden_size 512 \
    --encoder_output_size 128 \
    --encoder_heads 4 \
    --encoder_layers 5 \
    --decoder_layers 5 \
    --decoder_sigma 0.0 \
    --reg_weight 0.0 \
    --variational 1 \
    --use_atom_attr 1 \
    --lr 1e-4 \
    --lr_gamma 0.1 \
    --weight_decay 0 \
    --beta 1e-5 \
    --beta_cycle 200 \
    --cycle_length 200 \
    --dropout 0.0 \
    --ratio_negative_edges 4.0 \
    --use_focal 1 \
    --weight_edge_loss 1.25 \
    --max_epochs 15 \
    --warmup_epochs 4 \
    --use_classes_weights 0 \
    --predict_hydrogens_formal_charges 1 \
    --reinject_size 1
```
---

## Evaluating a trained model

Default evaluation command:
```
python script_evaluation.py \
    --model_name <file name> \
    --models_root saved_models \
    --data_root data
```
---

## Citation

If you use GraViti in your research, please cite:

```
@misc{bresson2026gravitigraphlevelvariationalautoencoders,
      title={GraViti: Graph-Level Variational Autoencoders with Relaxed Permutation Invariance},
      author={Roman Bresson and Konstantinos Divriotis and Johannes F. Lutzeyer and Iakovos Evdaimon and Michalis Vazirgiannis},
      year={2026},
      eprint={2605.16668},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.16668}
}
```
