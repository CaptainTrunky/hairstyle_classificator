
***Training***

```

python train.py train $(realpath ~/workspace/datasets/hair-train/data256x256) 20
```

***Inference***

```
python inference.py run $(realpath ~/workspace/datasets/hair-val) ./best_model.pth
```

Output path "./results.csv" is hardcoded.


Potential ways for improvement

1) investigate larger models with static/quantization aware training for fitting into 1.5mb constraint
2) more rigorius experiments over augmentations
3) larger batch size probably would be beneficial
4) modern optimizers (adabeleif, ranger, etc) + LR scheduling
5) we don't really need RGB data for this task, could investigate grayscale domain for slight model optimization
6) dataset checks, there could be ambigouos samples
7) could check larger models distillation / models pruning
