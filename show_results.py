#!/usr/bin/env python
# coding=utf-8
import flor
df = flor.dataframe()
print(df)
accs = flor.dataframe("model_name","data_path", "resize", "hflip", "normalize", 'train_acc', 'val_acc', 'test_idnet_acc', 'test_sidtd_acc')
print(accs)
