Evaluation
---


Thins you need to change in `scripts/eval.py`:

```python
DATA_FOLDER = "/home/kin/workspace/DUFOMap/data/"
METHODS_NAME = "edomap"
SEQUENCE_SELECT = ["00"]
```

Then run:

```bash
cd scripts
cmake -B build && cmake --build build
./build/export_eval_pcd ${data_path} ${method_name}_output.pcd ${min_dis}

# example
./build/export_eval_pcd /home/kin/workspace/DUFOMap/data/00 edomap_output.pcd 0.05
python scripts/eval.py
```

Example output:
```
➜  EDOMap git:(feat/3d_binary_tree) ✗ python3 /home/kin/workspace/EDOMap/scripts/eval.py
Processing:  00  with method:  edomap
Sequence:  00
| Methods   |     # TN |   # TP |     SA ↑ |       DA ↑ |      AA ↑ |
|-----------+----------+--------+----------+------------+-----------|
| removert  | 17168885 |  39863 |  99.4361 |  41.5313   |  64.2628  |
| erasor    | 11516999 |  94577 |  66.7024 |  98.5352   |  81.0711  |
| octomap   | 11749691 |  95689 |  68.0501 |  99.6937   |  82.366   |
| octomapg  | 14835269 |  94906 |  85.9206 |  98.8779   |  92.1719  |
| octomapfg | 16068334 |  94709 |  93.0621 |  98.6727   |  95.8263  |
| dynablox  | 16707284 |  87041 |  96.7627 |  90.6838   |  93.6739  |
| dufomap   | 17002553 |  94686 |  98.4728 |  98.6487   |  98.5607  |
| gt        | 17266247 |  95983 | 100      | 100        | 100       |
| edomap    | 16728107 |    xxx |  96.8833 |   xxxx     |   xxx     |
====================  Friendly dividing line ^v^  ==================== 
```