Install Detectron2
```shell
python -m pip install -e detectron2
```
Run the command 
```shell
python -m torch.distributed.launch --nproc_per_node=4 main.py
```
