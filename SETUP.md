bash
```bash
export PYTHONPATH=.
```
powershell
```powershell
$env:PYTHONPATH = "."
```
cmd
```cmd
set PYTHONPATH=.
```

### Run command
in windows 
```
 python .\train\train_meld.py --train-dir data\MELD\train --validation-dir data\MELD\dev --test-dir data\MELD\test --model-dir saved_models\
```