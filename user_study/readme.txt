Before the participant begins, be sure to edit the task and 
method in the preference_engine.py file.

```
pl = PreferenceLearner(1, 'handover', 'infogain')
```
the tasks can be: 'handover' or 'blossom'
methods can be: 'infogain', 'CMAES', 'CMAESIG'

To begin the experiment, run python

```
python3 start_interface.py
```