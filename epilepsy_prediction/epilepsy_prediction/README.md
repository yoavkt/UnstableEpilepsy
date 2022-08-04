# public_epilepsy

## How to use
Install the package cd to the containing folder install locally using 
```
pip install -e .
```

The package has two main methods

```
load_data
```
 which loads a pickle file with a dictionary containing a X feature matrix and y series.
 An additional method is 
 ```
load_model
```
Which loads the xgboost  models for the specific drug