# Response profile

The package includes the components to conduct a response profile experiment.
Usually such experiments will include four steps, loading the data, feature selection
positivity exclusion,training a causal model and reporting the results.
The package includes tools to:
* load train test data from a csv file.
* Create a "pipeline" causal model that includes feature selection, positivity exclusion and fits a causal model
* Deserializing the model from a YAML file
* Tools to load the response profiles for other uses.

## Install

```
pip install -e ./responseprofiles
```
## Load data: `load_extracted_data.py`
Includes the method load_extracted_data that loads a train and test set dictionaries,
containing X dataframe and a and y series. The method assumes that the data is saved
in a csv file with id and an additional containing the id's with train test columns.

The method can also return a one versus rest treatment vector i.e to turn a multiple
treatment vector into a binary.


## Define a model (Deserialization) `dictionary_deserializer.py`
The deserialization class turns a dictionary into an initialized object
(This is in fact general utility that can be used outside response profile).
The dictionary has two keys, the first is name which is the name of the class the
second is the args that contains the arguments for initialization.
For example the following dictionary defined by a yaml file:

```yaml
name: class1
args:
  arg1: 5
  arg2: 6
```
NestedDictionaryDeserialization TYPE_FACTORY ( a dictionary) that links the name string
to the appropriate class.

args can be a primitive, list or dictionary NestedDictionaryDeserialization will
return an object of the format name(args), name(*args) or name(**args) according
to the content of the args. NestedDictionaryDeserialization can deserialize nested
dictionaries (args could contain a dictionary with a name args pair).

```yaml
name: obj1
args:
  arg1:
    name: obj2
    args:
      - 1
      - 2
  arg2: obj2
    args:
      b1: b1
      b2: b2
  arg3: True
```
The above example will create the following object:

`obj1(arg1=obj2(*[1,2]), arg2=obj2(b1='b1', b2='b2'), arg3=True)`

OutcomeProfilePipeline
Includes two classes the first, OutcomeProfilePipeline, connects feature selection,
positivity and casualty model into one object, the second OutcomeProfileExperiment manages
data loading and reporting as well.

## Response profile pipeline `outcome_profile_pipeline.py`

Let's have a look at the `OutcomeProfilePipeline` class with a simple definition:

```
outcome_estimator = StratifiedStandardization(GradientBoostingClassifier())
positivity_estimator = Trimming(make_pipeline(StandardScaler,LogisticRegression))
feature_transformation = ConstantFilter()
rp_pip = OutcomeProfilePipeline(outcome_estimator=outcome_estimator,
                                positivity_estimator=Trimming(),
                                feature_transformation=feature_transformation)
rp_pip.fit()
```

The above fit will actually do three consecutive fits, the first is fit transform
the data with `ConstantFilter` followed by fit transform the `positivity_estimator`
and only then fitting the causal model.


## Outcome data loader `outcome_data_loader.py`
The method `load_treatment_profile` load a treatment profile it enables loading
the treatment estimations, the response profiles or both. it assumes that two
.csv files will be in the same folder `exp1_postivity.csv` & `exp1_outcome.csv`
In each table multiple one versus the rest experiments can be described.
In the positivity each experiment will have a column with the treatment name
in the outcome file each treatment will have two columns one with the response for the
treated and one for the rest case.

## One versus another estimator one_versus_rest_effect_estimator.py
OneVersusRestEffectEstimator is a class that includes a base pipeline model and
duplicates it per treatment. The object trains a pipeline per treatment where the control
groups are every other treatment. Hence, the prediction or estimation methods
(predict, predict proba, predict_positivity etc.) return a data frame of predictions
with a column (for effect or positivity estimations) or two  (for the outcome)
per treatment.

```
outcome_estimator = StratifiedStandardization(GradientBoostingClassifier())
positivity_estimator = Trimming(make_pipeline(StandardScaler,LogisticRegression))
feature_transformation = ConstantFilter()
rp_pip = OutcomeProfilePipeline(outcome_estimator=outcome_estimator,
                                positivity_estimator=Trimming(),
                                feature_transformation=feature_transformation)
one_versus_rest = OneVersusRestEffectEstimator(base_model=rp_pip)
one_versus_rest.fit(X,a,y)
```

In the code above a new 'rp_pip' will be fitted per unique treatment in a.

## Experiment evaluator experiment_evaluators.py
OneVersusRestExperimentEvaluator object can be used to evaluate a one versus another
experiment. This is a collection of outcome evaluators (one per treatment) and of the
positivity evaluators (one per treatment). It can easily create figure for every pipeline
and create aggregated preference data frames.

```
outcome_estimator = StratifiedStandardization(GradientBoostingClassifier())
positivity_estimator = Trimming(make_pipeline(StandardScaler,LogisticRegression))
feature_transformation = ConstantFilter()
rp_pip = OutcomeProfilePipeline(outcome_estimator=outcome_estimator,
                                positivity_estimator=Trimming(),
                                feature_transformation=feature_transformation)
one_versus_rest = OneVersusRestEffectEstimator(base_model=rp_pip)
one_versus_rest.fit(X,a,y)
eval = OneVersusRestExperimentEvaluator(effect_est)
```

## Using it all together effect_profile.py
The script load data using the experiment definitions from a YAML file. The one
versus rest from a YAML file as well. fits the one versus rest and using an evaluator
creates figure and evaluations tables.
The script outputs the results in mlflow format

- data-folders - contains the data set file for train and test (X_train\test etc..).
- train-test-definition - the folder containing the partition to train and test
- effect-est-config-files - The YAML containing the experiment definitions multiple file applicable
- output-folder - The main output folder. in it a folder will be created for each data with a subdirectory for each model
- experiment-name - The name for the experiment if none the data set name will be used. Pay in mind that mlflow support
                   multiple experiments with the same name but different config files
- excluded-drugs-file The name of an excluded drug. Accepts multiple drugs.
```
    python scripts/estimate_effects.py \
    --data-folder  /Users/yoavkt/Documents/datasets/disability_progression_agonists \
    --train-test-definition  /Users/yoavkt/Documents/data_split.csv \
    --causal-model-config-files  ./scripts/one_versus_estimator.yaml \
    --effect-est-config-files  ./scripts/one_versus_estimator_dr.yaml \
    --excluded-drugs Drug1
    --excluded-drugs Drug2
```
The script supports multiple experiment names and multiple models.
