# Build an ML Pipeline for OSCAR object identification

## Table of contents

- [Preliminary steps](#preliminary-steps)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [Cookie cutter](#cookie-cutter)
  * [The configuration](#the-configuration)
  * [Running the entire pipeline or just a selection of steps](#running-the-entire-pipeline-or-just-a-selection-of-steps)
  * [Test an image based on a trained model](#test-an-image-based-on-a-trained-model) 

## Preliminary steps
### Create environment
Make sure to have pipenv installed and ready and you are at the app/objectid directory, then create a new environment:

```bash
# you may remove Pipfile.lock
> pipenv install
> pipenv shell
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### Cookie cutter
In order to make your job a little easier, you are provided a cookie cutter template that you can use to create 
stubs for new pipeline components. It is not required that you use this, but it might save you from a bit of 
boilerplate code. Just run the cookiecutter and enter the required information, and a new component 
will be created including the `python_env.yml` file, the `MLproject` file as well as the script. You can then modify these
as needed, instead of starting from scratch.
For example:

```bash
> cookiecutter cookie-mlflow-step -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

This will create a step called ``basic_cleaning`` under the directory ``src`` with the following structure:

```bash
> ls src/basic_cleaning/
python_env.yml  MLproject  run.py
```

You can now modify the script (``run.py``), the conda environment (``conda.yml``) and the project definition 
(``MLproject``) as you please.

The script ``run.py`` will receive the input parameters ``parameter1``, ``parameter2``,
``parameter3`` and it will be called like:

```bash
> mlflow run src/step_name -P parameter1=1 -P parameter2=2 -P parameter3="test"
```

### The configuration
As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file. 
Open this file and get familiar with its content. Remember: this file is only read by the ``main.py`` script 
(i.e., the pipeline) and its content is
available with the ``go`` function in ``main.py`` as the ``config`` dictionary. For example,
the name of the project is contained in the ``project_name`` key under the ``main`` section in
the configuration file. It can be accessed from the ``go`` function as 
``config["main"]["project_name"]``.

NOTE: do NOT hardcode any parameter when writing the pipeline. All the parameters should be 
accessed from the configuration file.

### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
# not recommended for now -- still in development stage
> cd app/objectid
> pipenv shell
> mlflow run .
```
This will run the entire pipeline. Please use the following to run working full pipeline for yolo.
You may configure all settings for both training and testing at the app/objectid/config.yaml at yolo section.

```bash
> cd app/objectid
> pipenv shell
> mlflow run . -P steps=box_label_labellmg,split_data_yolo,model_yolo_train,model_yolo_run
```

When developing or troubleshooting, it is useful to be able to run one step at a time. Say you want to run only
the ``box_label_labellmg`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=box_label_labellmg
```
If you want to run the ``box_label_labellmg`` and the ``split_data_yolo`` steps, you can similarly do:
```bash
> mlflow run . -P steps=box_label_labellmg,split_data_yolo
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
yolo -> split -> testSplit to 0 and yolo-> train -> imgsz to 640:

```bash
> mlflow run . \
  -P steps=box_label_labellmg,split_data_yolo,model_yolo_train \
  -P hydra_options="yolo.split.testSplit=0 yolo.train.imgsz=640"
```

### Test an image based on a trained model

First define the necessary parameters at the config.yaml at yolo.run section, or you may put in desired parameter as follow with hydra_options, with all default values stated at the config.yaml.    

```bash
> cd app/objectid
> pipenv shell
> mlflow run . \
  -P steps=model_yolo_run \
  -P hydra_options="yolo.run.conf=0.5"
```