# Artificial Intelligence Lab5

This is the code I wrote in Lab5 of `Artificial Intelligence` which is  a professional compulsory course of `ECNU DaSE`.

**Tips:  GPU is required**

## Setup

This implemetation is based on `Python3`. To run the code, you need the following dependencies:

- matplotlib==3.5.1
- numpy==1.22.2
- Pillow==9.2.0
- scikit_learn==1.1.1
- torch==1.10.2
- torchvision==0.11.3
- tqdm==4.62.3
- transformers==4.19.2

You can simply run

```
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.	

```
├-bert-base-uncased/    # pretrained model bert-base-uncased
├-data
│  │--dev_with_label.txt		# dev data after split
│  │--result.txt		# prediction result
│  │--test_without_label.txt		# test data for predict
│  │--train.txt			# train data before split,used to train the final model
│  │--train_with_label.txt			# train data after split
│  └-data/			# text and image data     
├-image/		# charts saved during training 
├-model
│  │--best-model.pth		# best model saved during training with train type 'dev'
│  │--best_model_with_all_train.pth # best model saved during training with train type 'all'
│--data_loader.py		# code for data processing
│--main.py		# main code for calling other parts
│--model.py		# code for model
│--MSA.py		# code for train、test、predict processing
│--README.md
│--requirements.txt
```

## Run code

To run code, the template of the script running on cmd is as follows

```
python main.py [args]
```

args(you can run `python main.py --help` to see the usage):

```
-h, --help            show this help message and exit
-run_type 		      train, test or predict
-train_type			  dev: origin train data -> train data + dev data; all: train on origin train data
-dev_size 			  the number(>= 1) or percent(< 1) of dev
-save_model_path	  save the good model.pth path
-text              	  input text when run dev
-image                input image when run dev
-epoch EPOCH          train epoch num
-batch_size			  batch size number
-lr                   learning rate
```

We provide 4 different `operation modes`. You can choose one according to the following description.

- `train`(default)

  - `dev`(default)

    In this operation modes,the code will **divide original train data into train data and dev data**,then model will be **trained on train data and show the result on dev data**.Finally,the code will save the best model in `/model/best-model.pth`.

    To run this operation modes,you can run following script on cmd:

    ```
    python main.py -run_type train -train_type dev -dev_size 511
    ```

  - `all`

    In this operation modes,the code will **train the model on all original train data**.Finally,the code will save the best model in `/model/best_model_with_all_train.pth`.

    To run this operation modes,you can run following script on cmd:

    ```
    python main.py -run_type train -train_type all
    ```

- `test`

  In this operation modes,the code will **show the result on dev data** with the model saved in `/model/best-model.pth`.You can **conduct ablation experiments** on this basis by reducing args `-text` and `-image`.

  To run this operation modes with both text and image input,you can run following script on cmd:

  ```
  python main.py -run_type test -text -image
  ```

  To run this operation modes with just text input,you can run following script on cmd:

  ```
  python main.py -run_type test -text
  ```

  To run this operation modes with just image input,you can run following script on cmd:

  ```
  python main.py -run_type test -image
  ```

- `predict`

  In this operation modes,the code will **predict the test data** with the model saved in `/model/best_model_with_all_train.pth`,and save the results in `/data/result.txt`.

  To run this operation modes,you can run following script on cmd:

  ```
  python main.py -run_type predict
  ```

**We provide the above files or models obtained during the lab, so we can directly run all the above operation modes**

**Because the model exceeds the GitHub size limit and cannot be uploaded, please get the file through the following link and put it into the corresponding folder according to the Repository structure**

```
link:https://pan.baidu.com/s/1F7MkNYPyet4b4DFvx2_61w 
code: dase
```

## Attribution

Parts of this code are based on the following repositories:

- [CLMLF](https://github.com/Link-Li/CLMLF)