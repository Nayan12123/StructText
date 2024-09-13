English

## _StrucTexT_: Structured Text Understanding with Multi-Modal Transformers

- [Architecture](#architecture)
- [Quick Experience](#quick-experience)
  * [Install PaddlePaddle and create the environment](#install-paddlepaddle)
  * [Download Base and Trained models](#download-inference-models)
  * [Infer Fine-tuned Models](#infer-fine-tuned-models)
- [Citation](#citation)


## Architecture
![structext_arch](./doc/structext_arch.png#pic_center)
<p align="center"> Model Architecture of StrucTexT </p>



### Segment-based Entity Labeling
   * datasets.
      * [FUNSD](https://guillaumejaume.github.io/FUNSD/) is a form understanding benchmark with 199 real, fully annotated, scanned form images, such as marketing, advertising, and scientific reports, which is split into 149 training samples and 50 testing samples. FUNSD dataset is suitable for a variety of tasks and we force on the sub-tasks of **T-ELB** and **S-ELK** for the semantic entity `question, answer and header`.
 

## Quick Experience
### Install PaddlePaddle
This code base needs to be executed on the `PaddlePaddle 2.1.0+` on environments of `Python 3.6+` . You can find how to prepare the environment from this [paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick) or use pip:

```bash
# We only support the evaluation on GPU by using PaddlePaddle, the installation command follows:
conda env create -f environment.yml
conda activate structext_env
# Navigate to v1 folder
cd v1
```



### Download pretrained model 

| Model link                                         |  Params(M)  |
| :------------------------------------------------- | :---------: |
| <a href="https://drive.google.com/file/d/19LGqUJFqwzY5XVp7rjA_T1JEHiVjkhCu/view?usp=sharing">StrucTexT Base Pretrained Model weights</a>       | 181 |

### Trained model paths

| Model link                                         |  Params(M)  |
| :------------------------------------------------- | :---------: |
| <a href="https://drive.google.com/file/d/1INBVSOZQg7jyxeyzX33CJVqCl4nYYgu3/view?usp=sharing">StrucTexT Base Finetuned for Segment Entity Labelling</a>       | 181 |
| <a href="https://drive.google.com/file/d/1CGp3iZ0p1KKrXnNbRfn-V_xIeaRCLfYl/view?usp=sharing" target="_blank">StrucTexT Base Finetuned for Segment Entity Linking</a>       | 181 |



### Train Base model
   * Segment-based ELB task on FUNSD

```python
# 1. download and extract the FUNSD dataset at <funsd_folder>. In the current repo it has already been downloaded in dataset folder
# 2. download the model: segment_labelling_trained_model.pdparams
# 3. generate the eval dataset form.
python data/make_funsd_data.py \
    --config_file ./configs/base/labeling_funsd.json \
    --label_dir ./dataset/testing_data/annotations/ \
    --out_dir ./dataset/testing_data/test_labels/
# 4. Train the model 
python training_scripts/labeling_segment/train_seg_labelling.py \
    --config_file "./configs/base/labeling_funsd.json" \
    --label_path "./dataset/training_data/train_labels" \
    --image_path "./dataset/training_data/images" \
    --weights_path "../StrucTexT_base_pretrained.pdparams"
# 5. An optional parameter for gradient accumalation steps is also taken as an input By default it is 4. Which can be changed by attaching the below command with the above command
    --accumulation_steps <your grad acc step.>

# 6. evaluate the labeling task in the FUNSD dataset.
python ./tools/eval_infer.py \
    --config_file ./configs/base/labeling_funsd.json \
    --task_type labeling_segment \
    --label_path ./dataset/testing_data/test_labels/ \
    --image_path ./dataset/testing_data/images/ \
    --weights_path segment_labelling_trained_model.pdparams
```
   * Segment-based ELK task on FUNSD

```python
# 1. download and extract the FUNSD dataset at <funsd_folder>.
# 2. download the model: segment_labelling_trained_model.pdparams
# 3. generate the eval dataset form.
python data/make_funsd_data.py \
    --config_file ./configs/base/linking_funsd.json \
    --label_dir ./dataset/testing_data/annotations/ \
    --out_dir ./dataset/testing_data/test_labels/
# 4. Training the linking model execute the below given command.
python training_scripts/linking/train_linking.py \ 
    --config_file "./configs/base/linking_funsd.json"\
    --label_path "./dataset/training_data/train_labels"\
    --image_path "./dataset/training_data/images"\
    --weights_path "../StrucTexT_base_pretrained.pdparams"
# 5. An optional parameter for gradient accumalation steps is also taken as an input By default it is 4. Which can be changed by attaching the below command with the above command
    --accumulation_steps <your grad acc step.>

# 6. evaluate the linking task in the FUNSD dataset.
python ./tools/eval_infer.py \
    --config_file ./configs/base/linking_funsd.json \
    --task_type linking \
    --label_path ./dataset/testing_data/test_labels/ \
    --image_path ./dataset/testing_data/images/ \
    --weights_path entity_linking_trained_model.pdparams
```
