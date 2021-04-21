## Finetune Bert LM Head

Based on experiment assumption, we are given finetuned Bert models on various tasks. However, these models have classifiers different than a LM Head. To make these models suitable for later data generation, we take off the original model classfier, initialize a LM Head on the model, freeze the encoder weights, and then finetune LM Head on Wikipedia dataset.

Directory of original Bert model: ../Berts/sst_base_l12/
Directory to put the new model with LM Head: Berts_LM/sst_1ep/

Path of operation: /rscratch/bohan/ZQBert/zero-shot-qbert/direct_generate

```bash
python train_head.py --model_name_or_path ../Berts/sst_base_l12/ \
--output_dir Berts_LM/sst_1ep/ \
--do_train \
--num_train_epochs 1 \
--max_seq_length 512
```

## Generate Dataset

Second step is to use the above model to generate dataset. The new model with LM Head is used to generate sentences. The original Bert model is used to generate labels. You can adjust batch_num and batch_size to change the size of dataset.

Directory of original Bert model: ../Berts/sst_base_l12/
Directory to put the new model with LM Head: Berts_LM/sst_1ep/
Directory to put generated dataset: gen_data/sst/
Name of new generated data: sst_ep1_8700.tsv

Path of operation: /rscratch/bohan/ZQBert/zero-shot-qbert/direct_generate

```bash
python direct_generate.py --LM_model Berts_LM/sst_1ep/ \
--TA_model ../Berts/sst_base_l12/ \
--output_dir gen_data/sst/ \
--file_name sst_ep1_8700 \
--batch_num 87 \
--batch_size 100
```

## Quantize Model

After generating dataset, we quantize the original Bert model.

Directory of original Bert model: Berts/sst_base_l12/
Directory to put generated dataset: direct_generate/gen_data/sst/
Name of train data: sst_ep1_8700.tsv
Name of val data: sst_1000.tsv
Directory to put quantized model: results/sst_ep1_8700/

Path of operation: /rscratch/bohan/ZQBert/zero-shot-qbert

```bash
python quant_run_glue.py --task_name SST-2 \
--do_lower_case \
--data_dir direct_generate/gen_data/sst/ \
--train_name sst_ep1_8700 \
--val_name sst_1000 \
--model Berts/sst_base_l12/ \
--learning_rate 2e-5 \
--weight_bit 4 \
--activation_bit 4 \
--output_dir results/sst_ep1_8700/
```

## Evaluation

After quantization, we can evaluate on the real data to see the performance.

Directory to put the real data: ../GLUE-baselines/glue_data/SST-2/
Directory to put quantized model: results/sst_ep1_8700/
Fake empty directory to get the code running: empty

Path of operation: /rscratch/bohan/ZQBert/zero-shot-qbert

``bash
python run_glue_old.py --task_name SST-2 \
--do_eval True \
--do_lower_case \
--data_dir ../GLUE-baselines/glue_data/SST-2/ \
--model results/sst_ep1_8700/ \
--output_dir empty
