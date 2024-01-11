```
# Algorithm Description of Team fanyuming2023 for AAAI2024 COMPETITION ON MATH PROBLEM SOLVING - TRACK1 and TRACK2

Our solution involves a series of steps from data collection, data preprocessing, model training, to inference.

## Data Collection

We collected multiple datasets including ape210k, exeq300k, math23k, math, gsm8k, schoolmath, and the TAL-SAQ5K dataset released by the competition organizers.

## Data Preprocessing

The `code/process_json.py` script was used to process the ape210k, gsm8k, and math23k datasets into the input-output format for fine-tuning. As these datasets directly provide the final answers, we can directly use the answers as output for processing.

> Note: We did try to use thinking-process data for model development, but the model didn't perform well possibly due to limitations in capabilities when trained with these datasets. Therefore, we did not continue to use this method for fine-tuning in the following stages.

For the math dataset, which is in English, we utilized the script `LLM_translate_and_thinking_toans.py` to translate it and extract all question-answer pairs from all the subfolders of the math dataset, the output being stored in `/aaai2024comp/math_dataset/MATH/train.json` and `test.json`. 
Specifically, we used the `qwen14b` model to first perform the translation task, and then fed it with a prompt like `'Question: '+question+'\nProblem-solving process: '+thinking+'\nThe final answer to the above problem based on the problem-solving process will be? Please return the answer directly in the form of a floating-point number.'` to make it only return the final answer, thereby unifying the dataset with the previous ones. 
The processed data is saved as `train_dataset/math_test_summ.json` and `train_dataset/math_train_summ.json`.

This concludes the data selection stage.

## Model Training

We adapted the `qwen14b-chat` model as our base model and fine-tuned it using instructions. The training framework utilized was `code/LLaMA-Efficient-Tuning`, with specific hyperparameters that can be found in `code/LLaMA-Efficient-Tuning-1031/qwen-full-14b-math.sh`.

> Special thanks to the open source contributor [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Inference

For the inference stage, we utilized the `code/LLM_get_final_answer.py` code. 
To tackle the issue regarding model outputs that might contain Chinese characters, English characters, fraction symbols, etc., which couldn't be directly converted into floating-point numbers, we designed a regular expression matching function to translate these into the final output, as shown in the code.

you can replace the path in the code to get the Track1 or Track2 result.

The math datasets we gathered and the processed training datasets can be downloaded from the following links:

Collected Data: [https://cloud.189.cn/web/share?code=YnMj2uJJz6Nb](https://cloud.189.cn/web/share?code=YnMj2uJJz6Nb) (Access Code: y286)
Processed Training Data: [https://cloud.189.cn/web/share?code=iYVzyaUfA3ua](https://cloud.189.cn/web/share?code=iYVzyaUfA3ua) (Access Code: 5h0o)

Special thanks to the competition organizers for their hard work.
```