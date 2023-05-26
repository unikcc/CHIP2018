# Ping An Technology Medical Text Matching Competition

## Model Description
+ Based on a slightly modified ESIM, only character-level information is used.
+ No feature engineering is used except for pre-trained character-level word embeddings.
+ Unfortunately, due to limited resources, only a single model was built with 10-fold local cross-validation.

## Running the Program
+ mkdir data/pingan
+ Place the dataset under ./data/pingan, e.g., ./data/pingan/char_embedding.csv.
+ cd scripts
+ (Optional) pip install requirements.txt
+ chmod +x step.sh
+ Run ./step.sh to execute the entire process, including preprocessing, training, validation, and prediction.

## Others
+ **Believe your local cv!**
