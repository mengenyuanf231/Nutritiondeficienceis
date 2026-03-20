# Title: DSLNDD-Net: A Multi-Scale Edge-Aware Attention Network for Nutrient Deficiency Detection in Dense Strawberry Leaves
### block.py: the code of model 
### loss.py,metrics.py: DSUIoU
### two69-all.yaml: the code for model construction
## Train
To train the gan model by yourself, please: 
1. Download the ```'dataset.zip'``` following the above ```Dataset``` section and unzip the ```"datasets/"``` to the root of this repo.
2. Check the configuration
3. Train the model by running:
    ```
    python train.py
    ```
## Test
To generate synthetic dataset on your own pc, simply check the arguments in test.py and run:
   ```
   python test.py
   ```

## Data availability
>>The raw data, among the labels are available upon reasonable request in(https://drive.google.com/drive/folders/1zBDmxUvISoSmLUKmp5pH7-DAOyEVliKZ?usp=drive_link))
>>weight(https://drive.google.com/drive/folders/1v1X2GWuixjs3Yw7hXA8t8CkVQxaANol-?usp=drive_link)

