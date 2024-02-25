
### TRAPOD Architechture

We have developed a transformer-based model to predict postoperative delirium (POD) in the recovery room on the basis of intraoperative time series namely TRAPOD. In contrast to LSTM and MLP based models, this architechture performed best. 
Please find the corresponding paper liked to this repository. We asessed delirium as a binary variable with POD 1 and no POD 0, based on the nursing delirium assessment scale (Nu-DESC). Other end point definitions like the confusion assessement metric (CAM)
might be available at other centers. In the appendix B of the original paper, you find information about the cohort definition, features, and preprocessing steps. We are open for any collaboration to validate our model in different medical centers. 

![trans_architecture](https://github.com/ngiesa/TRAPOD/assets/35224961/c1e2f79f-7976-42ae-b26b-f4b70ed13e19)

The figure depicts the model architecture that is adapted from Vaswani et al. and described in more detail in the original paper. The input is of shape (#features) X (#time steps) X (# surgeries) where the latter dimension is equal to the batch size you want to ingest as testing samples. 

### Model Access

Please find the trained models under /models, implemented baselines that were developed for POD are integrated into the /model_baselines folder. You can load the model with this command replacing XXXX with the models name:

<code>
  model = torch.load("./models/XXXX.pt")
</code>

The TRAPOD architecture is based on a transformer model ingesting sequential data. Hence, you find the model under TRAN_SEQ in the /models folder. For testing the model, you need to preprocess 238 features converting clinical variables into the right units (see /metadata/variable_metadata.csv) and put them into the right order (see /metadata/feature_descriptives.xlsx). Features are converted to an equi-distance time grid using 3 min. sampling interval with the first intraoperative 30 minutes leading to 10 time steps. We used LOCF and mean imputation that are also derived for the z-standardization process for scaling features. We have used feature means from the training set that can also be read from the metadata files. Finally, if you have constructed a testing set M_test with dimensions 238x10xY with Y as the number of surgeries, you can apply the model with 

<code>
  y_pred = model(M_test)
</code>
