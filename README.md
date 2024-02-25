
### TRAPOD Architechture

We have developed a transformer-based model to predict postoperative delirium (POD) in the recovery room on the basis of intraoperative time series namely TRAPOD. In contrast to LSTM and MLP based models, this architechture performed best. 
Please find the corresponding paper liked to this repository. We asessed delirium as a binary variable with POD 1 and no POD 0, based on the nursing delirium assessment scale (Nu-DESC). Other end point definitions like the confusion assessement metric (CAM)
might be available at other centers. In the appendix B of the original paper, you find information about the cohort definition, features, and preprocessing steps. We are open for any collaboration to validate our model in different medical centers. 


### Model access

Please find the trained models under /models, implemented baselines that were developed for POD are integrated into the /model_baselines folder. You can load the model with this command replacing XXXX with the models name:

<script>
  model = torch.load("./models/XXXX.pt")
</script>
