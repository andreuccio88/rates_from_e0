# Error functions 
rmse = function (truth, prediction)  {
  sqrt(mean((prediction - truth)^2))
}
mae = function(truth, prediction){
  mean(abs(prediction-truth))
}