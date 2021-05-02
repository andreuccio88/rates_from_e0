# Error functions 
rmse = function (truth, prediction)  {
  sqrt(mean((prediction - truth)^2))
}
mae = function(truth, prediction){
  mean(abs(prediction-truth))
}

# Scale
scale_min_max = function(dat,dat_test)  {
  min_dat = min(dat)
  max_dat = max(dat)
  dat_scaled=(dat-min_dat)/(max_dat-min_dat)
  dat_scaled_test = (dat_test-min_dat)/(max_dat-min_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, min = min_dat, max=max_dat))
}

scale_z = function(dat,dat_test)  {
  mean_dat = mean(dat)
  sd_dat = sd(dat)
  dat_scaled=(dat-mean_dat)/(sd_dat)
  dat_scaled_test = (dat_test-mean_dat)/(sd_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, mean_dat = mean_dat, sd_dat=sd_dat))
}

# Model

DNN_Multipop = function(train,test)  {
  
  #scale mx
  scaled = scale_min_max(train$logmx, test$logmx)
  train$mx_scale = scaled$train
  test$mx_scale = scaled$test
  
  #scale e0
  scaled2 = scale_min_max(train$ex, test$ex)
  train$e0_scale = scaled2$train
  test$e0_scale = scaled2$test
  
  #Regression
  train_reg = train[,c(2,4,8,9,10,11),with=F]
  test_reg = test[,c(2,4,8,9,10,11),with=F]
  year_scale = scale_min_max(train_reg$Year,test_reg$Year)
  train_reg$Year = year_scale[[1]]
  test_reg$Year = year_scale[[2]]
  
  #train
  x = list(Year      = train_reg$Year,
           Age = train_reg$Age, Country = train_reg$Country_fact, Sex=train_reg$Sex_fact, e0=train_reg$e0_scale)
  y = (main_output= train_reg$mx_scale)
  
  #test
  x_test = list(Year      = test_reg$Year,
                Age = test_reg$Age, Country = test_reg$Country_fact, Sex=test_reg$Sex_fact, e0=test_reg$e0_scale)
  y_test = (main_output= test_reg$mx_scale)
  
  require(keras)
  use_session_with_seed(1)
  # Embedding layers
  e0 <- layer_input(shape = c(1), dtype = 'float32', name = 'e0')
  Year <- layer_input(shape = c(1), dtype = 'float32', name = 'Year')
  Age <- layer_input(shape = c(1), dtype = 'int32', name = 'Age')
  Country <- layer_input(shape = c(1), dtype = 'int32', name = 'Country')
  Sex <- layer_input(shape = c(1), dtype = 'int32', name = 'Sex')
  
  Age_embed = Age %>% 
    layer_embedding(input_dim = 101, output_dim = 5,input_length = 1, name = 'Age_embed') %>%
    keras::layer_flatten()
  
  Sex_embed = Sex %>% 
    layer_embedding(input_dim = 2, output_dim = 5,input_length = 1, name = 'Sex_embed') %>%
    keras::layer_flatten()
  
  Country_embed = Country %>% 
    layer_embedding(input_dim = 49, output_dim = 5,input_length = 1, name = 'Country_embed') %>%
    keras::layer_flatten()
  
  main_output <- layer_concatenate(list(e0,Year,Age_embed,Sex_embed,Country_embed
  )) %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(0.10) %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(0.10) %>% 
    layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')
  
  model <- keras_model(
    inputs = c(Year,Age,Country,Sex,e0), 
    outputs = c(main_output))
  
  adam = optimizer_adam(lr=0.0005)
  lr_callback = callback_reduce_lr_on_plateau(factor=.80, patience = 5, verbose=1, cooldown = 5, min_lr = 0.00005)
  model_callback = callback_model_checkpoint(filepath = "best.mod", verbose = 1,save_best_only = TRUE)
  
  model %>% compile(
    optimizer = adam,
    loss = "mse")
  
  fit = model %>% fit(
    x = x,
    y = y, 
    epochs = 300,
    batch_size =  682.7,verbose = 1, shuffle = T, validation_split = 0.1, callbacks = list(lr_callback,model_callback))
  
  model = load_model_hdf5("best.mod")
  test$mx_DNN <-  model %>% predict(x_test)
  test$mx_DNN <- exp(test$mx_DNN*(scaled$max-scaled$min)+scaled$min)
  
return(test)
}