

Fit_DNN <- function(cnt,age,ys,yf,un,ep,yv){
  
  year <- ys:yf; l <- length(year); last <- yf+15
  
  # gender selection
  if(setting$gender == "M"){
    D <- df_m
  }else{
    D <- df_f
  }
  
  # regularization selection
   if(setting$reg == 0){
     typ <-  print_dot_callback <- callback_lambda(
       on_epoch_end = function(epoch, logs) {
         if (epoch %% 80 == 0) cat("\n")
         cat(".")
       })    
     
   }else{
     typ <- callback_early_stopping(monitor = "val_loss", min_delta = 0.0005,        
                                    patience = 100, verbose = 1, mode = 'min',
                                    baseline = NULL, restore_best_weights = TRUE)
     }
   
  
  ##################################
  ##                              ##
  ## Country and Period selection ##
  ##                              ##
  ##################################
  
  D <- D %>% mutate(log_mx=log(mx))
  B <- D %>%filter(year>=ys,year<=yf,age<=100,country==cnt) %>%  select(year,age,log_mx,ex)
  ex_ <- B %>% select(year,age,ex); mx_ <- B %>% select(year,age,log_mx)
  ex <- spread(ex_,age,ex);  mx <- spread(mx_,age,log_mx) 
  
  ##################################
  ##                              ##
  ##  Data partition for          ##
  ##  Deep Neural Network         ##
  ##                              ##
  ##################################
  
  set.seed(123)
  # Row sampling for train-test split
  smp=sample(1:nrow(ex),nrow(ex)/1.2)
  
  #TRAIN
  x_train <- as.matrix(ex[smp,c(1,2)]) #x_train
  y_train <- as.matrix(mx[smp,]) #y_train
 
  ##TEST
  x_test <- as.matrix(ex[-smp,c(1,2)]) #x_test
  y_test <- as.matrix(mx[-smp,]) #y_test
  
  # DELETE FIRST COLUMN: YEAR
  x_train <- x_train[,-c(1)]; y_train <- y_train[,-c(1)]
  x_test <- x_test[,-c(1)]; y_test <- y_test[,-c(1)]
  
  # Normalize training data
  x_train <- scale(x_train) 
  
  # Use means and standard deviations from training set to normalize test set
  col_means_train <- attr(x_train, "scaled:center") 
  col_stddevs_train <- attr(x_train, "scaled:scale")
  x_test <- scale(x_test, center = col_means_train, scale = col_stddevs_train)
  
  ##################################
  ##                              ##
  ##  DNN MODEL                   ##
  ##                              ##
  ##################################
  
  use_session_with_seed(42)
  build_model <- function() {
    
    model <- keras_model_sequential() %>%
      layer_dense(units =un, activation = 'relu',
                  input_shape = dim(x_train)[2]) %>%
      layer_dense(units = un, activation = "relu") %>%
      layer_dense(units = un, activation = "relu") %>%
      layer_dense(units = un, activation = "relu") %>%
      layer_dense(units = un, activation = "relu") %>%
      layer_dense(units = un, activation = "relu") %>%
      layer_dense(units = un, activation = "relu") %>%
      layer_dense(units = 101)
    
    model %>% compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(), 
      metrics = list("mean_absolute_error"))
    
    model}
  
  model <- build_model()
  model %>% summary()
  
  epochs <- ep
    history <- model %>% fit(
    x_train,
    y_train,
    epochs = epochs,
    batch_size = 1,
    validation_data = list(x_test, y_test),
    verbose = 2,
    callbacks = typ)
  
  plot(history, metrics = "mean_absolute_error", smooth = T) +
    coord_cartesian(ylim = c(0, 5))
  
  history
  test_predictions <- model %>% predict(x_test)
  test_predictions
  
  #########################################
  ##                                     ##
  ##  Backtesting                        ##
  ##                                     ##
  #########################################
  
  valid <- D %>% filter(year>yf&year<=yv,age<=100,country==cnt) %>%  select(year,age,log_mx,ex)
  ex_valid <- valid %>% select(year,age,ex);  mx_valid <- valid %>% select(year,age,log_mx)
  ex_val <- spread(ex_valid,age,ex); mx_val <- as.matrix(spread(mx_valid,age,log_mx))
  
  # e0 validation
  x_val <- as.matrix(ex_val[,c(2)])
  # e0 validation scaled
  x_val_sc <- scale(x_val, center = col_means_train, scale = col_stddevs_train)
  # NN prediction
  val_predictions <- model %>% predict(x_val_sc)
  
  # Smoothing
  year <- unique(valid$year)
  mx <- exp(val_predictions) #
  mx <- t(mx)
  a <- demogdata(mx,pop = mx,ages = age,years = year,type ="mortality",label = "prova",name="prova")
  smooth <- smooth.demogdata(extract.years(a,yf+1:last),method = "spline",weight = F,k = 80)
  
  
  #########################################
  ##                                     ##
  ##  Results                            ##
  ##                                     ##
  #########################################
  
  # Real data
  Real <- t(mx_val[,-1])
  # NN+Smoothing forecast 2006-last
  NN_s <- log(smooth$rate$prova[c(1:101),])
  # NN (NO Smoothing) forecast 2006-last
  NN <- t(val_predictions)
 
  return(list(NN_s,NN,Real))
  
}
