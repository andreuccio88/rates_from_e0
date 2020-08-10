write.csv2(matr.val,file = "RUS_M_REAL.csv")

write.csv2(matr.lc,file = "RUS_M_LC.csv")

write.csv2(matr.NN,file = "RUS_M_NN.csv")

write.csv2(ll_pred,file = "RUS_M_ll.csv")

write.csv2(ll_POIS_pred,file = "RUS_M_ll_POIS.csv")


plot(matr.val[,5])
lines(matr.NN[,5],col="red")
lines(matr.lc[,5],col="black")
lines(ll_pred[,5],col="blue")
lines(ll_POIS_pred[,5],col="blue")
lines(matr.val_predi[,5])

plot(matr.NN[,1])
lines(matr.NN[,1],col="red")
lines(matr.NN[,9],col="blue")

plot(matr.val[,1])
lines(matr.val[,1],col="red")
lines(matr.val[,9],col="blue")



plot(matr.NN[,1]-matr.NN[,6])
