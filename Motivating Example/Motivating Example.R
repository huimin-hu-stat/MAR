

### Likelihood test!!
library(missForest)
library(mice)
library(Rfast)
library(MCMCpack)
library(mice)
library(pracma)  # or use optim with method = "L-BFGS-B"


source("helper.R")
source("Density_Estimation_Final.R")


methods <- c("missForest", "mean")

#methods <- c( "missForest","mice_cart")

n<-5000
d<-3
delta<-0.01

meanestimateMLE<-list()
meanestimate<-list()

Resultsquantile<-list()
Resultsquantile1<-list()
Resultsquantile2<-list()
Resultsquantile3<-list()
imputationlist<-list()

nrep.total<-50

for (s in (1:nrep.total)){
  
  set.seed(1+s)
  
  X<- Rfast::rmvnorm(n, mu = rep(0,d), sigma =  matrix(c(1, 0.7, 0,0.7,1,0, 0,0,1), 3, 3, byrow=T))
  
  
  vectors <- matrix(c(
    rep(0, d),
    0, 1, rep(0,d-2),
    1, rep(0,d-1)
  ), nrow = 3, byrow = TRUE)
  
  M <- vectors[apply(X,1, function(x) sample(1:3, size = 1, prob=c( ( max(pnorm(x[1]),2*delta)+max(pnorm(x[2]),2*delta))/3, (2-max(pnorm(x[1]),2*delta))/3, (1-max(pnorm(x[2]),2*delta))/3), replace = TRUE)), ]
  
  X.NA<-X
  X.NA[M==1]<-NA
  
  
  colnames(X)<-NULL
  colnames(X)<-paste0("X",1:d)
  colnames(X.NA)<-paste0("X",1:d)
  
  
  imputations<-list()
  
  if ("knn" %in% methods){  imputations[["knn"]]<-impute.knn(as.matrix(X.NA))$data}
  if ("missForest" %in% methods){imputations[["missForest"]]<-missForest(X.NA)$ximp}
  if ("mice_cart" %in% methods){  blub <- mice(X.NA, method = "cart", m = 1)
  imputations[["mice_cart"]]<-mice::complete(blub, action="all")[[1]]}
  if ("mice_rf" %in% methods){  blub <- mice(X.NA, method = "rf", m = 1)
  imputations[["mice_rf"]]<-mice::complete(blub, action="all")[[1]]}
  
  if ("mean" %in% methods){  blub <- mice(X.NA, method = "mean", m = 1)
  imputations[["mean"]]<-mice::complete(blub, action="all")[[1]]}
  
  if ("mice_drf" %in% methods){  blub <- mice(X.NA, method = "DRF", m = 1)
  imputations[["mice_drf"]]<-mice::complete(blub, action="all")[[1]]}
  if ("mice_norm.nob" %in% methods){  blub <- mice(X.NA, method = "norm.nob", m = 1)
  imputations[["mice_norm.nob"]]<-mice::complete(blub, action="all")[[1]]}
  
  
  meanval<-rep(0, length(methods))
  corval<-rep(0, length(methods))
  varval<-rep(0, length(methods))
  quantileval<-rep(0, length(methods))
  
  names(meanval)<-methods
  names(corval)<-methods
  names(varval)<-methods
  names(quantileval)<-methods
  result <- optimize_mu_Sigma_constrained(X.NA)
  
  meanestimateMLE[[s]]<-result$mu_hat
  meanestimate[[s]]<-colMeans(X.NA, na.rm = T)
  
  meanval["MLE"]<-result$mu_hat["X1"]
  meanval["Mest"]<-mean(X.NA[,1], na.rm=T)
  
  corval["MLE"]<-result$Sigma_hat[2,1]
  corval["Mest"]<-cor(X.NA[!is.na(X.NA[,1])& !is.na(X.NA[,2]),1:2])[1,2]
  
  varval["MLE"]<-result$Sigma_hat[1,1]
  varval["Mest"]<-var(X.NA[!is.na(X.NA[,1]),1])
  
  quantileval["MLE"] <- qnorm(0.1, mean=meanval["MLE"], sd = sqrt(varval["MLE"]))
  quantileval["Mest"] <- quantile(X.NA[!is.na(X.NA[,1]),1], probs=0.1)
  
  
  for (method in c(methods)){
    
    
    Ximp<-imputations[[method]]
    
    colnames(Ximp)<-paste0("X",1:ncol(X))
    meanval[method]<-mean(Ximp[,1])
    corval[method]<-cor(Ximp)[1,2]
    varval[method]<-var(Ximp[,1])
    quantileval[method]<-quantile(Ximp[,1], probs=0.1)
  }
  
  
  Resultsquantile[[s]] <- meanval
  Resultsquantile1[[s]] <- corval
  Resultsquantile2[[s]] <- varval
  Resultsquantile3[[s]] <- quantileval
  
  imputationlist[[s]]<-imputations
  
  
  print(paste0("nrep ",s, " out of ", nrep.total ))
  
  
}



png(filename = "Quantile_Corr_Estimate.png",
    width = 1700,    # Width in pixels
    height = 500,    # Height in pixels
    res = 120)       # Resolution in dpi

par(mfrow=c(1,2))

truth<-qnorm(0.1)

tmp<-Resultsquantile3

## Setup
quantiledata<-t(sapply(1:length(tmp), function(j) tmp[[j]]))
quantiledatamtruth<-abs(quantiledata-truth)

meanvalsquantiles<-colMeans(quantiledatamtruth)

boxplot(quantiledata[,],,cex.axis=1.5,cex.lab=1.5) #quantiledata[,order(meanvalsquantiles, decreasing = T)]
abline(h=truth, col="blue", lty=2)


# # Close the PNG device
# dev.off()
# 
# 
# png(filename = "Var_Estimate.png",
#     width = 1700,    # Width in pixels
#     height = 800,    # Height in pixels
#     res = 120)       # Resolution in dpi



truth<-0.7

tmp<-Resultsquantile1

## Setup
quantiledata<-t(sapply(1:length(tmp), function(j) tmp[[j]]))
quantiledatamtruth<-abs(quantiledata-truth)

meanvalsquantiles<-colMeans(quantiledatamtruth)

boxplot(quantiledata,,cex.axis=1.5,cex.lab=1.5) #quantiledata[,order(meanvalsquantiles, decreasing = T)]
abline(h=truth, col="blue", lty=2)


# Close the PNG device
dev.off()

