# Renata Ramos
# Data 20/04/2023
# Neural networks for airfoil self noise with cross-validation

# Preambulo
cat("\014")
graphics.off()
rm(list=ls())


# ler arquivo
dados <- read.table("airfoil_self_noise.dat", header = F)

# numero de neuronios
nmax <- 40
nmin <- 30
#########################################################################
# Variavel de saida
out <- "Scaled sound pressure level"


#intervalos da rede
int.V1 <- c(min(dados$V1), max(dados$V1))
int.V2 <- c(min(dados$V2), max(dados$V2))
int.V3 <- c(min(dados$V3), max(dados$V3))
int.V4 <- c(min(dados$V4), max(dados$V4))
int.V5 <- c(min(dados$V5), max(dados$V5))
int.V6 <- c(min(dados$V6), max(dados$V6))


# normalizacao de 0 a 1
dadosN <- dados
for(i in 1:ncol(dados)){
  dadosN[,i] <- (dados[,i] - min(dados[,i]))/(max(dados[,i])-min(dados[,i]))
}

# Funcao de ativacao
relu <- function(x) max(0,x)
sigmoid <- function(x) 1/(1+exp(-x)) #same as logistic ?
softplus <- function(x) log(1+exp(2*x))

# Redes Neurais
library(neuralnet)

# # Visualizacao dos dados de treino e teste
# plot(rownames(training),training$y,ylab= out)
# points(rownames(testing),testing$y,pch = 16,col=2)
# legend("topright",legend=c("training","testing"),col=c(1,2),pch=c(1,16))
  
  
# Formula: dependencia das variaveis
allVars <- colnames(dadosN)
predictorVars <- allVars[-ncol(dados)]
predictorVars <- paste(predictorVars,collapse='+')
form <- as.formula(paste('V6 ~',predictorVars,collapse='+'))
  
# particao randomica dos dados
ind <- sample(5,nrow(dadosN), replace = TRUE, prob = c(0.2,0.2,0.2,0.2,0.2))

# vetor do erro por num de neuronios
error.min.neur <- numeric(nmax-nmin +1)

#contadores
ctotal <- 5*(nmax-nmin+1)
contador <- 1

error.min <- 100

# definicao do melhor num de neur
for(neuronios in nmin:nmax){
  
  #vetor do erro - cross-validation
  error.cross <- numeric(5)

    for(k in 1:5){
    
      # escolha da particao
      training <- dadosN[(ind!=k),]
      testing <- dadosN[(ind==k),]
    
      # aplicacao das redes
      n <- neuralnet(formula = form, data = training, hidden = neuronios, err.fct = "sse", rep = 5,
               act.fct = "logistic", linear.output = T, lifesign = "minimal", threshold = 0.05)

    
      # Comparacao de erros do test set
      # predicao
      p.test <- compute(n,testing[,-ncol(testing)])
      error.cross[k] <- sum((testing$y - p.test$net.result)^2)/nrow(testing)
      
      #Contador
      print(paste("Iteracao: ", contador, "/", ctotal ))
      contador <- contador + 1
      
    }
    
    # erro por num de neuronios
    error.min.neur[i] <- mean(error.cross)

    # comparacao para inicializacoes
    if(error.min.neur[i] < error.min){
      #net.op <- n
      hd <- neuronios
      error.min <- error.min.neur[i]
    }
  
}  
 
# plot do erro do cross para o num. de neuronios
jpeg(file=paste("Erro -", out, "from", paste(names(dados)[-ncol(dados)], collapse = " "),".jpeg"))
plot(seq(nmin,nmax), error.min.neur, type = "b", main = "Erro por Num de Neuronios", xlab = "Num de Neuronios", ylab = paste("Erro - ", out))
dev.off()


#### Treino final
print("--------------------Treino Final das Redes----------------------")


  # particao randomica dos dados
  ind <- sample(2,nrow(dadosN), replace = TRUE, prob = c(0.8,0.2))
  
  # definicao training e test set
  training <- dadosN[(ind!=2),]
  testing <- dadosN[(ind==2),]
  
  # treino das redes
  n <- neuralnet(formula = form, data = training, hidden = hd, err.fct = "sse", 
                 act.fct = "logistic", linear.output = T, lifesign = "full", threshold = 0.005)
  
  # Comparacao de erros do test set
  # predicao
  p.test <- compute(n,testing[,-ncol(testing)])
  error.min <- sum((testing$V6 - p.test$net.result)^2)/nrow(testing)
  net.op <- n
 
   
   
  
  ######## Resultados Finais e Plot
  # Predicoes
  p.train<-compute(net.op,training[,-ncol(training)])
  p.test<-compute(net.op,testing[,-ncol(testing)])
  
  
  # Desnormalizacoo das predicoes
  p.train.unscaled<-p.train$net.result*(max(dados$V6)-min(dados$V6))+min(dados$V6)
  training.unscaled<-training$V6*(max(dados$V6)-min(dados$V6))+min(dados$V6)
  
  p.test.unscaled <- p.test$net.result*(max(dados$V6)-min(dados$V6))+min(dados$V6)
  testing.unscaled <- testing$y*(max(dados$V6)-min(dados$V6))+min(dados$V6)
  
  # Mean Squared Error
  MSE.train<-sum((training.unscaled-p.train.unscaled)^2)/nrow(training)
  MSE.test<-sum((testing.unscaled-p.test.unscaled)^2)/nrow(testing)
  
  
  # Calculo do coeficiente de determinacao
  SQtot.train<-sum((training$V6-mean(training$V6))^2)
  SQres.train<-sum((training$V6-p.train$net.result)^2)
  R2.train<-1-SQres.train/SQtot.train
  R2.train
  
  SQtot.test<-sum((testing$V6-mean(testing$V6))^2)
  SQres.test<-sum((testing$V6-p.test$net.result)^2)
  R2.test<-1-SQres.test/SQtot.test
  R2.test
  

  # Visualizacao das previsoes
  y.train <- training.unscaled
  ycalc.train <- p.train.unscaled
  y.test <- testing.unscaled
  ycalc.test <- p.test.unscaled
  
  #treino
  jpeg(file=paste("Predicoes -", out, "from", paste(names(dados)[-ncol(dados)], collapse = " "),".jpeg"))
  plot(y.train,ycalc.train,xlab= "y actual",ylab="y calc", 
       main = paste(out, " - ", hd, "neurons"), pch = 20)
  abline(0,1,lty=1) #linha de 45 graus

  R2.train.plot<-round(R2.train,digits=3)
  text(x = min(y.train)+(max(y.train)-min(y.train))/4, 
       y = max(y.train)-(max(y.train)-min(y.train))/4,
       label=bquote(R^2 ==.(R2.train.plot)))
  
  #test
  points(y.test, ycalc.test, col = 2, pch = 16, cex = 0.9)
  R2.test.plot <- round(R2.test,digits=3) 
  text(x = min(y.train)+(max(y.train)-min(y.train))/4, 
       y = max(y.train)-(max(y.train)-min(y.train))/3.2,
       label=bquote(R^2 ==.(R2.test.plot)), col = 2)
  
  legend(x = max(y.test)-(max(y.test)-min(y.test))/4,
         y = min(y.test)+(max(y.test)-min(y.test))/4,
         legend=c("Training", "Testing"),
         col=c(1, 2), pch=16, cex=0.8)
  dev.off()

  # # pesos
  # w1 <- net.op$weights[[1]][[1]]
  # w2 <- net.op$weights[[1]][[2]]
  # write.csv(w1,file = paste("w1", out, "from", paste(names(dados)[-ncol(dados)], collapse = " "), ".csv"))
  # write.csv(w2,file = paste("w2", out, "from", paste(names(dados)[-ncol(dados)], collapse = " "), ".csv"))
  



