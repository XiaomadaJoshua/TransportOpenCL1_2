library(matlab)
library(plot3D)
remove(list = ls())
#index <- c(0:299)*51*51+25*51+26
###################################################
## 0.2 by 0.2 beam check
PF200MGH <- readBin(con = "RunB8_200MeV/PrimaryFluence.bin", what = "int", n = 51*51*300, size = 4, endian = "big")
SF200MGH <- readBin(con = "RunB8_200MeV/SecondaryFluence.bin", what = "int", n = 51*51*300, size = 4, endian = "big")
PD200MGH <- readBin(con = "RunB8_200MeV/LETDosePrimary.bin", what = "integer", n = 51*51*300, size = 4, endian = "big")
SD200MGH <- readBin(con = "RunB8_200MeV/LETDoseSecondary.bin", what = "int", n = 51*51*300, size = 4, endian = "big")


PF200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/primaryFluence.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
SF200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/secondaryFluence.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
PD200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/primaryDose.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
SD200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/secondaryDose.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")

doseScale <- (sum(PD200 + SD200))/(sum(PD200MGH + SD200MGH))


PF200MGH <- reshape(as.matrix(PF200MGH), c(51, 51, 300))
SF200MGH <- reshape(as.matrix(SF200MGH), c(51, 51, 300))
PD200MGH <- reshape(as.matrix(PD200MGH), c(51, 51, 300))
SD200MGH <- reshape(as.matrix(SD200MGH), c(51, 51, 300))

PF200 <- reshape(as.matrix(PF200), c(51, 51, 300))
SF200 <- reshape(as.matrix(SF200), c(51, 51, 300))
PD200 <- reshape(as.matrix(PD200), c(51, 51, 300))
SD200 <- reshape(as.matrix(SD200), c(51, 51, 300))



fluenceScale <- round(max(PF200)/max(PF200MGH), 1)
PF200MGH <- fluenceScale * PF200MGH
plot(PF200[26,26, 1:300], main = "central beam fluence, 200MeV,", yaxt="n", xlab = "depth in mm", ylab = "primary fluence", type = 'l', col = 'red')
lines(PF200MGH[26, 26, 300:1])

SF200MGH <- fluenceScale * SF200MGH
plot(SF200[26, 26, 1:300], main = "central beam fluence, 200MeV", yaxt="n", xlab = "depth in mm", ylab = "secondary fluence", type = 'l', col = 'red')
lines(SF200MGH[26, 26, 300:1])




PD200MGH <- doseScale * PD200MGH
plot(PD200[26, 26, 1:300], main = "central beam dose, 200MeV,", yaxt="n", xlab = "depth in mm", ylab = "primary dose", type = 'l', col = 'red')
lines(PD200MGH[26, 26, 300:1])

SD200MGH <- doseScale * SD200MGH
plot(SD200[26, 26, 1:300], main = "central beam dose, 200MeV", yaxt="n", xlab = "depth in mm", ylab = "secondary dose", type = 'l', col = 'red')
lines(SD200MGH[26, 26, 300:1])

######################################################
## pencil beam total dose check
library(matlab)
library(plot3D)
remove(list = ls())
TD200MGH <- read.csv(file = "topasTotalDose/Scoring_200M.csv", sep = ",", fill = T, header = F)
TD200MGH <- TD200MGH$V4
TD200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/pencilBeam200MeV/totalDose.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")

doseScale <- sum(TD200)/sum(TD200MGH)
TD200MGH <- doseScale * TD200MGH

TD200MGH <- reshape(as.matrix(TD200MGH), c(300, 51, 51))
TD200 <- reshape(as.matrix(TD200), c(51, 51, 300))

plot(TD200[26, 26, 1:300], main = "central beam dose, 200MeV", xlab = "depth in mm", ylab = "total dose", type = 'l', col = 'red')
lines(TD200MGH[1:300, 26, 26])

plot(TD200[1:51, 20, 10], type = 'l', col = 'red')
lines(TD200MGH[10, 1:51, 20])



#####################################
# primary enrance fluence debug
library(matlab)
library(plot3D)
remove(list = ls())
PF200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/pencilBeam200MeV/primaryFluence.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
PF200 <- reshape(as.matrix(PF200), c(51, 51, 300))
PF200[26, 26, 1]
debug <- read.table("C:/Users/S158879/workspace/GitHub/TransportOpenCL1_2/CLTransport/debug.txt", quote="\"")

sum(debug$V3)/0.004

test <- rep(T, times = length(debug$v3))
temp <- debug$V3[1]
for(i in 1:length(debug$V3)){
  #temp = temp + debug$V3[i+1]
  test[i] <- abs(debug$V3[i+1] + debug$V4[i] - debug$V4[i+1]) < debug$V4[i+1]*1e-4
}


which(!test)

