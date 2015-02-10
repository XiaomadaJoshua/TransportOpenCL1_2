library(matlab)
remove(list = ls())
#index <- c(0:299)*51*51+25*51+26

PF200MGH <- readBin(con = "RunB8_200MeV/PrimaryFluence.bin", what = "int", n = 51*51*300, size = 4, endian = "big")
SF200MGH <- readBin(con = "RunB8_200MeV/SecondaryFluence.bin", what = "int", n = 51*51*300, size = 4, endian = "big")
PD200MGH <- readBin(con = "RunB8_200MeV/LETDosePrimary.bin", what = "integer", n = 51*51*300, size = 4, endian = "big")
SD200MGH <- readBin(con = "RunB8_200MeV/LETDoseSecondary.bin", what = "int", n = 51*51*300, size = 4, endian = "big")
TD200MGH <- read.csv(file = "topasTotalDose/Scoring_200M.csv", sep = ",", fill = T, header = F)
TD200MGH <- TD200MGH$V4
TD200MGH <- reshape(as.matrix(TD200MGH), c(300, 51, 51))
PF200MGH <- reshape(as.matrix(PF200MGH), c(51, 51, 300))
SF200MGH <- reshape(as.matrix(SF200MGH), c(51, 51, 300))
PD200MGH <- reshape(as.matrix(PD200MGH), c(51, 51, 300))
SD200MGH <- reshape(as.matrix(SD200MGH), c(51, 51, 300))


PF200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/primaryFluence.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
SF200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/200MeV/secondaryFluence.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
PD200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/200MeV/primaryDose.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
SD200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/200MeV/secondaryDose.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
TD200 <- readBin(con = "GitHub/TransportOpenCL1_2/CLTransport/Output/200MeV/totalDose.bin", what = "numeric", n = 51*51*300, size = 4, endian = "little")
TD200 <- reshape(as.matrix(TD200), c(51, 51, 300))
PF200 <- reshape(as.matrix(PF200), c(51, 51, 300))
SF200 <- reshape(as.matrix(SF200), c(51, 51, 300))
PD200 <- reshape(as.matrix(PD200), c(51, 51, 300))
SD200 <- reshape(as.matrix(SD200), c(51, 51, 300))

PF200MGH[26,26,300]

PF200[26,26,1]
plot(TD200MGH[1:300, 26, 26], type = 'l')
plot(TD200[26, 26, 1:300], type = 'l')











debug <- read.table("C:/Users/S158879/workspace/GitHub/TransportOpenCL1_2/CLTransport/debug.txt", quote="\"")

sum(debug$V3)/0.004

test <- rep(T, times = length(debug$v3))
temp <- debug$V3[1]
for(i in 1:length(debug$V3)){
  #temp = temp + debug$V3[i+1]
  test[i] <- abs(debug$V3[i+1] + debug$V4[i] - debug$V4[i+1]) < debug$V4[i+1]*1e-4
}


which(!test)














scale <- max(PF200center)/max(PF200centerMGH)
PF200centerN <- PF200centerMGH*scale
SF200centerN <- SF200centerMGH*scale

plot(PF200center, main = "central beam fluence, 200MeV,", yaxt="n", xlab = "depth in mm", ylab = "primary fluence", type = 'l', col = 'red')
lines(PF200centerN)

plot(SF200center, main = "central beam fluence, 200MeV", yaxt="n", xlab = "depth in mm", ylab = "secondary fluence", type = 'l', col = 'red')
lines(SF200centerN)

scale <- 10/0.003953
PD200centerN <- PD200centerMGH*scale
SD200centerN <- SD200centerMGH*scale

plot(PD200center, main = "central beam dose, 200MeV,", yaxt="n", xlab = "depth in mm", ylab = "primary dose", type = 'l', col = 'red')
lines(PD200centerN)

plot(SD200center, main = "central beam dose, 200MeV", yaxt="n", xlab = "depth in mm", ylab = "secondary dose", type = 'l', col = 'red')
lines(SD200centerN)
