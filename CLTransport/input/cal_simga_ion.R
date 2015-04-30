remove(list = ls())
cal_sigma_ion <- function(energy, ne, Temin){
  re <- 2.8179403267e-13 #electron classical radius, in cm
  
  me <- 0.510998928  #electron mass, in MeV
  mc <- 11187.8957
  z <- 6
  
  E <- energy + mc
  gamma <- E/mc
  beta2 <- 1.0 - 1.0/(gamma*gamma)
  
  Temax <- cal_Temax(energy)
  
  #    plot(energy, Temax)
  
  output <- 2*pi*re*re*me*ne*z*z/beta2*((-1.0/Temax - beta2/Temax*log(Temax) + Temax/2/(E*E)) 
                                    - (-1.0/Temin - beta2/Temax*log(Temin) + Temin/2/(E*E)) )
  output[Temax <= Temin] = 0
  return(output)
}

cal_Temax <- function(energy){
  me <- 0.510998928 
  mc <- 11187.8957
  memp <- me / mc    #ratio between me and proton mass
  
  E <- energy + mc
  gamma <- E/mc
  #  beta2 <- 1.0 - 1.0./(gamma.*gamma)
  
  output <- 2*me*(gamma*gamma-1)/(1+2*gamma*memp + memp*memp) 
  return(output)
}

energy <- seq(from = 1, to = 5500, by = 1)
Temin <- 0.1
mc <- 11187.8957
me <- 0.510998928  
z <- 6
ne <- 3.342774e23 
na <- 6.0221413e23
hydrogen <- 1.008
oxygen <- 15.994
barn2cm2 <- 1e-24
sigma <- NULL
sigma$energy <- energy
sigma$sigIon <- cal_sigma_ion(energy = energy, ne = ne, Temin = Temin)


CHI <- read.delim("C:/Users/S158879/workspace/GitHub/OclCarbon/CLTransport/input/nuclearRawData/InelasticHadronicCrossSection_primary_C12_target_H1.txt")
COI <- read.delim("C:/Users/S158879/workspace/GitHub/OclCarbon/CLTransport/input/nuclearRawData/InelasticHadronicCrossSection_primary_C12_target_O16.txt")

CHI <- CHI$Cross.section..barn.*barn2cm2*na*2/(2*hydrogen+oxygen)
COI <- COI$Cross.section..barn.*barn2cm2*na/(2*hydrogen+oxygen)

plot(y = CHI, x = energy, type = 'l')
plot(y = COI, x = energy, type = 'l')


sigma$sigCHI <- CHI
sigma$sigCOI <- COI
sigma$sigCCI <- 0

write.table(x = sigma, file = "C:/Users/S158879/workspace/GitHub/TransportOpenCL1_2/CLTransport/input/carbon.crossSection", row.names = F, sep = '\t\t')















