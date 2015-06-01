remove(list = ls())
cal_sigma_ion <- function(energy, Temin, m){
  re <- 2.8179403267e-13 #electron classical radius, in cm
  ne <- 3.342774e23 
  me <- 0.510998928  #electron mass, in MeV
  
  E <- energy + m
  gamma <- E/m
  beta2 <- 1.0 - 1.0/(gamma*gamma)
  
  Temax <- cal_Temax(energy, m)
  
  #    plot(energy, Temax)
  
  output <- 2*pi*re*re*me*ne*z*z/beta2*
            (1/Temin - 1/Temax - beta2/Temax*log(Temax/Temin) + (Temax - Temin)/2/energy/energy)
  output[Temax <= Temin] = 0
  return(output)
}

cal_Temax <- function(energy, m){
  me <- 0.510998928 
  memp <- me / m    #ratio between me and proton mass
  
  E <- energy + m
  gamma <- E/m
  #  beta2 <- 1.0 - 1.0./(gamma.*gamma)
  
  output <- 2*me*(gamma*gamma-1)/(1+2*gamma*memp + memp*memp) 
  return(output)
}

energy <- seq(from = 0.5, to = 500, by = 0.5)
Temin <- 0.1
mp <- 938.272046
me <- 0.510998928  
z <- 1
sigma <- NULL
sigma$energy <- energy
sigma$sigIon <- cal_sigma_ion(energy = energy, Temin = Temin, m = mp)
















