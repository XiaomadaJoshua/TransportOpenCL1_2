remove(list = ls())
cal_rstp <- function(energy, Temin, m, charge){
##  Summary of this function goes here
##   Detailed explanation goes here
  
  I <- 83e-6  #mean exitation enery of water, in MeV
  ne <- 3.342774e23   # for water, in 1/cm^3
  re <- 2.8179403267e-13 #electron classical radius, in cm
  
  E <- energy + m
  gamma <- E/m
  beta2 <- 1.0 - 1.0/(gamma*gamma)
  
  
  Temax <- cal_Temax(energy, m)
  Tup <- pmin(Temin, Temax)
  #  plot(energy, Temax)
  
  
  # density corretction, ONLY works for water
  hnu <- sqrt(4*pi*ne*re^3)*me*137
  C <- 1+ 2*log(I/hnu)
  xa <- C/4.606
  a <- 4.606*(xa - 0.2)/(2-0.2)^3
  x <- log(gamma*gamma*beta2)/4.606
  delta <- rep(0, length(x))
  delta[x >= 0.2 & x <= 2] <- 4.606*x[x >= 0.2 & x <= 2] - C + a*(2 - x[x >= 0.2 & x <= 2])^3
  delta[x > 2] <- 4.606*x[x > 2] - C
  
  output <- 2*pi*re*re*me*ne*charge*charge/beta2*(log(2*me*beta2*gamma*gamma*Tup/I/I) 
                                   - beta2*(1+Tup/Temax) - delta)
  return(output)
}

cal_Temax <- function(energy, m){
  memp <- me / m    #ratio between me and proton mass
  
  E <- energy + m
  gamma <- E/m
  #  beta2 <- 1.0 - 1.0./(gamma.*gamma)
  
  output <- 2*me*(gamma*gamma-1)/(1+2*gamma*memp + memp*memp) 
  return(output)
}

cal_b <- function(energy, Lw, m){
  dTp = 0.5
  gamma <- (energy + m)/m
  beta2 <- 1.0 - 1.0/(gamma*gamma)
  
  C <- Lw*beta2
  dE <- dTp
  # semilogx(Ene,C)
  dCdE <- rep(0, length(C))
  dCdE[2:(length(C)-1)] <- (C[3:length(C)] - C[1:(length(C)-2)])/2/dE
  dCdE[1] <- (C[2] - C[1])/dE[1]
  dCdE[length(C)] <- (C[length(C)] - C[length(C)-1])/dE
  b <- energy/C*dCdE

  return(b)
}

energy <- seq(from = 0.5, to = 350, by = 0.5)
Temin <- 0.1
mp <- 938.272046
me <- 0.510998928  
z <- 1
Lw <- cal_rstp(energy = energy, Temin = Temin, m = mp, charge = z)
b <- cal_b(energy = energy, Lw = Lw, m = mp)
rspw <- NULL
rspw$energy <- energy
rspw$Sw <- rep(0, length(energy))
rspw$MW <- rep(0, length(energy))
rspw$Lw <- Lw
rspw$b <- b





write.table(x = rspw, file = "C:/Users/S158879/workspace/GitHub/TransportOpenCL1_2/CLTransport/input/mcpro.rstpw", row.names = F, sep = '\t\t')

