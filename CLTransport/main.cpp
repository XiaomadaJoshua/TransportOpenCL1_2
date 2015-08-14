#include "cl.hpp"
#include "DensCorrection.h"
#include "OpenCLStuff.h"
#include "MacroCrossSection.h"
#include "RSPW.h"
#include "MSPR.h"
#include "Phantom.h"
#include "Proton.h"
#include "ParticleStatus.h"
#include "MCEngine.h"
#include <string>

#include "DensCorrection.cpp"
#include "MacroCrossSection.cpp"
#include "MCEngine.cpp"
#include "MSPR.cpp"
#include "OpenCLStuff.cpp"
#include "ParticleStatus.cpp"
#include "Phantom.cpp"
#include "Proton.cpp"
#include "RSPW.cpp"
#include "Secondary.cpp"

int main(){
	int temp;
//	std::cin >> temp;
	MCEngine mc("proton_config");
	mc.simulate(MINPROTONENERGY);
//	std::cin >> temp;
}
