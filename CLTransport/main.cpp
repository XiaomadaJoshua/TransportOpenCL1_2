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




int main(){
	int temp;
//	std::cin >> temp;
	MCEngine mc("proton_config");
	mc.simulate(MINPROTONENERGY);
//	std::cin >> temp;
}
