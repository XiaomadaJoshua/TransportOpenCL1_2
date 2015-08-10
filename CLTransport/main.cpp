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
//	cl_float8 * temp = new cl_float8[1000000]();
//	std::cin >> temp;
	MCEngine mc("proton_config");
	mc.simulate(MINPROTONENERGY);
//	std::cin >> temp;
}
