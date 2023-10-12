# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	g++-10 -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
