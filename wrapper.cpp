#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ImageCoder.cpp"
#include "ac_enc.cpp"
#include "ac_dec.cpp"


namespace py = pybind11;

template<typename T>
T*** from_py_array2(py::array_t<T>& py_array)
{
	return (T***)py_array.data();
}

template<typename T>
T* from_py_array(py::array_t<T>& py_array)
{
	return (T*)py_array.data();
}


PYBIND11_MODULE(EntropyCodec, m){
    m.def("HiddenLayersEncoder", [](py::array_t<unsigned char>& layer1, int w1, int h1, int z1, 
		py::array_t<unsigned char>& stream, py::array_t<int>& bitsize)
    {
		auto l1 = from_py_array(layer1);
		auto st = from_py_array(stream);
		auto bs = from_py_array(bitsize);
		
		BitPlaneEncoder(st, l1, w1, h1, z1, bs);
    });

	m.def("HiddenLayersDecoder", [](py::array_t<unsigned char>& layer1, int w1, int h1, int z1,
		py::array_t<unsigned char>& stream, py::array_t<int>& offset)
    {
		auto l1 = from_py_array(layer1);
		auto st = from_py_array(stream);
		auto of = from_py_array(offset);

		BitPlaneDecoder(st, l1, w1, h1, z1, of);
    });
}

