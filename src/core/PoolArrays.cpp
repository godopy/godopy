#include "PoolArrays.hpp"
#include "Color.hpp"
#include "Defs.hpp"
#include "GodotGlobal.hpp"
#include "CoreTypes.hpp"

#include <gdnative/pool_arrays.h>

#include <_lib/godot/core/types.hpp>

namespace godot {

PoolByteArray::PoolByteArray() {
	godot::api->godot_pool_byte_array_new(&_godot_array);
}

PoolByteArray::PoolByteArray(const PoolByteArray &p_other) {
	godot::api->godot_pool_byte_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolByteArray &PoolByteArray::operator=(const PoolByteArray &p_other) {
	godot::api->godot_pool_byte_array_destroy(&_godot_array);
	godot::api->godot_pool_byte_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolByteArray::PoolByteArray(const Array &array) {
	godot::api->godot_pool_byte_array_new_with_array(&_godot_array, (godot_array *)&array);
}

PoolByteArray::PoolByteArray(PyObject *obj) {
	if (Py_TYPE(obj) == PyGodotWrapperType_PoolByteArray) {
		godot::api->godot_pool_byte_array_new_copy(&_godot_array, _python_wrapper_to_poolbytearray(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 1 && PyArray_TYPE(arr) == NPY_UINT8)) {
			godot::api->godot_pool_byte_array_new(&_godot_array);
			godot::api->godot_pool_byte_array_resize(&_godot_array, PyArray_SIZE(arr));
			godot_pool_byte_array_write_access *_write_access = godot::api->godot_pool_byte_array_write(&_godot_array);

			const uint8_t *dst = godot::api->godot_pool_byte_array_write_access_ptr(_write_access);
			memcpy((void *)dst, (void *)PyArray_GETPTR1(arr, 0), PyArray_SIZE(arr));

			godot::api->godot_pool_byte_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be an array of bytes");
		}
	} else if (PyBytes_Check(obj)) {
		godot::api->godot_pool_byte_array_new(&_godot_array);
		godot::api->godot_pool_byte_array_resize(&_godot_array, PyBytes_GET_SIZE(obj));
		godot_pool_byte_array_write_access *_write_access = godot::api->godot_pool_byte_array_write(&_godot_array);

		const uint8_t *ptr = godot::api->godot_pool_byte_array_write_access_ptr(_write_access);
		memcpy((void *)ptr, (void *)PyBytes_AS_STRING(obj), PyBytes_GET_SIZE(obj));

		godot::api->godot_pool_byte_array_write_access_destroy(_write_access);

	} else if (PyByteArray_Check(obj)) {
		godot::api->godot_pool_byte_array_new(&_godot_array);
		godot::api->godot_pool_byte_array_resize(&_godot_array, PyByteArray_GET_SIZE(obj));
		godot_pool_byte_array_write_access *_write_access = godot::api->godot_pool_byte_array_write(&_godot_array);

		const uint8_t *ptr = godot::api->godot_pool_byte_array_write_access_ptr(_write_access);
		memcpy((void *)ptr, (void *)PyByteArray_AS_STRING(obj), PyByteArray_GET_SIZE(obj));

		godot::api->godot_pool_byte_array_write_access_destroy(_write_access);

	} else if (PyObject_CheckBuffer(obj)) {
		throw std::invalid_argument("PoolByteArray construction from  generic Python buffers is not implemented yet");

	} else if (PySequence_Check(obj)) {
		WARN_PRINT("Possible data loss: casting Python sequence to 8-bit unsigned integers");
		godot::api->godot_pool_byte_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_byte_array_resize(&_godot_array, _size);
		godot_pool_byte_array_write_access *_write_access = godot::api->godot_pool_byte_array_write(&_godot_array);

		uint8_t *ptr = godot::api->godot_pool_byte_array_write_access_ptr(_write_access);

		for (int i = 0; i < _size; i++) {
			PyObject *item = PySequence_GetItem(obj, i);
			*(ptr + i) = (uint8_t)PyNumber_AsSsize_t(item, NULL);
		}

		godot::api->godot_pool_byte_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}
}

PoolByteArray::PoolByteArray(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 1 && PyArray_TYPE(arr) == NPY_UINT8)) {
		godot::api->godot_pool_byte_array_new(&_godot_array);
		godot::api->godot_pool_byte_array_resize(&_godot_array, PyArray_SIZE(arr));
		godot_pool_byte_array_write_access *_write_access = godot::api->godot_pool_byte_array_write(&_godot_array);
		uint8_t *dst = godot::api->godot_pool_byte_array_write_access_ptr(_write_access);
		uint8_t *src = (uint8_t *)PyArray_GETPTR1(arr, 0);

		memcpy((void *)dst, (void *)src, PyArray_SIZE(arr));

		godot::api->godot_pool_byte_array_write_access_destroy(_write_access);
	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a numpy array of bytes");
	}
}

PoolByteArray PoolByteArray_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolByteArray) ? *(PoolByteArray *)_python_wrapper_to_poolbytearray(obj) : PoolByteArray(obj);
}

PoolByteArray::Read PoolByteArray::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_byte_array_read(&_godot_array);
	return read;
}

PoolByteArray::Write PoolByteArray::write() {
	Write write;
	write._write_access = godot::api->godot_pool_byte_array_write(&_godot_array);
	return write;
}

void PoolByteArray::append(const uint8_t data) {
	godot::api->godot_pool_byte_array_append(&_godot_array, data);
}

void PoolByteArray::append_array(const PoolByteArray &array) {
	godot::api->godot_pool_byte_array_append_array(&_godot_array, &array._godot_array);
}

int PoolByteArray::insert(const int idx, const uint8_t data) {
	return godot::api->godot_pool_byte_array_insert(&_godot_array, idx, data);
}

void PoolByteArray::invert() {
	godot::api->godot_pool_byte_array_invert(&_godot_array);
}

void PoolByteArray::push_back(const uint8_t data) {
	godot::api->godot_pool_byte_array_push_back(&_godot_array, data);
}

void PoolByteArray::remove(const int idx) {
	godot::api->godot_pool_byte_array_remove(&_godot_array, idx);
}

void PoolByteArray::resize(const int size) {
	godot::api->godot_pool_byte_array_resize(&_godot_array, size);
}

void PoolByteArray::set(const int idx, const uint8_t data) {
	godot::api->godot_pool_byte_array_set(&_godot_array, idx, data);
}

uint8_t PoolByteArray::operator[](const int idx) {
	return godot::api->godot_pool_byte_array_get(&_godot_array, idx);
}

int PoolByteArray::size() const {
	return godot::api->godot_pool_byte_array_size(&_godot_array);
}

PoolByteArray::~PoolByteArray() {
	godot::api->godot_pool_byte_array_destroy(&_godot_array);
}

PoolIntArray::PoolIntArray() {
	godot::api->godot_pool_int_array_new(&_godot_array);
}

PoolIntArray::PoolIntArray(const PoolIntArray &p_other) {
	godot::api->godot_pool_int_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolIntArray &PoolIntArray::operator=(const PoolIntArray &p_other) {
	godot::api->godot_pool_int_array_destroy(&_godot_array);
	godot::api->godot_pool_int_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolIntArray::PoolIntArray(const Array &array) {
	godot::api->godot_pool_int_array_new_with_array(&_godot_array, (godot_array *)&array);
}

PoolIntArray::PoolIntArray(PyObject *obj) {
	if (Py_TYPE(obj) == PyGodotWrapperType_PoolIntArray) {
		godot::api->godot_pool_int_array_new_copy(&_godot_array, _python_wrapper_to_poolintarray(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 1 && PyArray_ISNUMBER(arr))) {
			if (PyArray_TYPE(arr) != NPY_INT) {
				WARN_PRINT("Possible data loss: casting unknown numeric array to 32-bit signed integers");
			}

			godot::api->godot_pool_int_array_new(&_godot_array);
			godot::api->godot_pool_int_array_resize(&_godot_array, PyArray_SIZE(arr));
			godot_pool_int_array_write_access *_write_access = godot::api->godot_pool_int_array_write(&_godot_array);

			int *dst = godot::api->godot_pool_int_array_write_access_ptr(_write_access);
			int *src = (int *)PyArray_GETPTR1(arr, 0);

			if (PyArray_NBYTES(arr) == (long)(PyArray_SIZE(arr) * sizeof(int))) {
				memcpy((void *)dst, (void *)src, PyArray_NBYTES(arr));
			} else {
				for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
					*(dst + idx) = *(int *)PyArray_GETPTR1(arr, idx);
				}
			}

			godot::api->godot_pool_int_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be an array of numbers");
		}
	} else if (PySequence_Check(obj)) {
		WARN_PRINT("Possible data loss: casting Python sequence to 32-bit integers");

		godot::api->godot_pool_int_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_int_array_resize(&_godot_array, _size);
		godot_pool_int_array_write_access *_write_access = godot::api->godot_pool_int_array_write(&_godot_array);

		int *ptr = godot::api->godot_pool_int_array_write_access_ptr(_write_access);

		for (int i = 0; i < _size; i++) {
			PyObject *item = PySequence_GetItem(obj, i);
			*(ptr + i) = (int)PyNumber_AsSsize_t(item, NULL);
		}

		godot::api->godot_pool_byte_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}
}

PoolIntArray::PoolIntArray(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 1 && PyArray_ISNUMBER(arr))) {
		if (PyArray_TYPE(arr) != NPY_INT) {
			WARN_PRINT("Possible data loss: casting unknown numeric array to 32-bit signed integers");
		}

		godot::api->godot_pool_int_array_new(&_godot_array);
		godot::api->godot_pool_int_array_resize(&_godot_array, PyArray_SIZE(arr));
		godot_pool_int_array_write_access *_write_access = godot::api->godot_pool_int_array_write(&_godot_array);

		int *dst = godot::api->godot_pool_int_array_write_access_ptr(_write_access);
		int *src = (int *)PyArray_GETPTR1(arr, 0);

		if (PyArray_NBYTES(arr) == (long)(PyArray_SIZE(arr) * sizeof(int))) {
			memcpy((void *)dst, (void *)src, PyArray_NBYTES(arr));
		} else {
			for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
				*(dst + idx) = *(int *)PyArray_GETPTR1(arr, idx);
			}
		}

		godot::api->godot_pool_int_array_write_access_destroy(_write_access);
	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a numpy array of numbers");
	}
}

PoolIntArray PoolIntArray_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolIntArray) ? *(PoolIntArray *)_python_wrapper_to_poolintarray(obj) : PoolIntArray(obj);
}

PoolIntArray::Read PoolIntArray::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_int_array_read(&_godot_array);
	return read;
}

PoolIntArray::Write PoolIntArray::write() {
	Write write;
	write._write_access = godot::api->godot_pool_int_array_write(&_godot_array);
	return write;
}

void PoolIntArray::append(const int data) {
	godot::api->godot_pool_int_array_append(&_godot_array, data);
}

void PoolIntArray::append_array(const PoolIntArray &array) {
	godot::api->godot_pool_int_array_append_array(&_godot_array, &array._godot_array);
}

int PoolIntArray::insert(const int idx, const int data) {
	return godot::api->godot_pool_int_array_insert(&_godot_array, idx, data);
}

void PoolIntArray::invert() {
	godot::api->godot_pool_int_array_invert(&_godot_array);
}

void PoolIntArray::push_back(const int data) {
	godot::api->godot_pool_int_array_push_back(&_godot_array, data);
}

void PoolIntArray::remove(const int idx) {
	godot::api->godot_pool_int_array_remove(&_godot_array, idx);
}

void PoolIntArray::resize(const int size) {
	godot::api->godot_pool_int_array_resize(&_godot_array, size);
}

void PoolIntArray::set(const int idx, const int data) {
	godot::api->godot_pool_int_array_set(&_godot_array, idx, data);
}

int PoolIntArray::operator[](const int idx) {
	return godot::api->godot_pool_int_array_get(&_godot_array, idx);
}

int PoolIntArray::size() const {
	return godot::api->godot_pool_int_array_size(&_godot_array);
}

PoolIntArray::~PoolIntArray() {
	godot::api->godot_pool_int_array_destroy(&_godot_array);
}

PoolRealArray::PoolRealArray() {
	godot::api->godot_pool_real_array_new(&_godot_array);
}

PoolRealArray::PoolRealArray(const PoolRealArray &p_other) {
	godot::api->godot_pool_real_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolRealArray &PoolRealArray::operator=(const PoolRealArray &p_other) {
	godot::api->godot_pool_real_array_destroy(&_godot_array);
	godot::api->godot_pool_real_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolRealArray::PoolRealArray(PyObject *obj) {
	if (Py_TYPE(obj) == PyGodotWrapperType_PoolRealArray) {
		godot::api->godot_pool_real_array_new_copy(&_godot_array, _python_wrapper_to_poolrealarray(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 1 && PyArray_ISNUMBER(arr))) {
			if (PyArray_TYPE(arr) == NPY_DOUBLE) {
				WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
			} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
				WARN_PRINT("Possible data loss: casting unknown numeric array to 32-bit floats");
			}

			godot::api->godot_pool_real_array_new(&_godot_array);
			godot::api->godot_pool_real_array_resize(&_godot_array, PyArray_SIZE(arr));
			godot_pool_real_array_write_access *_write_access = godot::api->godot_pool_real_array_write(&_godot_array);

			float *dst = godot::api->godot_pool_real_array_write_access_ptr(_write_access);
			float *src = (float *)PyArray_GETPTR1(arr, 0);

			if (PyArray_NBYTES(arr) == (long)(PyArray_SIZE(arr) * sizeof(float))) {
				memcpy((void *)dst, (void *)src, PyArray_NBYTES(arr));
			} else {
				for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
					*(dst + idx) = *(float *)PyArray_GETPTR1(arr, idx);
				}
			}

			godot::api->godot_pool_real_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be an array of numbers");
		}
	} else if (PySequence_Check(obj)) {
		WARN_PRINT("Possible data loss: casting Python sequence to 32-bit floats");
		godot::api->godot_pool_real_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_real_array_resize(&_godot_array, _size);
		godot_pool_real_array_write_access *_write_access = godot::api->godot_pool_real_array_write(&_godot_array);

		float *ptr = godot::api->godot_pool_real_array_write_access_ptr(_write_access);

		for (int i = 0; i < _size; i++) {
			PyObject *item = PySequence_GetItem(obj, i);
			// TODO: add checks
			PyObject *num = PyNumber_Float(item);
			*(ptr + i) = (float)PyFloat_AS_DOUBLE(item);
		}

		godot::api->godot_pool_real_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}
}

PoolRealArray::PoolRealArray(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 1 && PyArray_ISNUMBER(arr))) {
		if (PyArray_TYPE(arr) == NPY_DOUBLE) {
			WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
		} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
			WARN_PRINT("Possible data loss: casting unknown numeric array to 32-bit floats");
		}

		godot::api->godot_pool_real_array_new(&_godot_array);
		godot::api->godot_pool_real_array_resize(&_godot_array, PyArray_SIZE(arr));
		godot_pool_real_array_write_access *_write_access = godot::api->godot_pool_real_array_write(&_godot_array);

		float *dst = godot::api->godot_pool_real_array_write_access_ptr(_write_access);
		float *src = (float *)PyArray_GETPTR1(arr, 0);

		if (PyArray_NBYTES(arr) == (long)(PyArray_SIZE(arr) * sizeof(float))) {
			memcpy((void *)dst, (void *)src, PyArray_NBYTES(arr));
		} else {
			for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
				*(dst + idx) = *(float *)PyArray_GETPTR1(arr, idx);
			}
		}

		godot::api->godot_pool_real_array_write_access_destroy(_write_access);
	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a numpy array of numbers");
	}
}

PoolRealArray PoolRealArray_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolRealArray) ? *(PoolRealArray *)_python_wrapper_to_poolrealarray(obj) : PoolRealArray(obj);
}

PoolRealArray::Read PoolRealArray::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_real_array_read(&_godot_array);
	return read;
}

PoolRealArray::Write PoolRealArray::write() {
	Write write;
	write._write_access = godot::api->godot_pool_real_array_write(&_godot_array);
	return write;
}

PoolRealArray::PoolRealArray(const Array &array) {
	godot::api->godot_pool_real_array_new_with_array(&_godot_array, (godot_array *)&array);
}

void PoolRealArray::append(const real_t data) {
	godot::api->godot_pool_real_array_append(&_godot_array, data);
}

void PoolRealArray::append_array(const PoolRealArray &array) {
	godot::api->godot_pool_real_array_append_array(&_godot_array, &array._godot_array);
}

int PoolRealArray::insert(const int idx, const real_t data) {
	return godot::api->godot_pool_real_array_insert(&_godot_array, idx, data);
}

void PoolRealArray::invert() {
	godot::api->godot_pool_real_array_invert(&_godot_array);
}

void PoolRealArray::push_back(const real_t data) {
	godot::api->godot_pool_real_array_push_back(&_godot_array, data);
}

void PoolRealArray::remove(const int idx) {
	godot::api->godot_pool_real_array_remove(&_godot_array, idx);
}

void PoolRealArray::resize(const int size) {
	godot::api->godot_pool_real_array_resize(&_godot_array, size);
}

void PoolRealArray::set(const int idx, const real_t data) {
	godot::api->godot_pool_real_array_set(&_godot_array, idx, data);
}

real_t PoolRealArray::operator[](const int idx) {
	return godot::api->godot_pool_real_array_get(&_godot_array, idx);
}

int PoolRealArray::size() const {
	return godot::api->godot_pool_real_array_size(&_godot_array);
}

PoolRealArray::~PoolRealArray() {
	godot::api->godot_pool_real_array_destroy(&_godot_array);
}

PoolStringArray::PoolStringArray() {
	godot::api->godot_pool_string_array_new(&_godot_array);
}

PoolStringArray::PoolStringArray(const PoolStringArray &p_other) {
	godot::api->godot_pool_string_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolStringArray &PoolStringArray::operator=(const PoolStringArray &p_other) {
	godot::api->godot_pool_string_array_destroy(&_godot_array);
	godot::api->godot_pool_string_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolStringArray::PoolStringArray(const Array &array) {
	godot::api->godot_pool_string_array_new_with_array(&_godot_array, (godot_array *)&array);
}

PoolStringArray::PoolStringArray(PyObject *obj) {
	if (Py_TYPE(obj) == PyGodotWrapperType_PoolStringArray) {
		godot::api->godot_pool_string_array_new_copy(&_godot_array, _python_wrapper_to_poolstringarray(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 1 && PyArray_ISSTRING(arr))) {
			godot::api->godot_pool_string_array_new(&_godot_array);
			godot::api->godot_pool_string_array_resize(&_godot_array, PyArray_SIZE(arr));
			godot_pool_string_array_write_access *_write_access = godot::api->godot_pool_string_array_write(&_godot_array);

			String *ptr = (String *)godot::api->godot_pool_string_array_write_access_ptr(_write_access);

			for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
				PyObject *item = PyArray_GETITEM(arr, (const char *)PyArray_GETPTR1(arr, idx));
				*(ptr + idx) = String(item);
			}

			godot::api->godot_pool_string_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be an array of strings");
		}
	} else if (PySequence_Check(obj)) {
		godot::api->godot_pool_string_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_string_array_resize(&_godot_array, _size);
		godot_pool_string_array_write_access *_write_access = godot::api->godot_pool_string_array_write(&_godot_array);

		String *ptr = (String *)godot::api->godot_pool_string_array_write_access_ptr(_write_access);

		for (int idx = 0; idx < _size; idx++) {
			PyObject *item = PySequence_GetItem(obj, idx);
			*(ptr + idx) = String(item);
		}

		godot::api->godot_pool_string_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}
}

PoolStringArray::PoolStringArray(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 1 && PyArray_ISSTRING(arr))) {
		godot::api->godot_pool_string_array_new(&_godot_array);
		godot::api->godot_pool_string_array_resize(&_godot_array, PyArray_SIZE(arr));
		godot_pool_string_array_write_access *_write_access = godot::api->godot_pool_string_array_write(&_godot_array);

		String *ptr = (String *)godot::api->godot_pool_string_array_write_access_ptr(_write_access);

		for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
			PyObject *item = PyArray_GETITEM(arr, (const char *)PyArray_GETPTR1(arr, idx));
			*(ptr + idx) = String(item);
		}

		godot::api->godot_pool_string_array_write_access_destroy(_write_access);

	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a numpy array of strings");
	}
}

PoolStringArray PoolStringArray_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolStringArray) ? *(PoolStringArray *)_python_wrapper_to_poolstringarray(obj) : PoolStringArray(obj);
}

PoolStringArray::Read PoolStringArray::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_string_array_read(&_godot_array);
	return read;
}

PoolStringArray::Write PoolStringArray::write() {
	Write write;
	write._write_access = godot::api->godot_pool_string_array_write(&_godot_array);
	return write;
}

void PoolStringArray::append(const String &data) {
	godot::api->godot_pool_string_array_append(&_godot_array, (godot_string *)&data);
}

void PoolStringArray::append_array(const PoolStringArray &array) {
	godot::api->godot_pool_string_array_append_array(&_godot_array, &array._godot_array);
}

int PoolStringArray::insert(const int idx, const String &data) {
	return godot::api->godot_pool_string_array_insert(&_godot_array, idx, (godot_string *)&data);
}

void PoolStringArray::invert() {
	godot::api->godot_pool_string_array_invert(&_godot_array);
}

void PoolStringArray::push_back(const String &data) {
	godot::api->godot_pool_string_array_push_back(&_godot_array, (godot_string *)&data);
}

void PoolStringArray::remove(const int idx) {
	godot::api->godot_pool_string_array_remove(&_godot_array, idx);
}

void PoolStringArray::resize(const int size) {
	godot::api->godot_pool_string_array_resize(&_godot_array, size);
}

void PoolStringArray::set(const int idx, const String &data) {
	godot::api->godot_pool_string_array_set(&_godot_array, idx, (godot_string *)&data);
}

const String PoolStringArray::operator[](const int idx) {
	String s;
	godot_string str = godot::api->godot_pool_string_array_get(&_godot_array, idx);
	godot::api->godot_string_new_copy((godot_string *)&s, &str);
	godot::api->godot_string_destroy(&str);
	return s;
}

int PoolStringArray::size() const {
	return godot::api->godot_pool_string_array_size(&_godot_array);
}

PoolStringArray::~PoolStringArray() {
	godot::api->godot_pool_string_array_destroy(&_godot_array);
}

PoolVector2Array::PoolVector2Array() {
	godot::api->godot_pool_vector2_array_new(&_godot_array);
}

PoolVector2Array::PoolVector2Array(const PoolVector2Array &p_other) {
	godot::api->godot_pool_vector2_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolVector2Array &PoolVector2Array::operator=(const PoolVector2Array &p_other) {
	godot::api->godot_pool_vector2_array_destroy(&_godot_array);
	godot::api->godot_pool_vector2_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolVector2Array::PoolVector2Array(const Array &array) {
	godot::api->godot_pool_vector2_array_new_with_array(&_godot_array, (godot_array *)&array);
}

PoolVector2Array::PoolVector2Array(PyObject *obj) {
	bool errors = false;

	if (Py_TYPE(obj) == PyGodotWrapperType_PoolVector2Array) {
		godot::api->godot_pool_vector2_array_new_copy(&_godot_array, _python_wrapper_to_poolvector2array(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 2 && PyArray_ISFLOAT(arr) && PyArray_DIM(arr, 1) == 2)) {
			if (PyArray_TYPE(arr) == NPY_DOUBLE) {
				WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
			} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
				WARN_PRINT("Possible data loss: casting unknown float array to 32 bit floats");
			}

			npy_intp _size = PyArray_DIM(arr, 0);
			godot::api->godot_pool_vector2_array_new(&_godot_array);
			godot::api->godot_pool_vector2_array_resize(&_godot_array, _size);
			godot_pool_vector2_array_write_access *_write_access = godot::api->godot_pool_vector2_array_write(&_godot_array);

			Vector2 *ptr = (Vector2 *)godot::api->godot_pool_vector2_array_write_access_ptr(_write_access);

			for (int idx = 0; idx < _size; idx++) {
				*(ptr + idx) = Vector2(*(real_t *)PyArray_GETPTR2(arr, idx, 0), *(real_t *)PyArray_GETPTR2(arr, idx, 1));
			}

			godot::api->godot_pool_vector2_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be a 2-dimentional (N, 2) array of numbers");
		}
	} else if (PySequence_Check(obj)) {
		godot::api->godot_pool_vector2_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_vector2_array_resize(&_godot_array, _size);
		godot_pool_vector2_array_write_access *_write_access = godot::api->godot_pool_vector2_array_write(&_godot_array);

		Vector2 *ptr = (Vector2 *)godot::api->godot_pool_vector2_array_write_access_ptr(_write_access);

		PyObject *item = NULL;

		for (int idx = 0; idx < _size; idx++) {
			item = PySequence_GetItem(obj, idx);

			if (unlikely(item == NULL)) {
				errors = true;
				break;
			}

			*(ptr + idx) = Vector2_from_PyObject(item);
		}

		godot::api->godot_pool_vector2_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}

	if (errors) throw std::invalid_argument("could not convert Python object");
}

PoolVector2Array::PoolVector2Array(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 2 && PyArray_ISNUMBER(arr) && PyArray_DIM(arr, 1) == 2)) {
		if (PyArray_TYPE(arr) == NPY_DOUBLE) {
			WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
		} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
			WARN_PRINT("Possible data loss: casting unknown numeric array to 32 bit floats");
		}

		npy_intp _size = PyArray_DIM(arr, 0);
		godot::api->godot_pool_vector2_array_new(&_godot_array);
		godot::api->godot_pool_vector2_array_resize(&_godot_array, _size);
		godot_pool_vector2_array_write_access *_write_access = godot::api->godot_pool_vector2_array_write(&_godot_array);

		Vector2 *ptr = (Vector2 *)godot::api->godot_pool_vector2_array_write_access_ptr(_write_access);

		for (int idx = 0; idx < _size; idx++) {
			*(ptr + idx) = Vector2(*(real_t *)PyArray_GETPTR2(arr, idx, 0), *(real_t *)PyArray_GETPTR2(arr, idx, 1));
		}

	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a 2-dimentional (N, 2) numpy array of numbers");
	}
}

PoolVector2Array PoolVector2Array_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolVector2Array) ?
	       *(PoolVector2Array *)_python_wrapper_to_poolvector2array(obj) : PoolVector2Array(obj);
}

PoolVector2Array::Read PoolVector2Array::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_vector2_array_read(&_godot_array);
	return read;
}

PoolVector2Array::Write PoolVector2Array::write() {
	Write write;
	write._write_access = godot::api->godot_pool_vector2_array_write(&_godot_array);
	return write;
}

void PoolVector2Array::append(const Vector2 &data) {
	godot::api->godot_pool_vector2_array_append(&_godot_array, (godot_vector2 *)&data);
}

void PoolVector2Array::append_array(const PoolVector2Array &array) {
	godot::api->godot_pool_vector2_array_append_array(&_godot_array, &array._godot_array);
}

int PoolVector2Array::insert(const int idx, const Vector2 &data) {
	return godot::api->godot_pool_vector2_array_insert(&_godot_array, idx, (godot_vector2 *)&data);
}

void PoolVector2Array::invert() {
	godot::api->godot_pool_vector2_array_invert(&_godot_array);
}

void PoolVector2Array::push_back(const Vector2 &data) {
	godot::api->godot_pool_vector2_array_push_back(&_godot_array, (godot_vector2 *)&data);
}

void PoolVector2Array::remove(const int idx) {
	godot::api->godot_pool_vector2_array_remove(&_godot_array, idx);
}

void PoolVector2Array::resize(const int size) {
	godot::api->godot_pool_vector2_array_resize(&_godot_array, size);
}

void PoolVector2Array::set(const int idx, const Vector2 &data) {
	godot::api->godot_pool_vector2_array_set(&_godot_array, idx, (godot_vector2 *)&data);
}

const Vector2 PoolVector2Array::operator[](const int idx) {
	Vector2 v;
	*(godot_vector2 *)&v = godot::api->godot_pool_vector2_array_get(&_godot_array, idx);
	return v;
}

int PoolVector2Array::size() const {
	return godot::api->godot_pool_vector2_array_size(&_godot_array);
}

PoolVector2Array::~PoolVector2Array() {
	godot::api->godot_pool_vector2_array_destroy(&_godot_array);
}

PoolVector3Array::PoolVector3Array() {
	godot::api->godot_pool_vector3_array_new(&_godot_array);
}

PoolVector3Array::PoolVector3Array(const PoolVector3Array &p_other) {
	godot::api->godot_pool_vector3_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolVector3Array &PoolVector3Array::operator=(const PoolVector3Array &p_other) {
	godot::api->godot_pool_vector3_array_destroy(&_godot_array);
	godot::api->godot_pool_vector3_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolVector3Array::PoolVector3Array(const Array &array) {
	godot::api->godot_pool_vector3_array_new_with_array(&_godot_array, (godot_array *)&array);
}

PoolVector3Array::PoolVector3Array(PyObject *obj) {
	bool errors = false;

	if (Py_TYPE(obj) == PyGodotWrapperType_PoolVector3Array) {
		godot::api->godot_pool_vector3_array_new_copy(&_godot_array, _python_wrapper_to_poolvector3array(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 2 && PyArray_ISFLOAT(arr) && PyArray_DIM(arr, 1) == 3)) {
			if (PyArray_TYPE(arr) == NPY_DOUBLE) {
				WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
			} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
				WARN_PRINT("Possible data loss: casting unknown float array to 32 bit floats");
			}

			npy_intp _size = PyArray_DIM(arr, 0);
			godot::api->godot_pool_vector3_array_new(&_godot_array);
			godot::api->godot_pool_vector3_array_resize(&_godot_array, _size);
			godot_pool_vector3_array_write_access *_write_access = godot::api->godot_pool_vector3_array_write(&_godot_array);

			Vector3 *ptr = (Vector3 *)godot::api->godot_pool_vector3_array_write_access_ptr(_write_access);

			for (int idx = 0; idx < _size; idx++) {
				*(ptr + idx) = Vector3(*(real_t *)PyArray_GETPTR2(arr, idx, 0), *(real_t *)PyArray_GETPTR2(arr, idx, 1), *(real_t *)PyArray_GETPTR2(arr, idx, 2));
			}

			godot::api->godot_pool_vector3_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be a 2-dimentional (N, 3) array of numbers");
		}
	} else if (PySequence_Check(obj)) {
		godot::api->godot_pool_vector3_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_vector3_array_resize(&_godot_array, _size);
		godot_pool_vector3_array_write_access *_write_access = godot::api->godot_pool_vector3_array_write(&_godot_array);

		Vector3 *ptr = (Vector3 *)godot::api->godot_pool_vector3_array_write_access_ptr(_write_access);

		PyObject *item = NULL;

		for (int idx = 0; idx < _size; idx++) {
			item = PySequence_GetItem(obj, idx);

			if (unlikely(item == NULL)) {
				errors = true;
				break;
			}

			*(ptr + idx) = Vector3_from_PyObject(item);
		}

		godot::api->godot_pool_vector3_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}

	if (errors) throw std::invalid_argument("could not convert Python object");
}

PoolVector3Array::PoolVector3Array(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 2 && PyArray_ISNUMBER(arr) && PyArray_DIM(arr, 1) == 3)) {
		if (PyArray_TYPE(arr) == NPY_DOUBLE) {
			WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
		} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
			WARN_PRINT("Possible data loss: casting unknown numeric array to 32 bit floats");
		}

		npy_intp _size = PyArray_DIM(arr, 0);
		godot::api->godot_pool_vector3_array_new(&_godot_array);
		godot::api->godot_pool_vector3_array_resize(&_godot_array, _size);
		godot_pool_vector3_array_write_access *_write_access = godot::api->godot_pool_vector3_array_write(&_godot_array);

		Vector3 *ptr = (Vector3 *)godot::api->godot_pool_vector3_array_write_access_ptr(_write_access);

		for (int idx = 0; idx < _size; idx++) {
			*(ptr + idx) = Vector3(*(real_t *)PyArray_GETPTR2(arr, idx, 0), *(real_t *)PyArray_GETPTR2(arr, idx, 1), *(real_t *)PyArray_GETPTR2(arr, idx, 2));
		}

	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a 2-dimentional (N, 3) numpy array of numbers");
	}
}

PoolVector3Array PoolVector3Array_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolVector3Array) ?
	       *(PoolVector3Array *)_python_wrapper_to_poolvector3array(obj) : PoolVector3Array(obj);
}

PoolVector3Array::Read PoolVector3Array::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_vector3_array_read(&_godot_array);
	return read;
}

PoolVector3Array::Write PoolVector3Array::write() {
	Write write;
	write._write_access = godot::api->godot_pool_vector3_array_write(&_godot_array);
	return write;
}

void PoolVector3Array::append(const Vector3 &data) {
	godot::api->godot_pool_vector3_array_append(&_godot_array, (godot_vector3 *)&data);
}

void PoolVector3Array::append_array(const PoolVector3Array &array) {
	godot::api->godot_pool_vector3_array_append_array(&_godot_array, &array._godot_array);
}

int PoolVector3Array::insert(const int idx, const Vector3 &data) {
	return godot::api->godot_pool_vector3_array_insert(&_godot_array, idx, (godot_vector3 *)&data);
}

void PoolVector3Array::invert() {
	godot::api->godot_pool_vector3_array_invert(&_godot_array);
}

void PoolVector3Array::push_back(const Vector3 &data) {
	godot::api->godot_pool_vector3_array_push_back(&_godot_array, (godot_vector3 *)&data);
}

void PoolVector3Array::remove(const int idx) {
	godot::api->godot_pool_vector3_array_remove(&_godot_array, idx);
}

void PoolVector3Array::resize(const int size) {
	godot::api->godot_pool_vector3_array_resize(&_godot_array, size);
}

void PoolVector3Array::set(const int idx, const Vector3 &data) {
	godot::api->godot_pool_vector3_array_set(&_godot_array, idx, (godot_vector3 *)&data);
}

const Vector3 PoolVector3Array::operator[](const int idx) {
	Vector3 v;
	*(godot_vector3 *)&v = godot::api->godot_pool_vector3_array_get(&_godot_array, idx);
	return v;
}

int PoolVector3Array::size() const {
	return godot::api->godot_pool_vector3_array_size(&_godot_array);
}

PoolVector3Array::~PoolVector3Array() {
	godot::api->godot_pool_vector3_array_destroy(&_godot_array);
}

PoolColorArray::PoolColorArray() {
	godot::api->godot_pool_color_array_new(&_godot_array);
}

PoolColorArray::PoolColorArray(const PoolColorArray &p_other) {
	godot::api->godot_pool_color_array_new_copy(&_godot_array, &p_other._godot_array);
}

PoolColorArray &PoolColorArray::operator=(const PoolColorArray &p_other) {
	godot::api->godot_pool_color_array_destroy(&_godot_array);
	godot::api->godot_pool_color_array_new_copy(&_godot_array, &p_other._godot_array);
	return *this;
}

PoolColorArray::PoolColorArray(const Array &array) {
	godot::api->godot_pool_color_array_new_with_array(&_godot_array, (godot_array *)&array);
}

PoolColorArray::PoolColorArray(PyObject *obj) {
	bool errors = false;

	if (Py_TYPE(obj) == PyGodotWrapperType_PoolColorArray) {
		godot::api->godot_pool_color_array_new_copy(&_godot_array, _python_wrapper_to_poolcolorarray(obj));

	} else if (PyArray_Check(obj)) {
		PyArrayObject *arr = (PyArrayObject *)obj;

		if (likely(PyArray_NDIM(arr) == 2 && PyArray_ISFLOAT(arr) && PyArray_DIM(arr, 1) == 4)) {
			if (PyArray_TYPE(arr) == NPY_DOUBLE) {
				WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
			} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
				WARN_PRINT("Possible data loss: casting unknown float array to 32 bit floats");
			}

			npy_intp _size = PyArray_DIM(arr, 0);
			godot::api->godot_pool_color_array_new(&_godot_array);
			godot::api->godot_pool_color_array_resize(&_godot_array, _size);
			godot_pool_color_array_write_access *_write_access = godot::api->godot_pool_color_array_write(&_godot_array);

			Color *ptr = (Color *)godot::api->godot_pool_color_array_write_access_ptr(_write_access);

			for (int idx = 0; idx < _size; idx++) {
				*(ptr + idx) = Color(*(float *)PyArray_GETPTR2(arr, idx, 0), *(float*)PyArray_GETPTR2(arr, idx, 1),
														 *(float *)PyArray_GETPTR2(arr, idx, 2), *(float *)PyArray_GETPTR2(arr, idx, 3));
			}

			godot::api->godot_pool_color_array_write_access_destroy(_write_access);

		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("argument must be a 2-dimentional (N, 4) array of numbers");
		}
	} else if (PySequence_Check(obj)) {
		godot::api->godot_pool_color_array_new(&_godot_array);
		int _size = PySequence_Length(obj);
		godot::api->godot_pool_color_array_resize(&_godot_array, _size);
		godot_pool_color_array_write_access *_write_access = godot::api->godot_pool_color_array_write(&_godot_array);

		Color *ptr = (Color *)godot::api->godot_pool_color_array_write_access_ptr(_write_access);

		PyObject *item = NULL;

		for (int idx = 0; idx < _size; idx++) {
			item = PySequence_GetItem(obj, idx);

			if (unlikely(item == NULL)) {
				errors = true;
				break;
			}

			*(ptr + idx) = Color_from_PyObject(item);
		}

		godot::api->godot_pool_color_array_write_access_destroy(_write_access);

	} else {
		throw std::invalid_argument("could not convert Python object");
	}

	if (errors) throw std::invalid_argument("could not convert Python object");
}

PoolColorArray::PoolColorArray(PyArrayObject *arr) {
	if (likely(PyArray_Check(arr) && PyArray_NDIM(arr) == 2 && PyArray_ISNUMBER(arr) && PyArray_DIM(arr, 1) == 4)) {
		if (PyArray_TYPE(arr) == NPY_DOUBLE) {
			WARN_PRINT("Possible data loss: casting 64-bit float array to 32 bits");
		} else if (PyArray_TYPE(arr) != NPY_FLOAT) {
			WARN_PRINT("Possible data loss: casting unknown numeric array to 32 bit floats");
		}

		npy_intp _size = PyArray_DIM(arr, 0);
		godot::api->godot_pool_color_array_new(&_godot_array);
		godot::api->godot_pool_color_array_resize(&_godot_array, _size);
		godot_pool_color_array_write_access *_write_access = godot::api->godot_pool_color_array_write(&_godot_array);

		Color *ptr = (Color *)godot::api->godot_pool_color_array_write_access_ptr(_write_access);

		for (int idx = 0; idx < _size; idx++) {
			*(ptr + idx) = Color(*(float *)PyArray_GETPTR2(arr, idx, 0), *(float *)PyArray_GETPTR2(arr, idx, 1),
													 *(float *)PyArray_GETPTR2(arr, idx, 2), *(float *)PyArray_GETPTR2(arr, idx, 3));
		}

	} else {
		// raises ValueError in Cython/Python context
		throw std::invalid_argument("argument must be a 2-dimentional (N, 4) numpy array of numbers");
	}
}

PoolColorArray PoolColorArray_from_PyObject(PyObject *obj) {
	return (Py_TYPE(obj) == PyGodotWrapperType_PoolColorArray) ?
	       *(PoolColorArray *)_python_wrapper_to_poolcolorarray(obj) : PoolColorArray(obj);
}

PoolColorArray::Read PoolColorArray::read() const {
	Read read;
	read._read_access = godot::api->godot_pool_color_array_read(&_godot_array);
	return read;
}

PoolColorArray::Write PoolColorArray::write() {
	Write write;
	write._write_access = godot::api->godot_pool_color_array_write(&_godot_array);
	return write;
}

void PoolColorArray::append(const Color &data) {
	godot::api->godot_pool_color_array_append(&_godot_array, (godot_color *)&data);
}

void PoolColorArray::append_array(const PoolColorArray &array) {
	godot::api->godot_pool_color_array_append_array(&_godot_array, &array._godot_array);
}

int PoolColorArray::insert(const int idx, const Color &data) {
	return godot::api->godot_pool_color_array_insert(&_godot_array, idx, (godot_color *)&data);
}

void PoolColorArray::invert() {
	godot::api->godot_pool_color_array_invert(&_godot_array);
}

void PoolColorArray::push_back(const Color &data) {
	godot::api->godot_pool_color_array_push_back(&_godot_array, (godot_color *)&data);
}

void PoolColorArray::remove(const int idx) {
	godot::api->godot_pool_color_array_remove(&_godot_array, idx);
}

void PoolColorArray::resize(const int size) {
	godot::api->godot_pool_color_array_resize(&_godot_array, size);
}

void PoolColorArray::set(const int idx, const Color &data) {
	godot::api->godot_pool_color_array_set(&_godot_array, idx, (godot_color *)&data);
}

const Color PoolColorArray::operator[](const int idx) {
	Color v;
	*(godot_color *)&v = godot::api->godot_pool_color_array_get(&_godot_array, idx);
	return v;
}

int PoolColorArray::size() const {
	return godot::api->godot_pool_color_array_size(&_godot_array);
}

PoolColorArray::~PoolColorArray() {
	godot::api->godot_pool_color_array_destroy(&_godot_array);
}

} // namespace godot
