#include <Python.h>

int Add(int x, int y)
{
 return x + y;
}

int Del(int x, int y)
{
 return x - y;
}

PyObject* WrappAdd(PyObject* self, PyObject* args)
{
 int x, y;
 if (!PyArg_ParseTuple(args, "ii", &x, &y))
 {
  return NULL;
 }
 return Py_BuildValue("i", Add(x, y));
}

PyObject* WrappDel(PyObject* self, PyObject* args)
{
 int x, y;
 if (!PyArg_ParseTuple(args, "ii", &x, &y))
 {
  return NULL;
 }
 return Py_BuildValue("i", Del(x, y));
}
static PyMethodDef test_methods[] = {
 {"Add", WrappAdd, METH_VARARGS, "something"},
 {"Del", WrappDel, METH_VARARGS, "something"},
 {NULL, NULL}
};

extern "C"
void inittest1()
{
 Py_InitModule("test1", test_methods);
}
