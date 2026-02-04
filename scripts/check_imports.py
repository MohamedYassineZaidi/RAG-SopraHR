import ctypes, sys
print('python', sys.executable)
for d in ['msvcp140.dll','VCRUNTIME140_1.dll','VCRUNTIME140.dll']:
    try:
        ctypes.WinDLL(d)
        print(d, 'OK')
    except Exception as e:
        print(d, 'MISSING', e)

for mod in ('pymupdf','greenlet','playwright'):
    try:
        m = __import__(mod)
        print(mod, 'import OK', getattr(m, '__file__', getattr(m, '__path__', None)))
    except Exception as e:
        print(mod, 'import FAILED:', e)
