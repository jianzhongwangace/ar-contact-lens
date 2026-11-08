"""Read DXGI adapter indices used by DirectML; no graphics context or capture."""
import ctypes as c
import os
import uuid

def adapters():
    if os.name!='nt':return []
    class Desc(c.Structure):
        _fields_=[('Description',c.c_wchar*128),('VendorId',c.c_uint32),
                  ('DeviceId',c.c_uint32),('SubSysId',c.c_uint32),('Revision',c.c_uint32),
                  ('DedicatedVideoMemory',c.c_size_t),('DedicatedSystemMemory',c.c_size_t),
                  ('SharedSystemMemory',c.c_size_t),('LuidLow',c.c_uint32),('LuidHigh',c.c_int32),
                  ('Flags',c.c_uint32)]
    def method(pointer,index,result,*types):
        table=c.cast(pointer,c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(result,c.c_void_p,*types)(table[index])
    factory=c.c_void_p()
    iid=(c.c_ubyte*16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    create=c.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes=[c.c_void_p,c.POINTER(c.c_void_p)]
    create.restype=c.c_int32
    if create(c.byref(iid),c.byref(factory))<0:return []
    result=[]
    try:
        for index in range(16):
            adapter=c.c_void_p()
            if method(factory,12,c.c_int32,c.c_uint32,c.POINTER(c.c_void_p))(factory,index,c.byref(adapter))<0:
                break
            try:
                desc=Desc()
                if method(adapter,10,c.c_int32,c.POINTER(Desc))(adapter,c.byref(desc))>=0:
                    result.append({'index':index,'name':desc.Description,'vendor':desc.VendorId,
                                   'memory_mib':desc.DedicatedVideoMemory//(1024*1024),'software':bool(desc.Flags&2)})
            finally:method(adapter,2,c.c_uint32)(adapter)
    finally:method(factory,2,c.c_uint32)(factory)
    return result

def preferred_adapter():
    devices=[d for d in adapters() if not d['software']]
    if not devices:raise RuntimeError('No hardware DXGI adapter found')
    return max(devices,key=lambda d:(d['vendor']==0x10de,d['memory_mib']))

if __name__=='__main__':
    import json
    print(json.dumps(adapters(),indent=2))
