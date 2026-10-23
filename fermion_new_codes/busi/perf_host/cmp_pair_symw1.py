import importlib.util, os, sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("C", "<internal>/cmp_v36v_busi.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
C.ROOTS = tuple(C.ROOTS) + ("<internal>/SYMW1",)
a, b = sys.argv[1], sys.argv[2]
C.compare("%s vs %s" % (a, b), a, "fz", b, "fz", os.path.join(a, "prim.log"), os.path.join(b, "prim.log"))
