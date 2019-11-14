import os
  
ROOT_DIR = os.path.dirname( os.path.realpath(__file__) )
BENCHMARKS_DIR = os.path.join(ROOT_DIR, "benchmarks")

# set PIN root directories, make sure these are correct to your setup
PIN_DIR      = "/data/installs/PIN" # this should be the location of your PIN installation folder
PIN_BIN      = os.path.join(PIN_DIR, "pin")
PIN_TOOL     = os.path.join(PIN_DIR, "source/tools/memaddr-pintool/obj-intel64/isampling.so")
PIN_CATEGORY = os.path.join(PIN_DIR, "source/tools/memaddr-pintool/obj-intel64/instcategory.so")
PIN_COUNT    = os.path.join(PIN_DIR, "source/tools/memaddr-pintool/obj-intel64/instcount.so")
