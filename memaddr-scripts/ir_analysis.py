import os
import re

from globals import *

def ir_analysis(benchmark_dir):

    print ("\tStarting analysis for LLVM IR code")

    # read llfi-indexed version of IR file and save a list of the IR instructions 
    ir_file = os.path.join(benchmark_dir, "ir.ll")
    try:
        print ("\tReading IR file")
        with open(ir_file, "r") as f:
            llvm_lines = f.readlines()
    except:
        print ("\tError reading ir.ll")
        return -1, -1 # move onto next folder if exception is thrown

    # break up IR code instructions into basic blocks
    print ("\tFinding memory address instructions")
    blocks = []
    start_line = 0
    for index, line in enumerate(llvm_lines):
        if line.strip() == "":
            end_line = index
            blocks.append((start_line, end_line))
            start_line = index + 1
        if "!0 = metadata" in line:
            map_start = index
    
    # create mapping to be consistent with a  metadata map used by LLFI to index instructions
    metadata_map_lines = llvm_lines[map_start:]
    metadata_map = {}
    for line in metadata_map_lines:
        if "!{i64 " in line:
            map_from = line.split("!{i64 ")[1].split("}")[0]
            map_to   = line.split("!")[1].split(" = ")[0]
            metadata_map[map_from] = map_to

    # go through each basic block and record all memory dependent instructions
    fi_indexes = []
    static_counter_total = 0
    for block in blocks:
        addresses = []
        for line in reversed(llvm_lines[block[0]:block[1]]):
            static_counter_total += 1
            try:
                if line[0:7] == "  store":
                    temp = re.sub('<.*?>', 'vector', line)
                    addr = temp.split(", ")[1].split(" ")[-1]
                    if "%" in addr:
                        addresses.append(addr)
                elif line.split(" ")[4] == "load":
                    if not "load double* getelementptr inbounds" in line:
                        addr = line.split(", ")[0].split(" ")[-1]
                        if "%" in addr:
                            addresses.append(addr)
            except IndexError:
                pass
            dest_reg = line.split("=")[0].replace(" ", "")
            if dest_reg in addresses:
                if "!llfi_index !" in line:
                    fi_indexes.append(line.strip("\n").split("!llfi_index !")[1])
                    addresses = addresses + re.findall(r'[%]\w*\b', line.split("=")[1])

    # use LLFI FI logs as instruction sampling to count frequency of memory address instructions
    # read from either llfi.stat.fi.injectedfaults.txt file or from LLFI stat directory
    counter_mem = 0
    counter_total = 0
    try:
        llfi_stat_file = os.path.join(benchmark_dir, "llfi.stat.fi.injectedfaults.txt")
        with open(llfi_stat_file) as f:
            fi_stats = f.readlines()
        for fi_stat in fi_stats:
            fi_run_index = fi_stat.split("fi_index=")[1].split(",")[0]
            try:
                mapped_index = metadata_map[fi_run_index]
                if mapped_index in fi_indexes:
                    counter_mem = counter_mem + 1
            except KeyError:
                print ("***** Index " + index + " not in map. *****")
            counter_total = counter_total + 1
    except FileNotFoundError:
        llfi_stat_dir = os.path.join(benchmark_dir, "llfi_stat_output")
        if not os.path.isdir(llfi_stat_dir):
            print ("\tLLFI logs not found for this benchmark")
            return -1, -1 # move onto next folder if logs are not found
        for fi_run in os.listdir(llfi_stat_dir):
            with open(os.path.join(llfi_stat_dir, fi_run), 'r') as f:
                fi_stat = f.read()
            fi_run_index = fi_stat.split("fi_index=")[1].split(",")[0]
            try:
                mapped_index = metadata_map[fi_run_index]
                if mapped_index in fi_indexes:
                    counter_mem = counter_mem + 1
            except KeyError:
                print ("***** Index " + index + " not in map. *****")
            counter_total = counter_total + 1

    # calculate and return memory address instruction percentage
    print ("\tDynamic IR count = " + str(counter_mem) + " / " + str(counter_total))
    
    static_counter_mem = len(fi_indexes)
    print ("\tStatic IR count  = " + str(static_counter_mem) + " / " + str(static_counter_total))
    
    dynamic_percent = counter_mem / counter_total
    static_percent = static_counter_mem / static_counter_total
    return dynamic_percent, static_percent