import os
import subprocess
import re
import random

from globals import *

def x86_analysis(program_inputs, benchmark_dir):

    print ("\tStarting analysis for x86 executable")

    x86_file      = os.path.join(benchmark_dir, "x86")
    isampling_output = os.path.join(benchmark_dir, "isampling.out") # this file holds the output of the isampling pin tool (sampling of instructions executed)
    instselect_output = os.path.join(benchmark_dir, "pintool.log") # this file holds the list of instructions to sample from (output of PINFI tools)

    # run PIN sampling tool on x86 executable (if isampling.out already exists, skip this as it is time consuming!)
    # if you need to generate a new isampling.out file, delete the old one
    if os.path.isfile(isampling_output):
        print ("\tisampling.out has already been generated for this benchmark, skipping PIN tool execution")
    else:
        print ("\tExecuting PIN tool to generate isampling.out")
        args = [PIN_BIN, "-t", PIN_TOOL, "--", x86_file]
        args.extend(program_inputs)
        try:
            p = subprocess.Popen(args, cwd=benchmark_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            p.wait()
        except:
            print ("\tError while executing PIN tool, command: " + " ".join(map(str,args)))
            return -1, -1 # move onto next folder if exception is thrown
    
    # run PINFI tools on x86 executable (if pintool.log already exists, skip this as it is time consuming!)
    # if you need to generate a new pintool.log file, delete the old one
    if os.path.isfile(instselect_output):
        print ("\tpintool.log has already been generated for this benchmark, skipping PINFI tools execution")
    else:
        print ("\tExecuting PINFI tools to generate pintool.log")
        args = [PIN_BIN, "-t", PIN_CATEGORY, "--", x86_file]
        args.extend(program_inputs)
        try:
            p = subprocess.Popen(args, cwd=benchmark_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            p.wait()
        except:
            print ("\tError while executing PINFI tools, command: " + " ".join(map(str,args)))
            return -1, -1 # move onto next folder if exception is thrown
        args = [PIN_BIN, "-t", PIN_COUNT, "--", x86_file]
        args.extend(program_inputs)
        try:
            p = subprocess.Popen(args, cwd=benchmark_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            p.wait()
        except:
            print ("\tError while executing PINFI tools, command: " + " ".join(map(str,args)))
            return -1, -1 # move onto next folder if exception is thrown

    # get list of instructions that are selected by PINFI from pintool.log
    instselect_list = []
    try:
        with open(instselect_output, "r") as f:
            for line in f:
                if "0x" in line:
                    PC = "0x" + line[6:12]
                    instselect_list.append(PC)
    except:
        print ("\tError reading pintool.log")
        return -1, -1 # move onto next folder if exception is thrown
    
    # get dictionary of executed instructions and their frequencies
    # instructions are limited to those chosen by PINFI (instselect)
    # we limit the sampling to a smaller number of samples
    #   format of dict is {PC: frequency}
    #   e.g., {0x40001e90: 211231, 0x40001e91: 42312}
    number_of_samples = 1000
    isampling_list_FULL = []
    try:
        with open(isampling_output, "r") as f:
            for line in f:
                if "0x4" in line:
                    PC = line.strip("\n")
                    if PC in instselect_list:
                        isampling_list_FULL.append(PC)
        if len(isampling_list_FULL) > number_of_samples:
            inst_list = random.sample(isampling_list_FULL, number_of_samples)
        else:
            inst_list = isampling_list_FULL.copy()
    except:
        print ("\tError reading isampling.out")
        return -1, -1 # move onto next folder if exception is thrown

    # disassemble x86 executable to intel syntax assembly code
    print ("\tDisassembling x86 executable")
    x86_dis_file = os.path.join(benchmark_dir, "x86_dis")
    os.system("objdump -d " + x86_file + " > " + x86_dis_file)
    with open(x86_dis_file, "r") as f:
        program_lines = f.readlines()
        program_lines.append("\n")
        program_lines.append("\n")

    # break up assembly code instructions into basic blocks
    print ("\tFinding memory address instructions")
    blocks = []
    start_line = 0
    for index, line in enumerate(program_lines):
        if line[0:10] == "0000000000":
            start_line = index + 1
        elif (line == "\n"):
            end_line = index
            if (end_line - start_line) > 1:
                blocks.append((start_line, end_line))
            start_line = index + 1
        else:
            try:
                opcode = line.split("\t")[2].split()[0]
                if (opcode[0] == "j") or (opcode[0:4] == "call") or (opcode[0:3] == "ret"):
                    end_line = index + 1
                    if (end_line - start_line) > 1:
                        blocks.append((start_line, end_line))
                    start_line = index + 1
            except IndexError:
                pass

    # record memory address instructions by looping through each basic block
    # and keeping track of registers holding memory addresses
    mem_addr_inst = []
    static_total_count = 0
    for block in blocks:
        memreg_list = set()
        for line in reversed(program_lines[block[0]:block[1]]):
            # parse instructions
            try:
                instruction = line.split("\t")[2].strip("\n").split("# ")[0].strip()
                PC = "0x" + line.split(":")[0].strip()
            except IndexError:
                continue
            try:
                temp = instruction.split()
                opcode = temp[0]
                operands = re.split(r',\s*(?![^()]*\))', temp[1])
            except IndexError:
                pass
            try:
                dest = operands[-1]
                if len(operands) == 1:
                    sources = []
                elif len(operands) == 2:
                    sources = operands[0]
                else:
                    sources = operands[0:-1]
            except IndexError:
                pass
            
            static_total_count += 1
            
            # classify instructions
            if "mov" in opcode:
                if ("(" in dest) and (")" in dest): # store instruction
                    # add registers inside the memory address (destination operand) to memreg_list
                    temps = dest[dest.find("(")+1:dest.find(")")].split(",")
                    for addr in temps:
                        if "%" in addr:
                            memreg_list.add(addr)
                    #continue
                elif ("(" in sources) and (")" in sources): # load instruction
                    # add registers inside the memory address (source operand) to memreg_list
                    temps1 = sources[sources.find("(")+1:sources.find(")")].split(",")
                    for addr in temps1:
                        if "%" in addr:
                            memreg_list.add(addr)
                    # record mem addr inst
                    temps2 = dest[dest.find("(")+1:dest.find(")")].split(",")
                    for addr in temps2:
                        if "%" in addr:
                            if addr in memreg_list:
                                mem_addr_inst.append(PC)
                                #memreg_list.discard(addr)
                    #continue
            # add registers inside the memory address (destination operand) to memreg_list
            if ("(" in dest) and (")" in dest):
                # add registers inside the memory address (destination operand) to memreg_list
                temps = dest[dest.find("(")+1:dest.find(")")].split(",")
                for addr in temps:
                    if "%" in addr:
                        memreg_list.add(addr)
                #continue
            # check if destination register(s) is in memreg_list
            if dest in memreg_list:
                mem_addr_inst.append(PC)
                # add source register(s) to memreg_list
                if isinstance(sources, str):
                    sources_list = [sources]
                else:
                    sources_list = sources
                for addr in sources_list:
                    if "%" in addr:
                        memreg_list.add(addr)
    
    
    # obtain total frequency of memory address instructions based on the dynamic instruction sampling
    inst_total_count = 0
    mem_addr_total_count = 0
    for PC in inst_list:
        inst_total_count += 1
        if PC in mem_addr_inst:
            mem_addr_total_count += 1

    # calculate and return memory address instruction percentages
    print ("\tDynamic x86 count = " + str(mem_addr_total_count) + " / " + str(inst_total_count))
    
    static_mem_addr_count = len(mem_addr_inst)
    print ("\tStatic x86 count  = " + str(static_mem_addr_count) + " / " + str(static_total_count))

    dynamic_percent = mem_addr_total_count / inst_total_count
    static_percent  = static_mem_addr_count / static_total_count
    return  dynamic_percent, static_percent 
