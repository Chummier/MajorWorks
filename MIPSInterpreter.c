#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include "computer.h"
#undef mips			/* gcc already has a def for mips */

/* I've deleted parts of this code that I didn't write
My code for this program is an implementation of the five stages for a MIPS
assembly code processor-
1. Instruction fetching
2. Instruction Decoding
3. Execution
4. Memory load
5. Memory write

The program takes in a MIPS program and simulates an actual memory space
and uses memory addresses to store data at
*/


/*
 *  Return the contents of memory at the given address. Simulates
 *  instruction fetch. 
 */
unsigned int Fetch ( int addr) {
    return mips.memory[(addr-0x00400000)/4];
}

void printIName(int opcode){
    if (opcode == 9){
        printf("addiu ");
    } else if (opcode == 12){
        printf("andi ");
    } else if (opcode == 4){
        printf("beq ");
    } else if (opcode == 5){
        printf("bne ");
    } else if (opcode == 15){
        printf("lui ");
    } else if (opcode == 35){
        printf("lw ");
    } else if (opcode == 13){
        printf("ori ");
    } else if (opcode == 43){
        printf("sw ");
    } else {
        
        exit(1);
    }
}

void printRName(int funct){
    if (funct == 33){
        printf("addu ");
    } else if (funct == 36){
        printf("and ");
    } else if (funct == 8){
        printf("jr ");
    } else if (funct == 37){
        printf("or ");
    } else if (funct == 42){
        printf("slt ");
    } else if (funct == 0){
        printf("sll ");
    } else if (funct == 2){
        printf("srl ");
    } else if (funct == 35){
        printf("subu ");
    }  else {
        exit(1);

    }
}

/* Decode instr, returning decoded instruction. */
void Decode ( unsigned int instr, DecodedInstr* d, RegVals* rVals) {
    /* Your code goes here */

    // Get the first 6 bits then shift it over to the right to easily get the opcode in decimal
    int opcode = (instr&0xfc000000)>>26;
    (*d).op = opcode;
    if (instr == 0){                                    
        exit(1);
    }

    if (opcode == 0){
        (*d).type = R;
    } else if (opcode == 2 || opcode == 3){
        (*d).type = J;
    } else {
        (*d).type = I;
    }

    if (opcode == 2 || opcode == 3){
        (*d).type = J;
        (*d).regs.j.target = (instr&0x03ffffff)<<2;
    } else if (opcode == 0){
        (*d).regs.r.rs = (instr&0x03e00000)>>21;
        (*rVals).R_rs = mips.registers[(*d).regs.r.rs];

        (*d).regs.r.rt = (instr&0x001f0000)>>16;
        (*rVals).R_rt = mips.registers[(*d).regs.r.rt];

        (*d).regs.r.rd = (instr&0x0000f800)>>11;
        (*rVals).R_rd = mips.registers[(*d).regs.r.rd];

        (*d).regs.r.shamt = (instr&0x000007c0)>>6;
        (*d).regs.r.funct = (instr&0x0000003f);
    } else {
        (*d).regs.i.rs = (instr&0x03e00000)>>21;
        (*rVals).R_rs = mips.registers[(*d).regs.i.rs];

        (*d).regs.i.rt = (instr&0x001f0000)>>16;
        (*rVals).R_rt = mips.registers[(*d).regs.i.rt];

        if ((instr&0x00008000) == 0x00008000){
            (*d).regs.i.addr_or_immed = (instr&0x00007fff)-0x00008000;
        } else {
            (*d).regs.i.addr_or_immed = (instr&0x00007fff);
        }

    }
}

/*
 *  Print the disassembled version of the given instruction
 *  followed by a newline.
 */
void PrintInstruction ( DecodedInstr* d) {
    /* Your code goes here */

        // Print j
    if ((*d).op == 2){
        printf("j 0x%8.8x\n", (*d).regs.j.target);

        // Print jal
    } else if ((*d).op == 3){
        printf("jal 0x%8.8x\n", (*d).regs.j.target);

        // Print R-instructions
    } else if ((*d).op == 0){
        int func = (*d).regs.r.funct;
        printRName(func);

            // add, addu, and, nor, or, slt, sltu, sub, subu
        if (func > 31 && func < 44){
            printf("\t$%d, $%d, $%d", (*d).regs.r.rd, (*d).regs.r.rs, (*d).regs.r.rt);

            // div, divu, mult, multu
        } else if (func > 23 && func < 28){
            printf("\t$%d, $%d", (*d).regs.r.rs, (*d).regs.r.rt);

            // mfhi, mflo
        } else if (func == 16 || func == 18){
            printf("\t$%d", (*d).regs.r.rd);

            // sll, srl, sra
        } else if (func > -1 && func < 4){
            printf("\t$%d, $%d, %d", (*d).regs.r.rd, (*d).regs.r.rt, (*d).regs.r.shamt);

            // jr
        } else {
            printf("\t$%d", (*d).regs.r.rs);
        }
        printf("\n");

        // Print I-instructions
    } else {
        int operation = (*d).op;
        printIName(operation);

            // beq, bne
        if (operation == 4 || operation == 5){
            printf("\t$%d, $%d, 0x%8.8x", (*d).regs.i.rs, (*d).regs.i.rt, mips.pc + 4 + ((*d).regs.i.addr_or_immed)*4);

            // sb, sw, lw
        } else if (operation == 40 || operation == 43 || operation == 35){
            printf("\t$%d, %d($%d)", (*d).regs.i.rt, (*d).regs.i.addr_or_immed, (*d).regs.i.rs);

            // lui
        } else if (operation == 15){
            printf("\t$%d, %d", (*d).regs.i.rt, (*d).regs.i.addr_or_immed);

            // addi, addiu, ori, slti, sltiu, andi
        } else if (operation > 7 && operation < 14){
            printf("\t$%d, $%d, %d", (*d).regs.i.rt, (*d).regs.i.rs, (*d).regs.i.addr_or_immed);

        }
        printf("\n");
    }
}

/* Perform computation needed to execute d, returning computed value */
int Execute ( DecodedInstr* d, RegVals* rVals) {
    /* Your code goes here */
    int opcode = (*d).op;
        // R-type
    if (opcode == 0){
        int func = (*d).regs.r.funct;

            // jr
        if (func == 8){
            return (*rVals).R_rs;

            // addu
        } else if (func == 33){
            return (*rVals).R_rs + (*rVals).R_rt;

            // and
        }  else if (func == 36){
            return (*rVals).R_rs & (*rVals).R_rt;

            // or
        } else if (func == 37){
            return (*rVals).R_rs | (*rVals).R_rt;

            // slt 
        } else if (func == 42){
            int temp = (*rVals).R_rs - (*rVals).R_rt;
            if (temp < 0) {
                return 1;
            } else {
                return 0;
            }

            // sll
        } else if (func == 0){
            return (*rVals).R_rt << (*d).regs.r.shamt;

            // srl
        } else if (func == 2){
            return (*rVals).R_rt >> (*d).regs.r.shamt;

            // subu
        }  else if (func == 35){
            return (*rVals).R_rs - (*rVals).R_rt;
        }

        // beq
    } else if (opcode == 4){
        if ((*rVals).R_rs == (*rVals).R_rt){
            return 1;
        } else {
            return 0;
        }

        // bne 
    } else if (opcode == 5){
        if ((*rVals).R_rs == (*rVals).R_rt){
            return 0;
        } else {
            return 1;
        }

        // addiu
    } else if (opcode == 9){
        return (*rVals).R_rs + (*d).regs.i.addr_or_immed;

        // ori
    } else if (opcode == 13){
        return (*rVals).R_rs | (0x00000000&((*d).regs.i.addr_or_immed));

        // lui
    } else if (opcode == 15){
        return 0xffff0000&((*d).regs.i.addr_or_immed<<16);

        // andi
    } else if (opcode == 12){
        return (*rVals).R_rs & (*d).regs.i.addr_or_immed;

        // lw, sw
    } else if (opcode == 35 || opcode == 43){
        return (*rVals).R_rs + (*d).regs.i.addr_or_immed;

        // jal
    }  else if (opcode == 3){
        return mips.pc+4;
    }
  return 0;
}

/* 
 * Update the program counter based on the current instruction. For
 * instructions other than branches and jumps, for example, the PC
 * increments by 4 (which we have provided).
 */
void UpdatePC ( DecodedInstr* d, int val) {
    mips.pc+=4;
    /* Your code goes here */

        // jr 
    if ((*d).op == 0 && (*d).regs.r.funct == 8){
        mips.pc = val;

        // j, jal
    } else if ((*d).op == 2 || (*d).op == 3){
        mips.pc = 0x0ffffffc & ((*d).regs.j.target);

        // take beq or bne
    } else if ((*d).op == 4 || (*d).op == 5){
        if (val == 1){
            mips.pc += ((*d).regs.i.addr_or_immed)*4;
        }
    }
}

/*
 * Perform memory load or store. Place the address of any updated memory 
 * in *changedMem, otherwise put -1 in *changedMem. Return any memory value 
 * that is read, otherwise return -1. 
 *
 * Remember that we're mapping MIPS addresses to indices in the mips.memory 
 * array. mips.memory[0] corresponds with address 0x00400000, mips.memory[1] 
 * with address 0x00400004, and so forth.
 *
 */
int Mem( DecodedInstr* d, int val, int *changedMem) {
    *changedMem = -1;

        // lw
    if ((*d).op == 35){
        return mips.memory[(val - 0x00400000)/4];

        // sw
    } else if ((*d).op == 43){
        mips.memory[(val - 0x00400000)/4] = mips.registers[(*d).regs.r.rt];
        *changedMem = val;
        return -1;
    }

  return val;
}

/* 
 * Write back to register. If the instruction modified a register--
 * (including jal, which modifies $ra) --
 * put the index of the modified register in *changedReg,
 * otherwise put -1 in *changedReg.
 */
void RegWrite( DecodedInstr* d, int val, int *changedReg) {
    *changedReg = -1;

        // jal
    if ((*d).op == 3){
        mips.registers[31] = val;
        *changedReg = 31;

        // addu, subu, sll, srl, and, or, slt
    } else if ((*d).op == 0 && (*d).regs.r.funct != 8){
        mips.registers[(*d).regs.r.rd] = val;
        *changedReg = (*d).regs.r.rd;

        // addiu, andi, ori, lui
    } else if ((*d).op > 8 && (*d).op < 16){
        mips.registers[(*d).regs.i.rt] = val;
        *changedReg = (*d).regs.i.rt;

        // lw
    } else if ((*d).op == 35){
        mips.registers[(*d).regs.i.rt] = val;
        *changedReg = (*d).regs.i.rt;
    }
}
