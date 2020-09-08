# A MIPS assembly program that 
# MIPSInterpreter.c would execute

.text
addiu $t0, $zero, -3
addiu $t1, $zero, 5
addu $t2, $t0, $t1
subu $t2, $t1, $t0
sll $t2, $t2, 8
srl $t2, $t2, 4
and $t1, $t2, $t0
andi $t3, $sp, 0xabcdef12
or $t3, $zero, $t3
ori $t3, $zero, 0x12345678
lui $s0, 0x1234
slt $t1, $t1, $s0
beq $at, $t3, label
label:
bne $t0, $t1, label2
label2:
jal label4
sw $t2, -4($sp)
lw $t7, -4($sp)
j label3
label4:
jr $ra
label3:
