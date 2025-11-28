# Thanks for the fantastic help from: 
# https://github.com/agra-uni-bremen/microrv32/blob/master/microrv32/sw/basic-c/Makefile
# export PATH=$PATH:/opt/riscv/bin

all : main.c bootstrap.S
	riscv-none-embed-gcc -Og bootstrap.S sys.S main.c -o main.elf -march=rv32ima -mabi=ilp32 -mstrict-align -nostartfiles -ffreestanding -T link.ld
	
clean:
	rm -f main.lf

.PHONY: all clean