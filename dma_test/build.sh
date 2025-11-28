export PATH=$PATH:/home/popo/.local/xPacks/riscv-none-embed-gcc/xpack-riscv-none-embed-gcc-10.2.0-1.2/bin
make clean
make all
riscv-none-embed-objcopy -O binary main.elf program.bin
riscv-none-embed-objdump -D main.elf > program.sym
cp program.bin ~/Downloads/imrisc