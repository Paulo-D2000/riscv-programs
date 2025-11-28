/* Thanks for the fantastic help from: 
https://github.com/agra-uni-bremen/microrv32/blob/master/microrv32/sw/basic-c/link.ld*/

OUTPUT_ARCH( "riscv" )
ENTRY(_bootstrap)

/* and Discord user dutracgi,
https://interrupt.memfault.com/blog/how-to-write-linker-scripts-for-firmware */

/* Needs to match mem_map.h (TODO how to share variables?) */
MEMORY
{
  ram      (rw) : ORIGIN = 0x41000000, LENGTH = 0x10000000
}

STACK_SIZE = 131072; /*Default is 4096 for gcc? whats a good size?*/

/* Section Definitions */
SECTIONS
{
    .text :
    {
        KEEP(*(.vectors .vectors.*))
        *(.text*)
    } > ram

    /* .bss section which is used for uninitialized data */
    /* Could this be in wrong place somehow? */
    /*.bss (NOLOAD) :
    {
        *(.bss*)
        *(COMMON)
    } > ram*/

    .data :
    {
        *(.data*)
        *(.sdata*)
        *(.rodata*)
        *(.srodata*)
        *(.bss*) /* .bss section which is used for uninitialized data */
        *(COMMON)
    } > ram

    /* stack section */
    .stack (NOLOAD):
    {
        . = ALIGN(4);
        stack_begin = .; /* stack base is here */
        . = . + STACK_SIZE;
        . = ALIGN(4);
        stack_end = .; /* stack top is here <- sp */
    } > ram

    _end = . ;
}