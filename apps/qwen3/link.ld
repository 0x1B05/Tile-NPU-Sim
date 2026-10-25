ENTRY(_start)

SECTIONS
{
    . = 0x80000000;
    .text : { *(.text.crt0) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    }
    ASSERT(__bss_end <= 0x80F00000, "image overlaps the Q3P prompt region")
    /DISCARD/ : { *(.note*) *(.comment*) *(.eh_frame*) *(.riscv.attributes) }
}
