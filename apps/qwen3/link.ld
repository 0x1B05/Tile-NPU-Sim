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
    ASSERT(__bss_end <= 0x80B00000, "image overlaps the snapshot region")
    /DISCARD/ : { *(.note*) *(.comment*) *(.eh_frame*) *(.riscv.attributes) }
}
