from __future__ import annotations

import ctypes
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import pypcode
except ImportError:
    pypcode = None

try:
    import unicorn
    from unicorn import arm_const
except ImportError:
    unicorn = None
    arm_const = None


# Self-authored ARM/Thumb instruction fixtures. No Nokia code or data is used.
INSTRUCTIONS = {
    "arm_cmp32": (False, "600ab4ee", "fpscr"),
    "arm_cmpe32": (False, "e00ab4ee", "fpscr"),
    "arm_cmp64": (False, "410bb4ee", "fpscr"),
    "arm_cmpe64": (False, "c10bb4ee", "fpscr"),
    "arm_cmpzero32": (False, "400ab5ee", "fpscr"),
    "arm_abs64": (False, "c10bb0ee", "d0"),
    "arm_f32_to_f64": (False, "e00ab7ee", "d0"),
    "arm_f64_to_f32": (False, "c10bb7ee", "s0"),
    "thumb_cmp32": (True, "b4ee600a", "fpscr"),
    "thumb_cmpe32": (True, "b4eee00a", "fpscr"),
    "thumb_cmp64": (True, "b4ee410b", "fpscr"),
    "thumb_cmpe64": (True, "b4eec10b", "fpscr"),
    "thumb_cmpzero32": (True, "b5ee400a", "fpscr"),
    "thumb_abs64": (True, "b0eec10b", "d0"),
    "thumb_f32_to_f64": (True, "b7eee00a", "d0"),
    "thumb_f64_to_f32": (True, "b7eec10b", "s0"),
}

SEQUENCES = {
    "arm_halves_to_double": (False, "100a00ee901a00ee401bb0ee201a30ee", "reg_d1"),
    "thumb_halves_to_double": (True, "00ee100a00ee901ab0ee401b30ee201a", "reg_d1"),
    "arm_double_to_halves": (False, "410bb0ee102a10ee903a10ee", "(reg_r3<<32)|reg_r2"),
    "thumb_double_to_halves": (True, "b0ee410b10ee102a10ee903a", "(reg_r3<<32)|reg_r2"),
    "arm_upper_halves": (False, "100a0fee901a0fee4febb0ee", "reg_d14"),
    "thumb_upper_halves": (True, "0fee100a0fee901ab0ee4feb", "reg_d14"),
}


def helper_source() -> str:
    from tools.make_frontend_aot_seed import make_seed

    source = """\
typedef struct { int dummy; } NokiaFrontendMachine;
static uint8_t *segment(NokiaFrontendMachine*m,uint32_t a,unsigned n) { return 0; }
static int dummy(void){
    uint64_t reg_pc=0;
    if(!reg_pc)reg_pc=UINT64_C(1);
dispatch:
    switch((uint32_t)reg_pc&~1u){
    default: goto yielded;
    }
finished:
    return 1;
yielded:
    return 0;
}
"""
    seed = make_seed(source, 0x1000)
    return seed[seed.index("static uint64_t nokia_float_add"):
                seed.index("static uint8_t *segment")]


@unittest.skipUnless(pypcode is not None and shutil.which("cc"),
                     "pypcode and a C compiler are required")
class ArmVfpAotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tools.extend_prime_aot import cname, emit_instruction

        functions = []
        cls.translations = {}
        for name, (thumb, hex_code, result) in INSTRUCTIONS.items():
            raw = bytes.fromhex(hex_code)
            context = pypcode.Context("ARM:LE:32:v8T" if thumb else "ARM:LE:32:v8")
            operations = list(context.translate(raw, base_address=0x1000,
                                                max_instructions=1).ops)
            cls.translations[name] = operations
            variables = set()
            for operation in operations:
                nodes = ([operation.output] if operation.output is not None else []) + list(operation.inputs)
                variables.update(cname(node) for node in nodes
                                 if node.space.name in ("register", "unique"))
            initial = {"reg_d0": "a", "reg_d1": "b", "reg_fpscr": "flags"}
            if name.endswith("cmp32") or name.endswith("cmpe32"):
                initial["reg_d0"] = "(a&UINT64_C(0xffffffff))|(b<<32)"
            if name.endswith("f32_to_f64"):
                initial["reg_d0"] = "b<<32"
            declarations = "\n".join(f"    uint64_t {name}={initial.get(name, '0')};"
                                      for name in sorted(variables))
            statements = "\n".join("    " + statement for statement in emit_instruction(
                operations, set(), 0x1000, len(raw), thumb, raw))
            result_value = "(reg_d0 & UINT64_C(0xffffffff))" if result == "s0" else f"reg_{result}"
            functions.append(f"""\
uint64_t {name}(uint64_t a,uint64_t b,uint64_t flags){{
{declarations}
{statements}
    return {result_value};
}}
""")
        for name, (thumb, hex_code, result) in SEQUENCES.items():
            raw = bytes.fromhex(hex_code)
            variables, rendered = set(), []
            for offset in range(0, len(raw), 4):
                instruction = raw[offset:offset + 4]
                context = pypcode.Context("ARM:LE:32:v8T" if thumb else "ARM:LE:32:v8")
                operations = list(context.translate(instruction, base_address=0x1000 + offset,
                                                    max_instructions=1).ops)
                for operation in operations:
                    nodes = ([operation.output] if operation.output is not None else []) + list(operation.inputs)
                    variables.update(cname(node) for node in nodes
                                     if node.space.name in ("register", "unique"))
                rendered.extend(emit_instruction(operations, set(), 0x1000 + offset, 4,
                                                 thumb, instruction))
            initial = {"reg_r0": "a", "reg_r1": "b", "reg_d1": "a" if "double_to_halves" in name else "0"}
            declarations = "\n".join(f"    uint64_t {variable}={initial.get(variable, '0')};"
                                      for variable in sorted(variables))
            statements = "\n".join("    " + statement for statement in rendered)
            functions.append(f"""\
uint64_t {name}(uint64_t a,uint64_t b,uint64_t flags){{
{declarations}
{statements}
    return {result};
}}
""")
        source = ("#include <stdint.h>\n#include <string.h>\n" + helper_source() +
                  "static uint64_t nokia_shl(uint64_t v,uint64_t n,unsigned s)"
                  "{return n>=8*s?0:v<<n;}\n" + "\n".join(functions))
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        c_file, library = root / "vfp.c", root / "vfp.so"
        c_file.write_text(source, encoding="utf-8")
        subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-shared", "-fPIC",
                        str(c_file), "-o", str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        for name in (*INSTRUCTIONS, *SEQUENCES):
            function = getattr(cls.library, name)
            function.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            function.restype = ctypes.c_uint64

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def call(self, name: str, a: int, b: int, flags: int = 0) -> int:
        return getattr(self.library, name)(a, b, flags)

    def test_comparisons_and_nan_status_in_arm_and_thumb(self) -> None:
        cases32 = [(0xbf800000, 0x3f800000, 0x80000000, 0),
                   (0, 0x80000000, 0x60000000, 0),
                   (0x40000000, 0x3f800000, 0x20000000, 0),
                   (0x7f800000, 0x7f800000, 0x60000000, 0),
                   (0x7fc00000, 0x3f800000, 0x30000000, 1),
                   (0x7f800001, 0x3f800000, 0x30000000, 2)]
        cases64 = [(0xbff0000000000000, 0x3ff0000000000000, 0x80000000, 0),
                   (0, 0x8000000000000000, 0x60000000, 0),
                   (0x4000000000000000, 0x3ff0000000000000, 0x20000000, 0),
                   (0x7ff8000000000000, 0x3ff0000000000000, 0x30000000, 1),
                   (0x7ff0000000000001, 0x3ff0000000000000, 0x30000000, 2)]
        preserved = 0x08000012
        for mode in ("arm", "thumb"):
            for bits, cases in ((32, cases32), (64, cases64)):
                for a, b, expected, nan_kind in cases:
                    for signalling in (False, True):
                        name = f"{mode}_{'cmpe' if signalling else 'cmp'}{bits}"
                        invalid = nan_kind == 2 or (signalling and nan_kind == 1)
                        with self.subTest(name=name, a=hex(a), b=hex(b)):
                            self.assertEqual(self.call(name, a, b, preserved),
                                             preserved | expected | int(invalid))

    def test_compare_zero_and_absolute_value(self) -> None:
        for mode in ("arm", "thumb"):
            self.assertEqual(self.call(f"{mode}_cmpzero32", 0xbf800000, 0), 0x80000000)
            self.assertEqual(self.call(f"{mode}_cmpzero32", 0x7fc00000, 0), 0x30000000)
            self.assertEqual(self.call(f"{mode}_abs64", 0, 0xbff8000000000000),
                             0x3ff8000000000000)
            self.assertEqual(self.call(f"{mode}_abs64", 0, 0xfff8000000000001),
                             0x7ff8000000000001)

    def test_comparisons_do_not_depend_on_fast_math(self) -> None:
        source = ("#include <stdint.h>\n#include <string.h>\n" + helper_source() +
                  "int main(void){return !(nokia_float_compare(0x7fc00000u,0x3f800000u,4,0)==0 && "
                  "nokia_float_compare(0x80000000u,0,4,0)==1 && "
                  "nokia_float_compare(0x80000001u,0,4,1)==1);}\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c_file, executable = root / "fast.c", root / "fast"
            c_file.write_text(source, encoding="utf-8")
            subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-ffast-math",
                            str(c_file), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)

    def test_float_width_conversions_at_default_rounding(self) -> None:
        for mode in ("arm", "thumb"):
            self.assertEqual(self.call(f"{mode}_f32_to_f64", 0, 0x3fc00000),
                             0x3ff8000000000000)
            self.assertEqual(self.call(f"{mode}_f32_to_f64", 0, 0x80000000),
                             0x8000000000000000)
            self.assertEqual(self.call(f"{mode}_f64_to_f32", 0, 0xbff8000000000000),
                             0xbfc00000)
            self.assertEqual(self.call(f"{mode}_f64_to_f32", 0, 0x7ff0000000000000),
                             0x7f800000)

    def test_single_and_double_registers_share_storage(self) -> None:
        for mode in ("arm", "thumb"):
            self.assertEqual(self.call(f"{mode}_halves_to_double", 0x3f800000, 0x40000000),
                             0x4000000040400000)
            self.assertEqual(self.call(f"{mode}_double_to_halves", 0x1122334455667788, 0),
                             0x1122334455667788)
            self.assertEqual(self.call(f"{mode}_upper_halves", 0x55667788, 0x11223344),
                             0x1122334455667788)

    def test_rejects_an_unrecognised_compare_encoding(self) -> None:
        from tools.extend_prime_aot import emit_instruction

        ops = self.translations["arm_cmp32"]
        with self.assertRaisesRegex(ValueError, "unrecognised ARM VFP comparison"):
            emit_instruction(ops, set(), 0x1000, 4, False, b"\x00\x00")
        with self.assertRaisesRegex(ValueError, "unrecognised ARM VFP comparison encoding"):
            emit_instruction(ops, set(), 0x1000, 4, False, b"\x00\x00\x00\x00")

    def test_missing_helpers_are_rejected_before_emitting_c(self) -> None:
        from tools.extend_prime_aot import require_float_helpers

        with self.assertRaisesRegex(ValueError, "seed lacks floating-point helpers"):
            require_float_helpers("", ["reg_d0 = nokia_float_add(reg_d0,reg_d1,8);"])
        with self.assertRaisesRegex(ValueError, "seed lacks floating-point helpers"):
            require_float_helpers("uint64_t nokia_float_add(uint64_t, uint64_t, unsigned);",
                                  ["reg_d0 = nokia_float_add(reg_d0,reg_d1,8);"])
        require_float_helpers(helper_source(), ["reg_d0 = nokia_float_add(reg_d0,reg_d1,8);"])

    def test_unsupported_float_widths_fail_instead_of_returning_zero(self) -> None:
        from tools.extend_prime_aot import emit

        register = types.SimpleNamespace(space=types.SimpleNamespace(name="register"), size=2)
        boolean = types.SimpleNamespace(space=types.SimpleNamespace(name="register"), size=1)
        operation = types.SimpleNamespace(opcode=types.SimpleNamespace(name="FLOAT_EQUAL"),
                                          output=boolean, inputs=[register, register])
        with self.assertRaisesRegex(ValueError, "unsupported FLOAT_EQUAL width"):
            emit(operation, set(), 0x1000, 4, False)

    def test_trace_generator_compiles_and_executes_a_synthetic_vcmp(self) -> None:
        from tools.make_frontend_aot_seed import make_seed

        template = """\
#include <stdint.h>
#include <string.h>
typedef struct { int unused; } NokiaFrontendMachine;
static uint32_t nokia_frontend_last_pc;
static uint64_t nokia_shl(uint64_t v,uint64_t n,unsigned s){return n>=8*s?0:v<<n;}
static uint8_t *segment(NokiaFrontendMachine*m,uint32_t a,unsigned n) {return 0;}
uint32_t synthetic_compare(uint32_t left,uint32_t right,uint32_t zero,uint32_t old_flags){
    uint64_t reg_pc=0;
    uint64_t reg_d0=0;
    uint64_t reg_fpscr=0;
    uint64_t reg_ZR=0;
    reg_d0=(uint64_t)left|((uint64_t)right<<32);
    reg_ZR=zero;reg_fpscr=old_flags;
    if(!reg_pc)reg_pc=UINT64_C(1);
dispatch:
    switch((uint32_t)reg_pc&~1u){
    default: goto yielded;
    }
finished:
    return (uint32_t)reg_fpscr;
yielded:
    return (uint32_t)reg_fpscr;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, trace, rom = (root / name for name in
                                ("seed.c", "trace.json", "instruction.bin"))
            seed.write_text(make_seed(template, 0x1000), encoding="utf-8")
            trace.write_text('{"instructions":[{"address":4096,"size":4,"thumb":false}]}',
                             encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "tools" / "generate_frontend_aot_from_trace.py"
            for variant, instruction in (("plain", "600ab4ee"), ("conditional", "600ab41e")):
                generated, library = root / f"{variant}.c", root / f"{variant}.so"
                rom.write_bytes(bytes.fromhex(instruction))
                subprocess.run([sys.executable, str(script), "--source", str(seed), "--trace", str(trace),
                                "--rom", str(rom), "--output", str(generated), "--rom-base", "0x1000",
                                "--default-entry", "0x1000", "--jobs", "1"], check=True,
                               capture_output=True, text=True)
                subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-shared", "-fPIC",
                                str(generated), "-o", str(library)], check=True)
                function = ctypes.CDLL(str(library)).synthetic_compare
                function.argtypes = [ctypes.c_uint32] * 4
                function.restype = ctypes.c_uint32
                self.assertEqual(function(0x7fc00000, 0x3f800000, 0, 0), 0x30000000)
                self.assertEqual(function(0x3f800000, 0x40000000, 0, 0), 0x80000000)
                if variant == "conditional":
                    self.assertEqual(function(0x7fc00000, 0x3f800000, 1, 0x60000012),
                                     0x60000012)

    @unittest.skipUnless(unicorn is not None, "Unicorn is required for ARM comparison")
    def test_comparison_against_unicorn(self) -> None:
        for name, (thumb, hex_code, result) in INSTRUCTIONS.items():
            if result != "fpscr":
                continue
            is64 = name.endswith("64")
            cases = ([(0xbff0000000000000, 0x3ff0000000000000),
                      (0x7ff8000000000000, 0x3ff0000000000000),
                      (0x7ff0000000000001, 0x3ff0000000000000)] if is64 else
                     [(0xbf800000, 0x3f800000), (0x7fc00000, 0x3f800000),
                      (0x7f800001, 0x3f800000)])
            generator = random.Random(0x6e3935)
            cases += [(generator.getrandbits(64 if is64 else 32),
                       generator.getrandbits(64 if is64 else 32)) for _ in range(12)]
            for a, b in cases:
                machine = unicorn.Uc(unicorn.UC_ARCH_ARM,
                                     unicorn.UC_MODE_THUMB if thumb else unicorn.UC_MODE_ARM)
                machine.mem_map(0x1000, 0x1000)
                machine.mem_write(0x1000, bytes.fromhex(hex_code))
                machine.reg_write(arm_const.UC_ARM_REG_C1_C0_2, 0xf00000)
                machine.reg_write(arm_const.UC_ARM_REG_FPEXC, 0x40000000)
                machine.reg_write(arm_const.UC_ARM_REG_FPSCR, 0)
                machine.reg_write(arm_const.UC_ARM_REG_D0 if is64 else arm_const.UC_ARM_REG_S0, a)
                machine.reg_write(arm_const.UC_ARM_REG_D1 if is64 else arm_const.UC_ARM_REG_S1, b)
                machine.emu_start(0x1001 if thumb else 0x1000, 0x1004)
                with self.subTest(name=name, a=hex(a), b=hex(b)):
                    self.assertEqual(self.call(name, a, b), machine.reg_read(arm_const.UC_ARM_REG_FPSCR))

    @unittest.skipUnless(unicorn is not None, "Unicorn is required for ARM comparison")
    def test_register_aliases_against_unicorn(self) -> None:
        for name, (thumb, hex_code, _result) in SEQUENCES.items():
            a, b = ((0x1122334455667788, 0) if "double_to_halves" in name else
                    ((0x55667788, 0x11223344) if "upper_halves" in name
                     else (0x3f800000, 0x40000000)))
            machine = unicorn.Uc(unicorn.UC_ARCH_ARM,
                                 unicorn.UC_MODE_THUMB if thumb else unicorn.UC_MODE_ARM)
            code = bytes.fromhex(hex_code)
            machine.mem_map(0x1000, 0x1000)
            machine.mem_write(0x1000, code)
            machine.reg_write(arm_const.UC_ARM_REG_C1_C0_2, 0xf00000)
            machine.reg_write(arm_const.UC_ARM_REG_FPEXC, 0x40000000)
            if "double_to_halves" in name:
                machine.reg_write(arm_const.UC_ARM_REG_D1, a)
            else:
                machine.reg_write(arm_const.UC_ARM_REG_R0, a)
                machine.reg_write(arm_const.UC_ARM_REG_R1, b)
            machine.emu_start(0x1001 if thumb else 0x1000, 0x1000 + len(code))
            if "double_to_halves" in name:
                expected = (machine.reg_read(arm_const.UC_ARM_REG_R3) << 32) | machine.reg_read(arm_const.UC_ARM_REG_R2)
            elif "upper_halves" in name:
                expected = machine.reg_read(arm_const.UC_ARM_REG_D14)
            else:
                expected = machine.reg_read(arm_const.UC_ARM_REG_D1)
            with self.subTest(name=name):
                self.assertEqual(self.call(name, a, b), expected)

    @unittest.skipUnless(unicorn is not None, "Unicorn is required for ARM comparison")
    def test_conversions_against_unicorn_at_default_rounding(self) -> None:
        samples32 = [0x3fc00000, 0x80000000, 0x00000001, 0x7f800000, 0x7fc00001]
        samples64 = [0xbff8000000000000, 0x8000000000000000, 0x0000000000000001,
                     0x7ff0000000000000, 0x7ff8000000000001, 0x47f0000000000000]
        for mode in ("arm", "thumb"):
            for kind, samples in (("f32_to_f64", samples32), ("f64_to_f32", samples64)):
                name = f"{mode}_{kind}"
                thumb, hex_code, _result = INSTRUCTIONS[name]
                for value in samples:
                    machine = unicorn.Uc(unicorn.UC_ARCH_ARM,
                                         unicorn.UC_MODE_THUMB if thumb else unicorn.UC_MODE_ARM)
                    machine.mem_map(0x1000, 0x1000)
                    machine.mem_write(0x1000, bytes.fromhex(hex_code))
                    machine.reg_write(arm_const.UC_ARM_REG_C1_C0_2, 0xf00000)
                    machine.reg_write(arm_const.UC_ARM_REG_FPEXC, 0x40000000)
                    machine.reg_write(arm_const.UC_ARM_REG_FPSCR, 0)
                    machine.reg_write(arm_const.UC_ARM_REG_S1 if kind == "f32_to_f64"
                                      else arm_const.UC_ARM_REG_D1, value)
                    machine.emu_start(0x1001 if thumb else 0x1000, 0x1004)
                    expected = machine.reg_read(arm_const.UC_ARM_REG_D0 if kind == "f32_to_f64"
                                                else arm_const.UC_ARM_REG_S0)
                    with self.subTest(name=name, value=hex(value)):
                        self.assertEqual(self.call(name, 0, value), expected)


if __name__ == "__main__":
    unittest.main()
