from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# These tests exercise the dispatcher and do not decode firmware instructions.
try:
    import pypcode  # noqa: F401
except ImportError:
    with patch.dict(sys.modules, {"pypcode": types.ModuleType("pypcode")}):
        from tools import extend_prime_aot as extender
else:
    from tools import extend_prime_aot as extender


SYNTHETIC_C = """\
#include <stdint.h>
static uint32_t nokia_frontend_last_pc;
static uint64_t nokia_mem_load(void *machine, uint32_t address, int size) {
    (void)machine; (void)size;
    return address == 0x53000010u ? 17 : 25;
}
static int run(uint64_t start) {
    int machine = 0;
    uint64_t reg_pc=0;
    uint64_t reg_r0=0;
    uint64_t reg_lr=0;
    reg_pc=start; reg_lr=0xfffffffeu;
    if(!reg_pc)reg_pc=UINT64_C(4096);
dispatch:
    if(reg_pc==0xfffffffeu)goto finished;
    switch((uint32_t)reg_pc&~1u){
    case 0x8019db50u: reg_r0=nokia_mem_load(&machine,0x53000010u,4);reg_pc=reg_lr;goto dispatch;
    case 0x8019db88u: reg_r0=nokia_mem_load(&machine,0x53000018u,4);reg_pc=reg_lr;goto dispatch;
    case 0x00001000u: goto L_00001000;
    case 0x00001004u: goto L_00001004;
    case 0x8019db50u: goto L_8019db50;
    case 0x8019db88u: goto L_8019db88;
    default: goto yielded;
    }
L_8019db50: reg_r0=nokia_mem_load(&machine,0x53000010u,4);reg_pc=reg_lr;goto dispatch;
L_8019db88: reg_r0=nokia_mem_load(&machine,0x53000018u,4);reg_pc=reg_lr;goto dispatch;
L_00001000:
    goto L_8019db50;
L_00001004:
    goto L_8019db88;
L_8019db50:
    nokia_frontend_last_pc=0x8019db50u;
    reg_r0=111; goto finished;
L_8019db88:
    nokia_frontend_last_pc=0x8019db88u;
    reg_r0=222; goto finished;
finished:
    return (int)reg_r0;
yielded:
    return -1;
}
int main(void) {
    return !(run(0x1000u)==17 && run(0x1004u)==25 &&
             run(0x8019db50u)==17 && run(0x8019db88u)==25);
}
"""


class FrontendRepairTest(unittest.TestCase):
    def assert_compiles_and_runs(self, source: str) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("a C compiler is required for this regression")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.c"
            executable = Path(directory) / "synthetic"
            path.write_text(source, encoding="utf-8")
            subprocess.run([compiler, "-std=c11", str(path), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)

    def test_repairs_existing_collisions_and_preserves_hook_behavior(self) -> None:
        repaired = extender.normalize_frontend_hooks(SYNTHETIC_C)
        self.assertEqual(extender.normalize_frontend_hooks(repaired), repaired)
        self.assertEqual(extender.frontend_labels(repaired),
                         {0x1000, 0x1004, 0x8019DB50, 0x8019DB88})
        for address in ("8019db50", "8019db88"):
            self.assertEqual(repaired.count(f"case 0x{address}u:"), 1)
            self.assertEqual(repaired.count(f"L_{address}:"), 1)
        self.assert_compiles_and_runs(repaired)

    def test_ambiguous_hook_and_unknown_duplicates_are_rejected(self) -> None:
        ambiguous = SYNTHETIC_C.replace("L_8019db50: reg_r0=", "L_8019db50: reg_r1=", 1)
        with self.assertRaisesRegex(ValueError, "ambiguous executive"):
            extender.normalize_frontend_hooks(ambiguous)
        other = SYNTHETIC_C.replace("L_00001004:\n", "L_00001000:\n", 1)
        with self.assertRaisesRegex(ValueError, "duplicate frontend labels"):
            extender.normalize_frontend_hooks(other)
        other = SYNTHETIC_C.replace("case 0x00001004u:", "case 0x00001000u:", 1)
        with self.assertRaisesRegex(ValueError, "duplicate frontend dispatch cases"):
            extender.normalize_frontend_hooks(other)

    def test_extend_does_not_lift_inline_hooks_again(self) -> None:
        trace = {"instructions": [
            {"address": 0x8019DB50, "size": 4, "thumb": False},
            {"address": 0x8019DB88, "size": 4, "thumb": False},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, trace_path, rom, first, second = (root / name for name in
                ("source.c", "trace.json", "rom.bin", "first.c", "second.c"))
            source.write_text(SYNTHETIC_C, encoding="utf-8")
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            with patch.object(extender, "read_rom", return_value=(b"", 0x80000000)), \
                 patch.object(extender, "collect_function", return_value=set()):
                extender.extend(source, trace_path, rom, first, repair_fallthroughs=True)
                extender.extend(first, trace_path, rom, second, repair_fallthroughs=True)
            repaired = second.read_text(encoding="utf-8")
            self.assertEqual(repaired.count("L_8019db50:"), 1)
            self.assertEqual(repaired.count("L_8019db88:"), 1)
            self.assert_compiles_and_runs(repaired)

    def test_repairs_jumps_to_previously_existing_labels(self) -> None:
        source = SYNTHETIC_C.replace("L_00001000:\n    goto L_8019db50;",
                                     "L_00001000:\n    goto unsupported;")
        source = source.replace("L_00001004:\n    goto L_8019db88;",
                                "L_00001004:\n    if (reg_r0) goto unsupported;\n"
                                "    goto L_8019db88;")
        source = source.replace("yielded:\n    return -1;", "yielded:\n    return -1;\n"
                                "unsupported:\n    return -2;")
        trace = {"instructions": [
            {"address": 0x1000, "size": 4, "thumb": False},
            {"address": 0x1004, "size": 4, "thumb": False},
        ]}

        class Context:
            def __init__(self, language: str) -> None:
                pass

            def translate(self, raw: bytes, base_address: int, max_instructions: int):
                if base_address == 0x1004:
                    target = types.SimpleNamespace(space=types.SimpleNamespace(name="ram"),
                                                   offset=0x8019DB50)
                    operation = types.SimpleNamespace(opcode=types.SimpleNamespace(name="CBRANCH"),
                                                      inputs=[target])
                else:
                    operation = types.SimpleNamespace(opcode=types.SimpleNamespace(name="COPY"))
                return types.SimpleNamespace(ops=[operation])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, trace_path, output = (root / name for name in
                                               ("source.c", "trace.json", "output.c"))
            source_path.write_text(source, encoding="utf-8")
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            with patch.object(extender, "read_rom", return_value=(b"", 0)), \
                 patch.object(extender, "collect_function", return_value=set()), \
                 patch.object(extender.pypcode, "Context", Context, create=True):
                extender.extend(source_path, trace_path, root / "rom", output,
                                repair_fallthroughs=True)
            repaired = output.read_text(encoding="utf-8")
            self.assertIn("L_00001000:\n    goto L_00001004;", repaired)
            self.assertIn("if (reg_r0) goto L_8019db50;", repaired)
            self.assert_compiles_and_runs(repaired.replace("run(0x1000u)==17", "run(0x1000u)==25"))

    def test_batch_runner_counts_inline_labels_as_existing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, trace_path, output = (root / name for name in
                                           ("source.c", "trace.json", "output.c"))
            content = extender.normalize_frontend_hooks(SYNTHETIC_C)
            source.write_text(content, encoding="utf-8")
            trace_path.write_text(json.dumps({"instructions": [
                {"address": 0x8019DB50, "size": 4, "thumb": False},
                {"address": 0x8019DB88, "size": 4, "thumb": False},
            ]}), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "tools" / "extend_aot_in_batches.py"
            result = subprocess.run([sys.executable, str(script), "--lifter", str(root / "missing"),
                                     "--source", str(source), "--trace", str(trace_path),
                                     "--rom", str(root / "rom"), "--output", str(output)],
                                    check=True, capture_output=True, text=True)
            self.assertIn("0 addresses still need lifting", result.stdout)
            self.assertEqual(output.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
