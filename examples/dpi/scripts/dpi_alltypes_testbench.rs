// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// All-types DPI testbench.
//
// Verifies that every supported DPI-C data type (byte, shortint, int,
// longint, chandle, string, real, shortreal) works correctly through the
// preprocessing → synthesis → compile → simulate pipeline.
// Also tests multi-argument DPI functions (pack_wide, narrow_add).

use simulator::{SimulatorBuilder, signals::SignalPath};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::info!("=== DPI All-Types Testbench ===");

    let use_dpi_lib = std::env::args().any(|a| a == "--dpi-lib");

    let mut builder = SimulatorBuilder::new(
        "examples/dpi/build_alltypes/gatelevel.gv".into(),
        "examples/dpi/build_alltypes/result.gemparts".into(),
        2,
    )
    .cpu();

    if use_dpi_lib {
        tracing::info!("Using C shared library for DPI callbacks");
        let ext = std::env::consts::DLL_EXTENSION;
        builder = builder.register_dpi_lib(format!(
            "examples/dpi/build_alltypes/libdpi_alltypes.{ext}"
        ))?;
    } else {
        tracing::info!("Using Rust closures for DPI callbacks");

        // identity_byte: 1 word in (8-bit), 1 word out
        builder =
            builder.register_dpi("identity_byte", |args| vec![args[0] & 0xFF]);

        // identity_shortint: 1 word in (16-bit), 1 word out
        builder = builder
            .register_dpi("identity_shortint", |args| vec![args[0] & 0xFFFF]);

        // identity_int: 1 word in (32-bit), 1 word out
        builder = builder.register_dpi("identity_int", |args| vec![args[0]]);

        // identity_longint: 2 words in (64-bit LE), 2 words out
        builder = builder
            .register_dpi("identity_longint", |args| vec![args[0], args[1]]);

        // identity_chandle: 2 words in (64-bit LE), 2 words out
        builder = builder
            .register_dpi("identity_chandle", |args| vec![args[0], args[1]]);

        // identity_string: 2 words in (64-bit LE), 2 words out
        builder = builder
            .register_dpi("identity_string", |args| vec![args[0], args[1]]);

        // scale_real: 2 words in (64-bit LE double), 2 words out — doubles it
        builder = builder.register_dpi("scale_real", |args| {
            let bits = (args[0] as u64) | ((args[1] as u64) << 32);
            let val = f64::from_bits(bits) * 2.0;
            let out = val.to_bits();
            vec![out as u32, (out >> 32) as u32]
        });

        // scale_shortreal: 1 word in (32-bit float), 1 word out — doubles it
        builder = builder.register_dpi("scale_shortreal", |args| {
            let val = f32::from_bits(args[0]) * 2.0;
            vec![val.to_bits()]
        });

        // pack_wide: 3 words in (byte, shortint, int), 2 words out (longint LE)
        builder = builder.register_dpi("pack_wide", |args| {
            let a = args[0] & 0xFF;
            let b = args[1] & 0xFFFF;
            let c = args[2];
            let result: u64 =
                (a as u64) | ((b as u64) << 8) | ((c as u64) << 24);
            vec![result as u32, (result >> 32) as u32]
        });

        // narrow_add: 2 words in (shortint, shortint), 1 word out (byte)
        builder = builder.register_dpi("narrow_add", |args| {
            let x = args[0] as u16;
            let y = args[1] as u16;
            vec![(x.wrapping_add(y) & 0xFF) as u32]
        });
    }

    let mut dut = builder.build()?;

    tracing::info!("Inputs:  {:?}", dut.list_inputs());
    tracing::info!("Outputs: {:?}", dut.list_outputs());

    dut.reset();

    let mut passed = 0;
    let mut failed = 0;

    // --- Test byte identity ---
    {
        let val: u8 = 0xAB;
        let sp: SignalPath = "in_byte".parse()?;
        dut.poke_signal(&sp, val as u64)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_byte".parse()?;
        let out = dut.peek_signal(&sp)?;
        check("byte identity", *out, val as u64, &mut passed, &mut failed);
    }

    // --- Test shortint identity ---
    {
        let val: u16 = 0xBEEF;
        let sp: SignalPath = "in_shortint".parse()?;
        dut.poke_signal(&sp, val as u64)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_shortint".parse()?;
        let out = dut.peek_signal(&sp)?;
        check(
            "shortint identity",
            *out,
            val as u64,
            &mut passed,
            &mut failed,
        );
    }

    // --- Test int identity ---
    {
        let val: u32 = 0xDEAD_BEEF;
        let sp: SignalPath = "in_int".parse()?;
        dut.poke_signal(&sp, val as u64)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_int".parse()?;
        let out = dut.peek_signal(&sp)?;
        check("int identity", *out, val as u64, &mut passed, &mut failed);
    }

    // --- Test longint identity ---
    {
        let val: u64 = 0x0123_4567_89AB_CDEF;
        let sp: SignalPath = "in_longint".parse()?;
        dut.poke_signal(&sp, val)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_longint".parse()?;
        let out = dut.peek_signal(&sp)?;
        check("longint identity", *out, val, &mut passed, &mut failed);
    }

    // --- Test chandle identity ---
    {
        let val: u64 = 0xCAFE_BABE_1234_5678;
        let sp: SignalPath = "in_chandle".parse()?;
        dut.poke_signal(&sp, val)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_chandle".parse()?;
        let out = dut.peek_signal(&sp)?;
        check("chandle identity", *out, val, &mut passed, &mut failed);
    }

    // --- Test string identity ---
    {
        let val: u64 = 0xDEAD_CAFE_0042_0001;
        let sp: SignalPath = "in_string".parse()?;
        dut.poke_signal(&sp, val)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_string".parse()?;
        let out = dut.peek_signal(&sp)?;
        check("string identity", *out, val, &mut passed, &mut failed);
    }

    // --- Test real (double) scaling ---
    {
        let val: f64 = 3.14;
        let expected = val * 2.0;
        let sp: SignalPath = "in_real".parse()?;
        dut.poke_signal(&sp, val.to_bits())?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_real".parse()?;
        let out_bits = dut.peek_signal(&sp)?;
        let out_val = f64::from_bits(*out_bits);
        let ok = (out_val - expected).abs() < 1e-10;
        if ok {
            tracing::info!("real scale: {val} * 2 = {out_val} OK");
            passed += 1;
        } else {
            tracing::error!(
                "real scale: {val} * 2 = {out_val} (expected {expected})"
            );
            failed += 1;
        }
    }

    // --- Test shortreal (float) scaling ---
    {
        let val: f32 = 1.5;
        let expected = val * 2.0;
        let sp: SignalPath = "in_shortreal".parse()?;
        dut.poke_signal(&sp, val.to_bits() as u64)?;
        clock_tick(&mut dut)?;
        let sp: SignalPath = "out_shortreal".parse()?;
        let out_bits = dut.peek_signal(&sp)?;
        let out_val = f32::from_bits(*out_bits as u32);
        let ok = (out_val - expected).abs() < 1e-6;
        if ok {
            tracing::info!("shortreal scale: {val} * 2 = {out_val} OK");
            passed += 1;
        } else {
            tracing::error!(
                "shortreal scale: {val} * 2 = {out_val} (expected {expected})"
            );
            failed += 1;
        }
    }

    // --- Test pack_wide (multi-arg: byte + shortint + int -> longint) ---
    {
        let pack_a: SignalPath = "pack_a".parse()?;
        let pack_b: SignalPath = "pack_b".parse()?;
        let pack_c: SignalPath = "pack_c".parse()?;
        let out_packed: SignalPath = "out_packed".parse()?;
        let pack_tests: Vec<(u8, u16, u32)> = vec![
            (0x12, 0x3456, 0x789A_BCDE),
            (0xFF, 0xFFFF, 0xFFFF_FFFF),
            (0x00, 0x0000, 0x0000_0000),
        ];
        for (a, b, c) in pack_tests {
            dut.poke_signal(&pack_a, a as u64)?;
            dut.poke_signal(&pack_b, b as u64)?;
            dut.poke_signal(&pack_c, c as u64)?;
            clock_tick(&mut dut)?;
            let out = dut.peek_signal(&out_packed)?;
            let expected: u64 =
                (a as u64) | ((b as u64) << 8) | ((c as u64) << 24);
            check(
                &format!("pack_wide(0x{a:02X}, 0x{b:04X}, 0x{c:08X})"),
                *out,
                expected,
                &mut passed,
                &mut failed,
            );
        }
    }

    // --- Test narrow_add (multi-arg: shortint + shortint -> byte) ---
    {
        let narrow_x: SignalPath = "narrow_x".parse()?;
        let narrow_y: SignalPath = "narrow_y".parse()?;
        let out_narrow: SignalPath = "out_narrow".parse()?;
        let narrow_tests: Vec<(u16, u16)> =
            vec![(0xCDEF, 0x1234), (0xFFFF, 0x0001), (0x0000, 0x0000)];
        for (x, y) in narrow_tests {
            dut.poke_signal(&narrow_x, x as u64)?;
            dut.poke_signal(&narrow_y, y as u64)?;
            clock_tick(&mut dut)?;
            let out = dut.peek_signal(&out_narrow)?;
            let expected: u64 = (x.wrapping_add(y) & 0xFF) as u64;
            check(
                &format!("narrow_add(0x{x:04X}, 0x{y:04X})"),
                *out,
                expected,
                &mut passed,
                &mut failed,
            );
        }
    }

    tracing::info!("=== Results: {passed}/{} passed ===", passed + failed);

    if failed == 0 {
        tracing::info!("ALL TESTS PASSED");
        Ok(())
    } else {
        anyhow::bail!("{failed} test(s) failed");
    }
}

fn clock_tick(
    dut: &mut simulator::InteractiveSimulator,
) -> anyhow::Result<()> {
    let clk: SignalPath = "clk".parse()?;
    dut.poke_signal(&clk, 0)?;
    dut.step();
    dut.poke_signal(&clk, 1)?;
    dut.step();
    Ok(())
}

fn check(
    name: &str,
    got: u64,
    expected: u64,
    passed: &mut u32,
    failed: &mut u32,
) {
    if got == expected {
        tracing::info!("{name}: 0x{got:X} OK");
        *passed += 1;
    } else {
        tracing::error!("{name}: got 0x{got:X}, expected 0x{expected:X}");
        *failed += 1;
    }
}
