// SPDX-FileCopyrightText: Copyright (c) Kevin Mehall, Zizheng Guo. Originally from eda-infra-rs (https://github.com/gzz2000/eda-infra-rs).
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
// SPDX-License-Identifier: MIT
use std::io;

/// A simple demo that uses the reader and writer to round-trip a VCD file from
/// stdin to stdout
pub fn main() {
    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut reader = vcd_ng::Parser::new(&mut stdin);
    let mut writer = vcd_ng::Writer::new(&mut stdout);

    let header = reader.parse_header().unwrap();
    writer.header(&header).unwrap();

    for cmd in reader {
        writer.command(&cmd.unwrap()).unwrap();
    }
}
