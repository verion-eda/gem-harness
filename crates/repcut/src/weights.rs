// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
use aig::{AIG, DriverType};
use staging::StagedAIG;

pub(crate) struct EndpointWeights;

impl EndpointWeights {
    pub(crate) fn from_staged_aig(
        aig: &AIG,
        staged: &StagedAIG,
        num_endpoint_groups: usize,
    ) -> Vec<u64> {
        let range = 0..num_endpoint_groups;
        let endpoint_pins_all: Vec<_> = range
            .clone()
            .flat_map(|endpt_i| {
                staged.get_endpoint_group(aig, endpt_i).inputs()
            })
            .collect();

        let mut node_weights = vec![0.0f32; aig.num_aigpins + 1];
        let order_all = aig.topo_traverse_generic(
            Some(&endpoint_pins_all),
            staged.primary_inputs.as_ref(),
        );
        for &i in &order_all {
            node_weights[i] = 1.;
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if (a >> 1) != 0 {
                    node_weights[i] += node_weights[a >> 1]
                        / ((aig.fanouts_start[(a >> 1) + 1]
                            - aig.fanouts_start[a >> 1])
                            as f32);
                }
                if (b >> 1) != 0 {
                    node_weights[i] += node_weights[b >> 1]
                        / ((aig.fanouts_start[(b >> 1) + 1]
                            - aig.fanouts_start[b >> 1])
                            as f32);
                }
            }
        }

        let num_fanouts_to_endpt = {
            let mut num_fanouts_to_endpt = vec![0usize; aig.num_aigpins + 1];
            for &i in &endpoint_pins_all {
                num_fanouts_to_endpt[i] += 1;
            }
            num_fanouts_to_endpt
        };

        range
            .map(|endpt_i| {
                staged
                    .get_endpoint_group(aig, endpt_i)
                    .inputs()
                    .map(|i| node_weights[i] / (num_fanouts_to_endpt[i] as f32))
                    .sum::<f32>()
                    .round() as u64
            })
            .collect()
    }
}
