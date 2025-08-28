//! Model graph validation functions
//! モデルグラフの検証機能

use crate::formats::pytorch::TensorData;
use std::collections::{HashMap, HashSet};

use super::errors::ParsingError;
use super::types::{LayerType, ModelGraph};

/// Model validation functionality
pub struct ModelValidator;

impl ModelValidator {
    /// Validate model graph for consistency
    /// モデルグラフの一貫性を検証
    pub fn validate_graph(graph: &ModelGraph) -> Result<(), ParsingError> {
        // Check for cycles
        Self::check_cycles(graph)?;

        // Check layer compatibility
        Self::check_layer_compatibility(graph)?;

        // Ensure all referenced layers exist
        for (from_layer, to_layers) in &graph.connections {
            if !graph.layers.contains_key(from_layer) {
                return Err(ParsingError::MissingConnection(format!(
                    "Source layer '{}' not found",
                    from_layer
                )));
            }

            for to_layer in to_layers {
                if !graph.layers.contains_key(to_layer) {
                    return Err(ParsingError::MissingConnection(format!(
                        "Target layer '{}' not found",
                        to_layer
                    )));
                }
            }
        }

        Ok(())
    }

    /// Check for cycles in the model graph
    /// モデルグラフの循環をチェック
    fn check_cycles(graph: &ModelGraph) -> Result<(), ParsingError> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn dfs_cycle_check(
            node: &str,
            connections: &HashMap<String, Vec<String>>,
            visited: &mut HashSet<String>,
            rec_stack: &mut HashSet<String>,
        ) -> Result<(), String> {
            visited.insert(node.to_string());
            rec_stack.insert(node.to_string());

            if let Some(neighbors) = connections.get(node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        dfs_cycle_check(neighbor, connections, visited, rec_stack)?;
                    } else if rec_stack.contains(neighbor) {
                        return Err(format!("Cycle detected: {} -> {}", node, neighbor));
                    }
                }
            }

            rec_stack.remove(node);
            Ok(())
        }

        for layer in graph.layers.keys() {
            if !visited.contains(layer) {
                if let Err(cycle_info) =
                    dfs_cycle_check(layer, &graph.connections, &mut visited, &mut rec_stack)
                {
                    return Err(ParsingError::CircularDependency(cycle_info));
                }
            }
        }

        Ok(())
    }

    /// Check layer dimension compatibility
    /// レイヤー次元の互換性をチェック
    fn check_layer_compatibility(graph: &ModelGraph) -> Result<(), ParsingError> {
        for (from_layer, to_layers) in &graph.connections {
            let from_info = &graph.layers[from_layer];

            for to_layer in to_layers {
                let to_info = &graph.layers[to_layer];

                // Check if output shape of from_layer matches input shape of to_layer
                if let (Some(output_shape), Some(input_shape)) =
                    (&from_info.output_shape, &to_info.input_shape)
                {
                    if !Self::shapes_compatible(output_shape, input_shape) {
                        return Err(ParsingError::IncompatibleDimensions {
                            layer1: from_layer.clone(),
                            layer2: to_layer.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if two shapes are compatible for layer connection
    /// 2つの形状がレイヤー接続に対して互換性があるかチェック
    fn shapes_compatible(output_shape: &[usize], input_shape: &[usize]) -> bool {
        // Simple compatibility check - in practice, this would be more sophisticated
        // considering transformations like flatten, reshape, etc.

        if output_shape.len() == 1 && input_shape.len() == 1 {
            // Both are 1D - must match exactly
            output_shape[0] == input_shape[0]
        } else if output_shape.len() > 1 && input_shape.len() == 1 {
            // Output is multi-dimensional, input expects 1D (e.g., after flatten)
            let output_size: usize = output_shape.iter().product();
            output_size == input_shape[0]
        } else {
            // For now, assume compatible if we can't determine
            true
        }
    }

    /// Validate that all referenced layers exist
    /// 参照されるすべてのレイヤーが存在することを検証
    pub fn validate_layer_references(
        desc: &super::formats::ArchitectureDescription,
        layers: &HashMap<String, super::types::LayerInfo>,
    ) -> Result<(), ParsingError> {
        let layer_names: HashSet<String> = desc.layers.iter().map(|l| l.name.clone()).collect();

        for connection in &desc.connections {
            if !layer_names.contains(&connection.from) {
                return Err(ParsingError::MissingConnection(format!(
                    "Connection references unknown source layer: {}",
                    connection.from
                )));
            }

            if !layer_names.contains(&connection.to) {
                return Err(ParsingError::MissingConnection(format!(
                    "Connection references unknown target layer: {}",
                    connection.to
                )));
            }
        }

        Ok(())
    }
}
