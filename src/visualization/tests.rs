//! 可視化機能の統合テスト
//! Visualization functionality integration tests

#[cfg(test)]
mod visualization_integration_tests {
    use super::super::*;
    use crate::models::high_level::TrainingHistory;
    use crate::tensor::Tensor;
    use crate::autograd::Variable;
    use crate::visualization::graph_viz::{GraphNode, GraphEdge, NodeType, NodeStyle, NodeShape, EdgeStyle, LineType, ArrowType};
    use crate::visualization::utils::{ColorPalette, create_dashboard, PlotStatistics, save_plot, resize_svg};
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn test_training_plotter_end_to_end() {
        // 学習履歴のセットアップ
        let mut history = TrainingHistory::<f32>::new();
        
        // 複数エポックのデータを追加
        for epoch in 1..=10 {
            let train_loss = 1.0 - (epoch as f32 * 0.08); // 減少する損失
            let val_loss = train_loss + 0.1; // 検証損失は少し高め
            
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), (epoch as f64 * 0.07 + 0.3).min(0.95));
            metrics.insert("precision".to_string(), (epoch as f64 * 0.06 + 0.4).min(0.92));
            
            history.add_epoch(train_loss, Some(val_loss), metrics);
        }
        
        // プロッターのセットアップ
        let plotter = TrainingPlotter::new();
        
        // 学習曲線の生成
        let svg_result = plotter.plot_training_curves(&history);
        assert!(svg_result.is_ok(), "Failed to generate training curves");
        
        let svg_content = svg_result.unwrap();
        assert!(svg_content.contains("<svg"));
        assert!(svg_content.contains("</svg>"));
        assert!(svg_content.len() > 100); // 適切なサイズのSVG
        
        // メトリクス時系列プロット
        let metrics_result = plotter.plot_metrics_timeline(&history, "accuracy");
        assert!(metrics_result.is_ok(), "Failed to generate metrics timeline");
        
        let metrics_svg = metrics_result.unwrap();
        assert!(metrics_svg.contains("<svg"));
        
        println!("✓ Training plotter integration test passed");
    }

    #[test]
    fn test_tensor_visualizer_comprehensive() {
        let visualizer = TensorVisualizer::new();
        
        // 1Dテンソルのテスト
        let data_1d = vec![1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5];
        let tensor_1d = Tensor::from_vec(data_1d, vec![7]);
        
        let bar_chart_result = visualizer.plot_bar_chart(&tensor_1d);
        assert!(bar_chart_result.is_ok(), "Failed to generate 1D bar chart");
        
        let bar_chart_svg = bar_chart_result.unwrap();
        assert!(bar_chart_svg.contains("rect")); // バーが含まれる
        assert!(bar_chart_svg.contains("class=\"bar\""));
        
        // 2Dテンソルのヒートマップテスト
        let data_2d = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let tensor_2d = Tensor::from_vec(data_2d, vec![3, 3]);
        
        let heatmap_result = visualizer.plot_heatmap(&tensor_2d);
        assert!(heatmap_result.is_ok(), "Failed to generate 2D heatmap");
        
        let heatmap_svg = heatmap_result.unwrap();
        assert!(heatmap_svg.contains("rect")); // ヒートマップセルが含まれる
        assert!(heatmap_svg.contains("class=\"cell\""));
        
        // 3Dテンソルのスライステスト
        let data_3d = (0..24).map(|i| i as f32).collect();
        let tensor_3d = Tensor::from_vec(data_3d, vec![2, 3, 4]);
        
        let slices_result = visualizer.plot_3d_slices(&tensor_3d);
        assert!(slices_result.is_ok(), "Failed to generate 3D slices");
        
        let slices_svg = slices_result.unwrap();
        assert!(slices_svg.contains("rect"));
        assert!(slices_svg.contains("Slice 0"));
        assert!(slices_svg.contains("Slice 1"));
        
        // 統計情報の可視化テスト
        let stats_result = visualizer.plot_statistics(&tensor_2d);
        assert!(stats_result.is_ok(), "Failed to generate statistics");
        
        println!("✓ Tensor visualizer comprehensive test passed");
    }

    #[test]
    fn test_graph_visualizer_creation() {
        let mut visualizer = GraphVisualizer::new();
        
        // テスト用のノードとエッジを追加
        let node1 = GraphNode {
            id: "var_0".to_string(),
            name: "Input".to_string(),
            node_type: NodeType::Variable,
            shape: vec![2, 2],
            position: (100.0, 100.0),
            style: NodeStyle {
                color: "#e3f2fd".to_string(),
                border_color: "#1976d2".to_string(),
                shape: NodeShape::Rectangle,
                size: (80.0, 40.0),
            },
        };
        
        let node2 = GraphNode {
            id: "op_0".to_string(),
            name: "Add".to_string(),
            node_type: NodeType::Operation("add".to_string()),
            shape: vec![2, 2],
            position: (200.0, 150.0),
            style: NodeStyle {
                color: "#e8f5e8".to_string(),
                border_color: "#4caf50".to_string(),
                shape: NodeShape::Ellipse,
                size: (70.0, 35.0),
            },
        };
        
        let edge = GraphEdge {
            from: "var_0".to_string(),
            to: "op_0".to_string(),
            label: Some("forward".to_string()),
            style: EdgeStyle {
                color: "#666".to_string(),
                width: 2.0,
                line_type: LineType::Solid,
                arrow_type: ArrowType::Standard,
            },
        };
        
        visualizer.add_node(node1);
        visualizer.add_node(node2);
        visualizer.add_edge(edge);
        
        // SVG生成テスト
        let svg_result = visualizer.to_svg();
        assert!(svg_result.is_ok(), "Failed to generate computation graph SVG");
        
        let svg_content = svg_result.unwrap();
        assert!(svg_content.contains("<svg"));
        assert!(svg_content.contains("Input"));
        assert!(svg_content.contains("Add"));
        assert!(svg_content.contains("line")); // エッジ
        
        // DOT形式生成テスト
        let dot_result = visualizer.to_dot();
        assert!(dot_result.is_ok(), "Failed to generate DOT format");
        
        let dot_content = dot_result.unwrap();
        assert!(dot_content.contains("digraph ComputationGraph"));
        assert!(dot_content.contains("var_0"));
        assert!(dot_content.contains("op_0"));
        
        // 統計情報テスト
        let stats = visualizer.get_statistics();
        assert_eq!(stats.get("total_nodes").unwrap(), &2);
        assert_eq!(stats.get("total_edges").unwrap(), &1);
        
        println!("✓ Graph visualizer creation test passed");
    }

    #[test]
    fn test_color_palette_functionality() {
        // カテゴリカルカラーパレット
        let categorical_colors = ColorPalette::categorical();
        assert_eq!(categorical_colors.len(), 10);
        assert!(categorical_colors[0].starts_with('#'));
        
        // 異なるインデックスで異なる色を取得
        let color_0 = ColorPalette::get_categorical_color(0);
        let color_1 = ColorPalette::get_categorical_color(1);
        assert_ne!(color_0, color_1);
        
        // インデックスのオーバーフロー処理
        let color_overflow = ColorPalette::get_categorical_color(100);
        assert!(color_overflow.starts_with('#'));
        
        // シーケンシャルカラー
        let color_min = ColorPalette::get_sequential_color(0.0);
        let color_max = ColorPalette::get_sequential_color(1.0);
        let color_mid = ColorPalette::get_sequential_color(0.5);
        
        assert_ne!(color_min, color_max);
        assert_ne!(color_min, color_mid);
        assert_ne!(color_max, color_mid);
        
        println!("✓ Color palette functionality test passed");
    }

    #[test]
    fn test_file_operations() {
        let dir = tempdir().expect("Failed to create temp directory");
        let file_path = dir.path().join("test_plot.svg");
        
        // テストSVGコンテンツ
        let svg_content = r#"<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="100" height="100" fill="red"/>
        </svg>"#;
        
        // SVGファイルの保存
        let save_result = save_plot(svg_content, &file_path, PlotFormat::Svg);
        assert!(save_result.is_ok(), "Failed to save SVG plot");
        
        // ファイルが存在することを確認
        assert!(file_path.exists(), "Plot file was not created");
        
        // ファイル内容の確認
        let saved_content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(saved_content, svg_content);
        
        // HTMLラップ機能のテスト
        let html_path = dir.path().join("test_plot.html");
        let html_result = save_plot(svg_content, &html_path, PlotFormat::Html);
        assert!(html_result.is_ok(), "Failed to save HTML plot");
        
        let html_content = std::fs::read_to_string(&html_path).unwrap();
        assert!(html_content.contains("<!DOCTYPE html>"));
        assert!(html_content.contains(svg_content));
        
        println!("✓ File operations test passed");
    }

    #[test]
    fn test_dashboard_creation() {
        // 複数のプロットでダッシュボードを作成
        let plots = vec![
            ("Training Loss", r#"<svg><rect x="0" y="0" width="100" height="100" fill="blue"/></svg>"#),
            ("Validation Accuracy", r#"<svg><circle cx="50" cy="50" r="25" fill="green"/></svg>"#),
            ("Learning Rate", r#"<svg><line x1="0" y1="0" x2="100" y2="100" stroke="red"/></svg>"#),
        ];
        
        let dashboard_result = create_dashboard(plots.clone());
        assert!(dashboard_result.is_ok(), "Failed to create dashboard");
        
        let dashboard_content = dashboard_result.unwrap();
        
        // ダッシュボードの基本構造を確認
        assert!(dashboard_content.contains("<!DOCTYPE html>"));
        assert!(dashboard_content.contains("RusTorch Visualization Dashboard"));
        
        // 各プロットが含まれることを確認
        for (title, svg) in plots {
            assert!(dashboard_content.contains(title));
            assert!(dashboard_content.contains(svg));
        }
        
        // プロット数が正しく表示されることを確認
        assert!(dashboard_content.contains("3 plots"));
        
        println!("✓ Dashboard creation test passed");
    }

    #[test]
    fn test_plot_statistics() {
        let svg_content = r#"<svg>
            <rect x="0" y="0" width="100" height="100"/>
            <circle cx="50" cy="50" r="25"/>
            <line x1="0" y1="0" x2="100" y2="100"/>
            <text x="50" y="50">Test</text>
        </svg>"#;
        
        let stats = PlotStatistics::new(svg_content, 150);
        
        // 要素数の確認
        assert_eq!(stats.total_elements, 4); // rect + circle + line + text
        
        // ファイルサイズの確認
        assert_eq!(stats.file_size_bytes, svg_content.len());
        
        // 生成時間の確認
        assert_eq!(stats.generation_time_ms, 150);
        
        // フォーマット機能のテスト
        let formatted = stats.format();
        assert!(formatted.contains("Elements: 4"));
        assert!(formatted.contains("150ms"));
        
        println!("✓ Plot statistics test passed");
    }

    #[test]
    fn test_svg_resize() {
        let original_svg = r#"<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="100" height="100" fill="red"/>
        </svg>"#;
        
        let resized_result = resize_svg(original_svg, 800, 600);
        assert!(resized_result.is_ok(), "Failed to resize SVG");
        
        let resized_svg = resized_result.unwrap();
        assert!(resized_svg.contains(r#"width="800""#));
        assert!(resized_svg.contains(r#"height="600""#));
        assert!(resized_svg.contains("rect")); // 元の内容が保持されている
        
        println!("✓ SVG resize test passed");
    }

    #[test]
    fn test_variable_visualization() {
        // Variableからテンソル可視化
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]);
        let variable = Variable::new(tensor, true); // 勾配計算有効
        
        let visualizer = TensorVisualizer::new();
        let variable_viz_result = visualizer.plot_variable(&variable);
        
        assert!(variable_viz_result.is_ok(), "Failed to visualize variable");
        
        let viz_svg = variable_viz_result.unwrap();
        assert!(viz_svg.contains("<svg"));
        assert!(viz_svg.contains("rect")); // ヒートマップセル
        
        println!("✓ Variable visualization test passed");
    }

    #[test]
    fn test_comprehensive_integration() {
        // 完全な統合テスト：学習履歴 + テンソル + グラフ
        
        // 1. 学習履歴の生成と可視化
        let mut history = TrainingHistory::<f32>::new();
        for i in 1..=5 {
            let loss = 1.0 / (i as f32);
            history.add_epoch(loss, Some(loss + 0.1), HashMap::new());
        }
        
        let plotter = TrainingPlotter::with_config(PlotConfig {
            width: 600,
            height: 400,
            dpi: 150,
            chart_type: ChartType::Single,
            background_color: "#ffffff".to_string(),
            font_size: 14,
            line_width: 2.5,
            marker_size: 5.0,
        });
        
        let training_plot = plotter.plot_training_curves(&history).unwrap();
        
        // 2. テンソル可視化
        let tensor_data = (0..16).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(tensor_data, vec![4, 4]);
        
        let tensor_viz = TensorVisualizer::with_config(TensorPlotConfig {
            colormap: ColorMap::Viridis,
            normalize: true,
            aspect: "equal".to_string(),
            title: Some("Sine Wave Heatmap".to_string()),
            show_colorbar: true,
            show_values: false,
            precision: 3,
        });
        
        let tensor_plot = tensor_viz.plot_heatmap(&tensor).unwrap();
        
        // 3. 計算グラフ
        let mut graph_viz = GraphVisualizer::with_layout(GraphLayout::Circular);
        let variable = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]), true);
        graph_viz.build_graph(&variable).unwrap();
        let graph_plot = graph_viz.to_svg().unwrap();
        
        // 4. ダッシュボード作成
        let dashboard_plots = vec![
            ("Training Progress", training_plot.as_str()),
            ("Tensor Visualization", tensor_plot.as_str()),
            ("Computation Graph", graph_plot.as_str()),
        ];
        
        let dashboard = create_dashboard(dashboard_plots).unwrap();
        
        // 結果の検証
        assert!(dashboard.contains("Training Progress"));
        assert!(dashboard.contains("Tensor Visualization"));
        assert!(dashboard.contains("Computation Graph"));
        assert!(dashboard.contains("RusTorch Visualization Dashboard"));
        
        println!("✓ Comprehensive integration test passed");
        println!("✅ All visualization integration tests completed successfully!");
    }
}